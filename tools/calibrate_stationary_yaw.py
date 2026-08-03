"""Bounded zero-throttle yaw probe for the National_Test vessel."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from usvlib4ros.msg.global_data import GlobalData
from usvlib4ros.navigation.device_action_bridge import (
    create_ros_device_action_bridge,
)
from usvlib4ros.navigation.fixed_map_runtime import (
    LiveInputAdapter,
    build_live_route_context,
)
from usvlib4ros.navigation.stationary_yaw_calibration import (
    CalibrationPhase,
    ProbeSafetySample,
    evaluate_stationary_yaw,
    stationary_yaw_abort_reason,
)
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.usvRosUtil import USVRosbridgeClient


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "logs"
PERIOD_S = 0.1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rudder-percent", type=int, default=10)
    parser.add_argument(
        "--maximum-displacement-m",
        type=float,
        default=0.15,
    )
    args = parser.parse_args()
    if not 1 <= args.rudder_percent <= 100:
        parser.error("--rudder-percent must be in [1, 100]")
    if (
        not math.isfinite(args.maximum_displacement_m)
        or args.maximum_displacement_m <= 0.0
    ):
        parser.error("--maximum-displacement-m must be positive")
    return args


def _minimum_valid_laser(sample) -> float:
    values = [
        value
        for value, valid in zip(
            sample.laser_ranges,
            sample.laser_valid_mask,
        )
        if valid
    ]
    if not values:
        raise RuntimeError("LASER_INVALID")
    return min(values)


def _phase_summary(
    *,
    samples,
    origin,
    snapshot,
) -> tuple[CalibrationPhase, dict[str, object]]:
    maximum_displacement = max(
        math.hypot(state.x - origin.x, state.y - origin.y)
        for state, _ in samples
    )
    phase = CalibrationPhase(
        yaw_rates_rad_s=tuple(
            state.yaw_rate for state, _ in samples[len(samples) // 2 :]
        ),
        displacement_m=maximum_displacement,
        minimum_clearance_m=min(
            snapshot.clearance_at(state) for state, _ in samples
        ),
        minimum_laser_m=min(
            _minimum_valid_laser(runtime_input)
            for _, runtime_input in samples
        ),
    )
    return phase, {
        "sample_count": len(samples),
        "yaw_rates_rad_s": list(phase.yaw_rates_rad_s),
        "maximum_displacement_m": phase.displacement_m,
        "minimum_clearance_m": phase.minimum_clearance_m,
        "minimum_laser_m": phase.minimum_laser_m,
    }


def main() -> int:
    args = _arguments()
    started_at = time.strftime("%Y%m%d-%H%M%S")
    output = OUTPUT_DIR / f"stationary-yaw-{started_at}.json"
    result: dict[str, object] = {
        "schema_version": "national-test-stationary-yaw-v1",
        "rudder_percent": args.rudder_percent,
        "maximum_displacement_m": args.maximum_displacement_m,
        "zero_throttle_only": True,
        "contains_sensitive_connection_data": False,
        "verdict": "aborted",
        "stop_reason": "INITIALIZATION_ERROR",
        "trace": [],
    }
    trace = result["trace"]
    data = GlobalData.getInstance()
    action_bridge = None

    def publish(throttle: int, rudder: int) -> None:
        data.updateThrottleRudderOutput(throttle, rudder, 0.0, 0, 0.0)
        data.updateAlgorithmOutput(0, 0, 0, 0.0, 4_000, 2)
        if action_bridge is not None:
            action_bridge.set_command(throttle, rudder)

    try:
        config = json.loads(
            (PROJECT_ROOT / "config.json").read_text(encoding="utf-8")
        )["ros2"]
        controller = Ros2Controller(
            host=config["host"],
            port=int(config["port"]),
            deviceId=config["deviceId"],
            globalData=data,
        )
        action_bridge = create_ros_device_action_bridge(config["deviceId"])
        action_bridge.start(publish_hz=30.0)

        def wait_for(attribute: str, expected: int, timeout_s: float) -> None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if int(getattr(data.device_data, attribute, 0) or 0) == expected:
                    return
                publish(0, 0)
                time.sleep(PERIOD_S)
            raise TimeoutError(f"{attribute} did not become {expected}")

        def wait_for_reset(initial_request_time: float) -> None:
            deadline = time.monotonic() + 30.0
            observed = False
            while time.monotonic() < deadline:
                device = data.device_data
                status = int(getattr(device, "reset_status", 0) or 0)
                request_time = float(
                    getattr(device, "reset_request_time", 0.0) or 0.0
                )
                observed = observed or status == 1 or (
                    request_time != initial_request_time
                )
                if observed and status == 2:
                    return
                publish(0, 0)
                time.sleep(PERIOD_S)
            raise TimeoutError("Unity reset did not complete")

        publish(0, 0)
        initial_request_time = float(
            getattr(data.device_data, "reset_request_time", 0.0) or 0.0
        )
        if not controller.reset_unity():
            raise RuntimeError("Unity reset request failed")
        wait_for_reset(initial_request_time)
        if not controller.set_auto_work():
            raise RuntimeError("automatic work-mode request failed")
        wait_for("work_model", 2, 10.0)

        route = controller.getRoute()
        pose_deadline = time.monotonic() + 10.0
        pose = None
        while time.monotonic() < pose_deadline:
            pose = getattr(data.scada_data, "pose", None)
            if (
                abs(float(getattr(pose, "lat", 0.0) or 0.0)) > 1e-9
                and abs(float(getattr(pose, "lng", 0.0) or 0.0)) > 1e-9
            ):
                break
            publish(0, 0)
            time.sleep(PERIOD_S)
        else:
            raise TimeoutError("ship pose was not received")

        context = build_live_route_context(
            route,
            pose,
            session_id=f"stationary-yaw-{started_at}",
        )
        adapter = LiveInputAdapter(data, context)
        snapshot = context.compiled_map.snapshot
        origin = None

        def phase(
            name: str,
            duration_s: float,
            rudder: int,
        ):
            nonlocal origin
            samples = []
            phase_started = time.monotonic()
            while time.monotonic() - phase_started < duration_s:
                publish(0, rudder)
                runtime_input = adapter.build()
                state = runtime_input.vessel_state
                if origin is None:
                    origin = state
                clearance = snapshot.clearance_at(state)
                displacement = math.hypot(
                    state.x - origin.x,
                    state.y - origin.y,
                )
                safety_sample = ProbeSafetySample(
                    pose_age_s=runtime_input.pose_age_s,
                    scan_age_s=runtime_input.scan_age_s,
                    clearance_m=clearance,
                    minimum_laser_m=_minimum_valid_laser(runtime_input),
                    displacement_m=displacement,
                )
                trace.append(
                    {
                        "phase": name,
                        "elapsed_s": time.monotonic() - phase_started,
                        "x_m": state.x,
                        "y_m": state.y,
                        "yaw_rad": state.yaw,
                        "yaw_rate_rad_s": state.yaw_rate,
                        "speed_mps": state.speed,
                        "displacement_m": displacement,
                        "clearance_m": clearance,
                        "minimum_laser_m": safety_sample.minimum_laser_m,
                        "pose_age_s": runtime_input.pose_age_s,
                        "scan_age_s": runtime_input.scan_age_s,
                    }
                )
                reason = stationary_yaw_abort_reason(
                    safety_sample,
                    maximum_displacement_m=args.maximum_displacement_m,
                )
                if reason is not None:
                    raise RuntimeError(f"{reason}:{name}")
                samples.append((state, runtime_input))
                time.sleep(PERIOD_S)
            if not samples:
                raise RuntimeError(f"NO_SAMPLES:{name}")
            return samples

        phases = {}
        baseline_samples = phase("baseline", 2.0, 0)
        positive_samples = phase(
            "positive",
            1.0,
            args.rudder_percent,
        )
        phase("positive_settle", 2.0, 0)
        negative_samples = phase(
            "negative",
            1.0,
            -args.rudder_percent,
        )
        phase("negative_settle", 2.0, 0)

        baseline, phases["baseline"] = _phase_summary(
            samples=baseline_samples,
            origin=origin,
            snapshot=snapshot,
        )
        positive, phases["positive"] = _phase_summary(
            samples=positive_samples,
            origin=origin,
            snapshot=snapshot,
        )
        negative, phases["negative"] = _phase_summary(
            samples=negative_samples,
            origin=origin,
            snapshot=snapshot,
        )
        evaluation = evaluate_stationary_yaw(
            baseline=baseline,
            positive=positive,
            negative=negative,
            maximum_displacement_m=args.maximum_displacement_m,
        )
        result.update(
            {
                "verdict": evaluation.verdict,
                "stop_reason": None,
                "evaluation": asdict(evaluation),
                "phases": phases,
                "map_payload_hash": snapshot.payload_content_hash,
            }
        )
        return 0 if evaluation.verdict == "competition_ready" else 2
    except Exception as exc:
        result["stop_reason"] = str(exc).splitlines()[0][:160]
        return 3
    finally:
        for _ in range(6):
            publish(0, 0)
            time.sleep(PERIOD_S)
        if action_bridge is not None:
            action_bridge.close()
        ros = USVRosbridgeClient.ros
        if ros is not None:
            ros.terminate()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"saved: {output}")


if __name__ == "__main__":
    raise SystemExit(main())
