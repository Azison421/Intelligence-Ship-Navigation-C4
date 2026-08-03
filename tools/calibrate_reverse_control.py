"""Bounded open-water probe for negative-throttle capability."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys
import time

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
from usvlib4ros.navigation.fixed_map_service import FixedMapNavigationService
from usvlib4ros.navigation.reverse_control_calibration import (
    evaluate_reverse_response,
)
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.usvRosUtil import USVRosbridgeClient


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "logs"
PERIOD_S = 0.1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--throttle-percent", type=int, default=-20)
    args = parser.parse_args()
    if not -100 <= args.throttle_percent <= -1:
        parser.error("--throttle-percent must be in [-100, -1]")
    return args


def _minimum_valid_laser(runtime_input) -> float:
    values = [
        float(value)
        for value, valid in zip(
            runtime_input.laser_ranges,
            runtime_input.laser_valid_mask,
        )
        if valid and math.isfinite(float(value))
    ]
    if not values:
        raise RuntimeError("LASER_INVALID")
    return min(values)


def _signed_speeds(samples, heading: float) -> tuple[float, ...]:
    values = []
    forward_x = math.cos(heading)
    forward_y = math.sin(heading)
    for (_, first), (_, second) in zip(samples, samples[1:]):
        duration = second.stamp_sim - first.stamp_sim
        if duration <= 1e-6:
            continue
        delta_x = second.x - first.x
        delta_y = second.y - first.y
        values.append(
            (delta_x * forward_x + delta_y * forward_y) / duration
        )
    return tuple(values)


def _tail_mean(values: tuple[float, ...]) -> float:
    if not values:
        raise RuntimeError("INSUFFICIENT_REVERSE_SAMPLES")
    return statistics.fmean(values[len(values) // 2 :])


def main() -> int:
    args = _arguments()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = OUTPUT_DIR / f"reverse-control-{stamp}.json"
    result: dict[str, object] = {
        "schema_version": "national-test-reverse-calibration-v1",
        "contains_sensitive_connection_data": False,
        "command_throttle_percent": args.throttle_percent,
        "verdict": "aborted",
        "stop_reason": "INITIALIZATION_ERROR",
        "limits": {
            "baseline_s": 2.0,
            "command_s": 1.0,
            "settle_s": 1.0,
            "start_clearance_m": 3.0,
            "start_laser_m": 3.0,
            "active_clearance_m": 0.5,
            "active_laser_m": 0.8,
            "maximum_displacement_m": 1.5,
        },
        "trace": [],
    }
    data = GlobalData.getInstance()
    bridge = None
    service = None

    def publish(throttle: int, rudder: int = 0) -> None:
        data.updateThrottleRudderOutput(throttle, rudder, 0.0, 0, 0.0)
        data.updateAlgorithmOutput(0, 0, 0, 0.0, 4_000, 2)
        if bridge is not None:
            bridge.set_command(throttle, rudder)

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
        bridge = create_ros_device_action_bridge(config["deviceId"])
        service = FixedMapNavigationService(
            controller,
            data,
            action_bridge=bridge,
        )
        bridge.start(publish_hz=30.0)
        controller.initParameterList()
        reset_request_time = float(
            getattr(data.device_data, "reset_request_time", 0.0) or 0.0
        )
        publish(0)
        if not controller.reset_unity():
            raise RuntimeError("UNITY_RESET_REQUEST_FAILED")
        if not service._wait_for_reset(
            initial_request_time=reset_request_time,
        ):
            raise TimeoutError("UNITY_RESET_TIMEOUT")
        if not controller.set_auto_work():
            raise RuntimeError("AUTO_MODE_REQUEST_FAILED")
        if not service._wait_for_auto():
            raise TimeoutError("AUTO_MODE_TIMEOUT")
        settle_deadline = time.monotonic() + 1.0
        while time.monotonic() < settle_deadline:
            publish(0)
            time.sleep(PERIOD_S)
        pose = service._wait_for_pose()
        route = controller.getRoute()
        if not getattr(route, "points", None):
            raise RuntimeError("ROUTE_UNAVAILABLE")
        context = build_live_route_context(
            route,
            pose,
            session_id=f"reverse-calibration-{stamp}",
        )
        adapter = LiveInputAdapter(data, context)
        snapshot = context.compiled_map.snapshot
        first_input = adapter.build()
        origin = first_input.vessel_state
        heading = origin.yaw

        def safety(runtime_input, *, starting: bool) -> None:
            state = runtime_input.vessel_state
            clearance = snapshot.clearance_at(state)
            minimum_laser = _minimum_valid_laser(runtime_input)
            displacement = math.hypot(
                state.x - origin.x,
                state.y - origin.y,
            )
            if not state.is_finite() or runtime_input.work_model != 2:
                raise RuntimeError("FEEDBACK_INVALID")
            if runtime_input.pose_age_s > 0.5:
                raise RuntimeError("POSE_STALE")
            if runtime_input.scan_age_s > 1.0:
                raise RuntimeError("SCAN_STALE")
            if starting and minimum_laser < 3.0:
                raise RuntimeError("START_LASER")
            if starting and clearance < 3.0:
                raise RuntimeError("START_CLEARANCE")
            if not starting and minimum_laser < 0.8:
                raise RuntimeError("LASER_STOP")
            if not starting and clearance <= 0.5:
                raise RuntimeError("CLEARANCE_STOP")
            if displacement > 1.5:
                raise RuntimeError("DISPLACEMENT_LIMIT")

        safety(first_input, starting=True)

        def phase(name: str, duration_s: float, throttle: int):
            samples = []
            started = time.monotonic()
            while time.monotonic() - started < duration_s:
                publish(throttle)
                runtime_input = adapter.build()
                safety(runtime_input, starting=False)
                state = runtime_input.vessel_state
                elapsed = time.monotonic() - started
                samples.append((elapsed, state))
                result["trace"].append(
                    {
                        "phase": name,
                        "elapsed_s": elapsed,
                        "x_m": state.x,
                        "y_m": state.y,
                        "yaw_rad": state.yaw,
                        "speed_mps": state.speed,
                        "yaw_rate_rad_s": state.yaw_rate,
                        "clearance_m": snapshot.clearance_at(state),
                        "minimum_laser_m": _minimum_valid_laser(
                            runtime_input
                        ),
                        "pose_age_s": runtime_input.pose_age_s,
                        "scan_age_s": runtime_input.scan_age_s,
                    }
                )
                time.sleep(PERIOD_S)
            if len(samples) < 2:
                raise RuntimeError(f"INSUFFICIENT_SAMPLES:{name}")
            return tuple(samples)

        baseline = phase("baseline", 2.0, 0)
        command = phase("command", 1.0, args.throttle_percent)
        phase("settle", 1.0, 0)
        publish(0)
        baseline_speed = _tail_mean(_signed_speeds(baseline, heading))
        command_speed = _tail_mean(_signed_speeds(command, heading))
        evaluation = evaluate_reverse_response(
            baseline_signed_speed_mps=baseline_speed,
            command_signed_speed_mps=command_speed,
        )
        result.update(
            {
                "verdict": evaluation.verdict,
                "stop_reason": None,
                "evaluation": asdict(evaluation),
                "map_payload_hash": snapshot.payload_content_hash,
            }
        )
    except Exception as exc:
        result["stop_reason"] = f"{type(exc).__name__}:{exc}"
    finally:
        for _ in range(6):
            publish(0)
            time.sleep(0.05)
        if bridge is not None:
            bridge.close()
        ros = USVRosbridgeClient.ros
        if ros is not None:
            ros.terminate()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(str(target))
    return 0 if result["verdict"] == "reverse_supported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
