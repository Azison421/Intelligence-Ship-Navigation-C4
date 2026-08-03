"""Adaptive, independently reset forward-control calibration for National_Test."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from dataclasses import asdict
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
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.planning.forward_control_profile import (
    STRAIGHT_PROBE_THROTTLES,
    ForwardProbe,
    ForwardProbeSafetySample,
    action_protocol_hash,
    build_forward_control_profile,
    forward_probe_abort_reason,
    initial_turn_probe_controls,
    supplemental_turn_probe_controls,
)
from usvlib4ros.usvRosUtil import USVRosbridgeClient


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "logs"
PERIOD_S = 0.1
BASELINE_S = 2.0
COMMAND_S = 2.0
COAST_S = 5.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=Path,
        help="Reuse completed probes from an earlier calibration log.",
    )
    return parser.parse_args()


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


def _derived_speed(samples) -> tuple[float, ...]:
    values = []
    for (_, first), (_, second) in zip(samples, samples[1:]):
        duration = second.stamp_sim - first.stamp_sim
        if duration > 1e-6:
            values.append(
                math.hypot(second.x - first.x, second.y - first.y)
                / duration
            )
    return tuple(values)


def _derived_yaw_rate(samples) -> tuple[float, ...]:
    values = []
    for (_, first), (_, second) in zip(samples, samples[1:]):
        duration = second.stamp_sim - first.stamp_sim
        if duration > 1e-6:
            delta = (second.yaw - first.yaw + math.pi) % (
                2.0 * math.pi
            ) - math.pi
            values.append(delta / duration)
    return tuple(values)


def _tail_mean(values: tuple[float, ...]) -> float:
    if not values:
        raise RuntimeError("INSUFFICIENT_CALIBRATION_SAMPLES")
    tail = values[len(values) // 2 :]
    return statistics.fmean(tail)


def main() -> int:
    args = _arguments()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = OUTPUT_DIR / f"forward-control-{stamp}.json"
    result: dict[str, object] = {
        "schema_version": "national-test-forward-calibration-v1",
        "contains_sensitive_connection_data": False,
        "verdict": "aborted",
        "stop_reason": "INITIALIZATION_ERROR",
        "limits": {
            "baseline_s": BASELINE_S,
            "command_s": COMMAND_S,
            "maximum_coast_s": COAST_S,
            "start_clearance_m": 3.0,
            "start_laser_m": 3.0,
            "active_clearance_m": 0.5,
            "active_laser_m": 0.8,
            "maximum_displacement_m": 1.5,
        },
        "trials": [],
    }
    probes: list[ForwardProbe] = []
    straight_response: dict[float, bool] = {}
    if args.resume is not None:
        resume_path = args.resume.resolve()
        previous = json.loads(resume_path.read_text(encoding="utf-8"))
        if previous.get("schema_version") != result["schema_version"]:
            raise ValueError("resume calibration schema is incompatible")
        previous_trials = previous.get("trials")
        if not isinstance(previous_trials, list):
            raise ValueError("resume calibration has no completed trials")
        result["trials"] = previous_trials
        result["resumed_from_sha256"] = sha256(
            resume_path.read_bytes()
        ).hexdigest()
        for trial in previous_trials:
            probe = ForwardProbe(
                float(trial["throttle"]),
                float(trial["rudder"]),
                float(trial["speed_mps"]),
                float(trial["yaw_rate_rad_s"]),
                float(trial["coast_distance_m"]),
            )
            probes.append(probe)
            if abs(probe.rudder) <= 1e-12:
                straight_response[probe.throttle] = (
                    probe.speed_mps
                    >= max(
                        0.05,
                        float(trial["baseline_speed_mps"]) + 0.03,
                    )
                )
    data = GlobalData.getInstance()
    bridge = None
    service = None

    def publish(throttle: int, rudder: int) -> None:
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

        def run_trial(throttle: float, rudder: float) -> ForwardProbe:
            result["active_trial"] = {
                "throttle": throttle,
                "rudder": rudder,
            }
            reset_request_time = float(
                getattr(
                    data.device_data,
                    "reset_request_time",
                    0.0,
                )
                or 0.0
            )
            publish(0, 0)
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
                publish(0, 0)
                time.sleep(PERIOD_S)
            pose = service._wait_for_pose()
            route = controller.getRoute()
            if not getattr(route, "points", None):
                raise RuntimeError("ROUTE_UNAVAILABLE")
            context = build_live_route_context(
                route,
                pose,
                session_id=(
                    f"forward-calibration-{stamp}-"
                    f"{int(throttle * 100)}-{int(rudder * 100)}"
                ),
            )
            adapter = LiveInputAdapter(data, context)
            snapshot = context.compiled_map.snapshot
            first_input = adapter.build()
            origin = first_input.vessel_state

            def safety(runtime_input, *, starting: bool) -> None:
                state = runtime_input.vessel_state
                sample = ForwardProbeSafetySample(
                    pose_age_s=runtime_input.pose_age_s,
                    scan_age_s=runtime_input.scan_age_s,
                    clearance_m=snapshot.clearance_at(state),
                    minimum_laser_m=_minimum_valid_laser(runtime_input),
                    displacement_m=math.hypot(
                        state.x - origin.x,
                        state.y - origin.y,
                    ),
                    feedback_ok=(
                        state.is_finite()
                        and runtime_input.work_model == 2
                        and runtime_input.task_status != 0
                    ),
                )
                reason = forward_probe_abort_reason(
                    sample,
                    starting=starting,
                )
                if reason is not None:
                    raise RuntimeError(reason)

            safety(first_input, starting=True)
            trace: list[dict[str, float | str]] = []

            def phase(
                name: str,
                duration_s: float,
                command_throttle: int,
                command_rudder: int,
                *,
                distance_limited: bool = False,
            ):
                samples = []
                started = time.monotonic()
                while time.monotonic() - started < duration_s:
                    publish(command_throttle, command_rudder)
                    runtime_input = adapter.build()
                    state = runtime_input.vessel_state
                    displacement = math.hypot(
                        state.x - origin.x,
                        state.y - origin.y,
                    )
                    if distance_limited and displacement >= 1.2:
                        safety(runtime_input, starting=False)
                        publish(0, 0)
                        break
                    safety(runtime_input, starting=False)
                    elapsed = time.monotonic() - started
                    samples.append((elapsed, state))
                    trace.append(
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
                return tuple(samples)

            baseline = phase("baseline", BASELINE_S, 0, 0)
            command = phase(
                "command",
                COMMAND_S,
                int(round(throttle * 100.0)),
                int(round(rudder * 100.0)),
                distance_limited=True,
            )
            command_end = command[-1][1]
            coast = phase(
                "coast",
                COAST_S,
                0,
                0,
                distance_limited=True,
            )
            publish(0, 0)

            baseline_speed = _tail_mean(_derived_speed(baseline))
            speed = _tail_mean(_derived_speed(command))
            yaw_rate = _tail_mean(_derived_yaw_rate(command))
            coast_distance = (
                0.0
                if not coast
                else math.hypot(
                    coast[-1][1].x - command_end.x,
                    coast[-1][1].y - command_end.y,
                )
            )
            trial = {
                "throttle": throttle,
                "rudder": rudder,
                "baseline_speed_mps": baseline_speed,
                "speed_mps": speed,
                "yaw_rate_rad_s": yaw_rate,
                "coast_distance_m": coast_distance,
                "trace": trace,
            }
            result["trials"].append(trial)
            result["active_trial"] = None
            return ForwardProbe(
                throttle,
                rudder,
                max(0.0, speed),
                yaw_rate,
                coast_distance,
            )

        def run_with_safe_reset(
            throttle: float,
            rudder: float,
        ) -> ForwardProbe:
            for attempt in range(5):
                try:
                    return run_trial(throttle, rudder)
                except RuntimeError as exc:
                    if str(exc) not in ("START_CLEARANCE", "START_LASER"):
                        raise
                    publish(0, 0)
                    result.setdefault("start_retries", []).append(
                        {
                            "throttle": throttle,
                            "rudder": rudder,
                            "attempt": attempt + 1,
                            "reason": str(exc),
                        }
                    )
            raise RuntimeError("NO_SAFE_RESET_POSITION")

        def completed_probe(
            throttle: float,
            rudder: float,
        ) -> ForwardProbe | None:
            return next(
                (
                    probe
                    for probe in probes
                    if abs(probe.throttle - throttle) <= 1e-12
                    and abs(probe.rudder - rudder) <= 1e-12
                ),
                None,
            )

        for throttle in STRAIGHT_PROBE_THROTTLES:
            probe = completed_probe(throttle, 0.0)
            if probe is None:
                probe = run_with_safe_reset(throttle, 0.0)
                probes.append(probe)
                baseline_speed = float(
                    result["trials"][-1]["baseline_speed_mps"]
                )
                straight_response[throttle] = (
                    probe.speed_mps
                    >= max(0.05, baseline_speed + 0.03)
                )

        effective = tuple(
            throttle
            for throttle in STRAIGHT_PROBE_THROTTLES
            if straight_response[throttle]
        )
        if not effective:
            raise RuntimeError("NO_EFFECTIVE_FORWARD_RESPONSE")
        for throttle, rudder in initial_turn_probe_controls(effective):
            if completed_probe(throttle, rudder) is None:
                probes.append(run_with_safe_reset(throttle, rudder))

        initial_profile = build_forward_control_profile(probes)
        for throttle, rudder in supplemental_turn_probe_controls(
            initial_profile
        ):
            if completed_probe(throttle, rudder) is None:
                probes.append(run_with_safe_reset(throttle, rudder))
        profile = build_forward_control_profile(probes)
        result.update(
            {
                "verdict": "calibrated",
                "stop_reason": None,
                "profile": asdict(profile),
                "action_protocol_hash": action_protocol_hash(profile),
            }
        )
    except Exception as exc:
        result["stop_reason"] = f"{type(exc).__name__}:{exc}"
    finally:
        if service is not None:
            for _ in range(3):
                service._publish_zero()
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
    return 0 if result["verdict"] == "calibrated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
