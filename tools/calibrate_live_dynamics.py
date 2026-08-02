"""Bounded Unity actuator probe for the National_Test reduced dynamics."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from usvlib4ros.msg.global_data import GlobalData
from usvlib4ros.navigation.device_action_bridge import (
    create_ros_device_action_bridge,
)
from usvlib4ros.navigation.fixed_map_runtime import (
    LiveInputAdapter,
    build_live_route_context,
)
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.usvRosUtil import USVRosbridgeClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE_THROTTLE_PERCENT = 30
PROBE_RUDDER_PERCENT = 30
PERIOD_S = 0.1


def _mean(values):
    return sum(values) / len(values)


def _derived_speeds(samples):
    values = []
    for first, second in zip(samples, samples[1:]):
        duration = second.stamp_sim - first.stamp_sim
        if duration > 1e-6:
            values.append(
                math.hypot(
                    second.x - first.x,
                    second.y - first.y,
                )
                / duration
            )
    return values


def _derived_yaw_rates(samples):
    values = []
    for first, second in zip(samples, samples[1:]):
        duration = second.stamp_sim - first.stamp_sim
        if duration > 1e-6:
            yaw_delta = (
                second.yaw - first.yaw + math.pi
            ) % (2.0 * math.pi) - math.pi
            values.append(yaw_delta / duration)
    return values


def main() -> int:
    config = json.loads(
        (PROJECT_ROOT / "config.json").read_text(encoding="utf-8")
    )["ros2"]
    data = GlobalData.getInstance()
    controller = Ros2Controller(
        host=config["host"],
        port=int(config["port"]),
        deviceId=config["deviceId"],
        globalData=data,
    )
    action_bridge = create_ros_device_action_bridge(config["deviceId"])
    action_bridge.start(publish_hz=30.0)

    def publish(throttle: int, rudder: int) -> None:
        data.updateThrottleRudderOutput(
            throttle,
            rudder,
            0.0,
            0,
            0.0,
        )
        data.updateAlgorithmOutput(
            0,
            0,
            0,
            0.0,
            4_000,
            2,
        )
        action_bridge.set_command(throttle, rudder)

    def wait_for(attribute: str, expected: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if int(
                getattr(data.device_data, attribute, 0) or 0
            ) == expected:
                return
            publish(0, 0)
            time.sleep(PERIOD_S)
        raise TimeoutError(f"{attribute} did not become {expected}")

    def wait_for_reset(initial_request_time: float) -> None:
        deadline = time.monotonic() + 30.0
        observed_current_request = False
        while time.monotonic() < deadline:
            status = int(
                getattr(data.device_data, "reset_status", 0) or 0
            )
            request_time = float(
                getattr(
                    data.device_data,
                    "reset_request_time",
                    0.0,
                )
                or 0.0
            )
            if status == 1 or request_time != initial_request_time:
                observed_current_request = True
            if observed_current_request and status == 2:
                return
            publish(0, 0)
            time.sleep(PERIOD_S)
        raise TimeoutError("Unity reset did not complete")

    try:
        publish(0, 0)
        reset_request_time = float(
            getattr(
                data.device_data,
                "reset_request_time",
                0.0,
            )
            or 0.0
        )
        if not controller.reset_unity():
            raise RuntimeError("Unity reset request failed")
        wait_for_reset(reset_request_time)
        if not controller.set_auto_work():
            raise RuntimeError("automatic work-mode request failed")
        wait_for("work_model", 2, 10.0)
        route = controller.getRoute()
        deadline = time.monotonic() + 10.0
        pose = None
        while time.monotonic() < deadline:
            pose = getattr(data.scada_data, "pose", None)
            if (
                abs(float(getattr(pose, "lat", 0.0) or 0.0)) > 1e-9
                and abs(float(getattr(pose, "lng", 0.0) or 0.0)) > 1e-9
            ):
                break
            time.sleep(PERIOD_S)
        else:
            raise TimeoutError("ship pose was not received")
        context = build_live_route_context(
            route,
            pose,
            session_id=f"live-calibration-{int(time.time())}",
        )
        adapter = LiveInputAdapter(data, context)

        def phase(
            name: str,
            duration_s: float,
            throttle: int,
            rudder: int,
        ):
            print(
                f"calibration: {name}, "
                f"advise=({throttle},{rudder}), {duration_s:.1f}s"
            )
            samples = []
            started = time.monotonic()
            while time.monotonic() - started < duration_s:
                publish(throttle, rudder)
                sample = adapter.build()
                state = sample.vessel_state
                if (
                    not context.compiled_map.snapshot.is_state_valid(state)
                    or sample.pose_age_s > 0.5
                    or sample.scan_age_s > 1.0
                ):
                    raise RuntimeError(
                        f"unsafe or stale input during {name}"
                    )
                valid_laser = [
                    value
                    for value, valid in zip(
                        sample.laser_ranges,
                        sample.laser_valid_mask,
                    )
                    if valid
                ]
                if valid_laser and min(valid_laser) < 0.6:
                    raise RuntimeError(
                        f"laser emergency stop during {name}"
                    )
                samples.append(state)
                time.sleep(PERIOD_S)
            first = samples[0]
            last = samples[-1]
            print(
                "calibration telemetry: "
                f"local_advise=("
                f"{data.navigation_output_data.advise_throttle},"
                f"{data.navigation_output_data.advise_rudder}), "
                f"device_feedback=("
                f"{getattr(data.device_data, 'throttle_percent', None)},"
                f"{getattr(data.device_data, 'rudder_percent', None)}), "
                f"work={getattr(data.device_data, 'work_model', None)}, "
                f"task={getattr(data.device_data, 'task_status', None)}, "
                f"delta={math.hypot(last.x-first.x,last.y-first.y):.4f}m, "
                f"yaw_delta={((last.yaw-first.yaw+math.pi)%(2*math.pi)-math.pi):.4f}rad"
            )
            return samples

        still = phase("stationary", 2.0, 0, 0)
        throttle = phase(
            "throttle",
            3.0,
            PROBE_THROTTLE_PERCENT,
            0,
        )
        phase("settle", 2.0, 0, 0)
        positive = phase(
            "positive rudder",
            2.0,
            PROBE_THROTTLE_PERCENT,
            PROBE_RUDDER_PERCENT,
        )
        negative = phase(
            "negative rudder",
            2.0,
            PROBE_THROTTLE_PERCENT,
            -PROBE_RUDDER_PERCENT,
        )
        publish(0, 0)

        throttle_tail = throttle[len(throttle) // 2 :]
        positive_tail = positive[len(positive) // 2 :]
        negative_tail = negative[len(negative) // 2 :]
        raw_speed_at_probe = _mean(
            [sample.speed for sample in throttle_tail]
        )
        raw_positive_rate = _mean(
            [sample.yaw_rate for sample in positive_tail]
        )
        raw_negative_rate = _mean(
            [sample.yaw_rate for sample in negative_tail]
        )
        derived_speeds = _derived_speeds(throttle)
        derived_positive_rates = _derived_yaw_rates(positive)
        derived_negative_rates = _derived_yaw_rates(negative)
        speed_at_probe = _mean(
            derived_speeds[len(derived_speeds) // 2 :]
        )
        positive_rate = _mean(
            derived_positive_rates[
                len(derived_positive_rates) // 2 :
            ]
        )
        negative_rate = _mean(
            derived_negative_rates[
                len(derived_negative_rates) // 2 :
            ]
        )
        if speed_at_probe < 0.02:
            raise RuntimeError("no throttle response observed")
        if (
            abs(positive_rate) < math.radians(0.1)
            or abs(negative_rate) < math.radians(0.1)
            or positive_rate * negative_rate >= 0.0
        ):
            raise RuntimeError(
                "bidirectional rudder response was not observed"
            )
        full_scale_speed = speed_at_probe / (
            PROBE_THROTTLE_PERCENT / 100.0
        )
        full_scale_yaw_rate = min(
            abs(positive_rate),
            abs(negative_rate),
        ) / (PROBE_RUDDER_PERCENT / 100.0)
        displacement = math.hypot(
            throttle[-1].x - throttle[0].x,
            throttle[-1].y - throttle[0].y,
        )
        jitter = max(
            math.hypot(
                sample.x - still[0].x,
                sample.y - still[0].y,
            )
            for sample in still
        )
        result = {
            "schema_version": "national-test-live-dynamics-v1",
            "probe_throttle_percent": PROBE_THROTTLE_PERCENT,
            "probe_rudder_percent": PROBE_RUDDER_PERCENT,
            "speed_at_probe_mps": speed_at_probe,
            "raw_scada_speed_at_probe_mps": raw_speed_at_probe,
            "full_scale_speed_estimate_mps": full_scale_speed,
            "positive_rudder_math_yaw_rate_rad_s": positive_rate,
            "negative_rudder_math_yaw_rate_rad_s": negative_rate,
            "raw_scada_positive_yaw_rate_rad_s": raw_positive_rate,
            "raw_scada_negative_yaw_rate_rad_s": raw_negative_rate,
            "full_scale_yaw_rate_estimate_rad_s": (
                full_scale_yaw_rate
            ),
            "positive_output_rudder_sign_in_math_frame": (
                1 if positive_rate > 0.0 else -1
            ),
            "throttle_phase_displacement_m": displacement,
            "stationary_jitter_max_m": jitter,
            "samples": (
                len(still)
                + len(throttle)
                + len(positive)
                + len(negative)
            ),
            "contains_sensitive_connection_data": False,
        }
        output = (
            PROJECT_ROOT
            / "artifacts"
            / "logs"
            / "national_test_live_dynamics.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"saved: {output}")
        return 0
    finally:
        publish(0, 0)
        time.sleep(0.5)
        publish(0, 0)
        action_bridge.close()
        ros = USVRosbridgeClient.ros
        if ros is not None:
            ros.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
