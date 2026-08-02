"""Read-only National_Test live gate and policy probe.

The official controller still publishes its default zero NavigationStatus,
but this probe never resets Unity, switches work mode, or publishes a
non-zero recommendation.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

from usvlib4ros.msg.global_data import GlobalData
from usvlib4ros.navigation.fixed_map_runtime import (
    DEFAULT_CHECKPOINT,
    FixedMapControllerCore,
    LiveInputAdapter,
    build_live_route_context,
    load_live_ready_policy,
)
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.usvRosUtil import USVRosbridgeClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = json.loads(
        (PROJECT_ROOT / "config.json").read_text(encoding="utf-8")
    )["ros2"]
    global_data = GlobalData.getInstance()
    controller = Ros2Controller(
        host=config["host"],
        port=int(config["port"]),
        deviceId=config["deviceId"],
        globalData=global_data,
    )
    try:
        deadline = time.monotonic() + 15.0
        route = controller.getRoute()
        while time.monotonic() < deadline:
            pose = getattr(global_data.scada_data, "pose", None)
            points = getattr(route, "points", None) or ()
            if (
                points
                and abs(float(getattr(pose, "lat", 0.0) or 0.0)) > 1e-9
                and abs(float(getattr(pose, "lng", 0.0) or 0.0)) > 1e-9
            ):
                break
            time.sleep(0.1)
            route = controller.getRoute()
        else:
            raise TimeoutError("live route or ship pose was not received")

        context = build_live_route_context(
            route,
            pose,
            session_id=f"shadow-live-{int(time.time())}",
        )
        policy = load_live_ready_policy(DEFAULT_CHECKPOINT, context)
        adapter = LiveInputAdapter(global_data, context)
        time.sleep(0.2)
        sample = adapter.build()
        hypothetical = replace(
            sample,
            work_model=2,
            task_status=1,
            pose_age_s=0.0,
            scan_age_s=0.0,
            device_age_s=0.0,
        )
        core = FixedMapControllerCore(
            context,
            policy,
        )
        decision = core.step(hypothetical)
        state = sample.vessel_state
        goal_enu = context.compiled_map.manifest.route_points_enu[
            decision.mission_index
        ]
        origin_enu = context.compiled_map.manifest.origin_enu
        goal_xy = (
            goal_enu[0] - origin_enu[0],
            goal_enu[1] - origin_enu[1],
        )
        goal_bearing = math.atan2(
            goal_xy[1] - state.y,
            goal_xy[0] - state.x,
        )
        yaw_error = (
            goal_bearing - state.yaw + math.pi
        ) % (2.0 * math.pi) - math.pi
        global_data.updateThrottleRudderOutput(
            0,
            0,
            decision.advised_heading_deg,
            decision.mission_index,
            decision.distance_to_goal_m,
        )
        print(
            json.dumps(
                {
                    "mode": "shadow-zero-output",
                    "route_points": len(route.points),
                    "route_fit_residual_m": context.fit_residual_m,
                    "grid": [
                        context.compiled_map.manifest.grid_width,
                        context.compiled_map.manifest.grid_height,
                    ],
                    "laser_valid_beams": sum(
                        sample.laser_valid_mask
                    ),
                    "actual_work_model": sample.work_model,
                    "actual_task_status": sample.task_status,
                    "state_valid": (
                        context.compiled_map.snapshot.is_state_valid(
                            state
                        )
                    ),
                    "state": {
                        "x_m": state.x,
                        "y_m": state.y,
                        "yaw_math_deg": math.degrees(state.yaw),
                        "speed_mps": state.speed,
                        "yaw_rate_math_deg_s": math.degrees(
                            state.yaw_rate
                        ),
                    },
                    "goal": {
                        "x_m": goal_xy[0],
                        "y_m": goal_xy[1],
                        "bearing_math_deg": math.degrees(goal_bearing),
                        "yaw_error_math_deg": math.degrees(yaw_error),
                    },
                    "trajectory_head": (
                        []
                        if core.trajectory is None
                        else [
                            {
                                "control": asdict(control),
                                "state": {
                                    "x_m": planned.x,
                                    "y_m": planned.y,
                                    "yaw_math_deg": math.degrees(
                                        planned.yaw
                                    ),
                                },
                            }
                            for control, planned in zip(
                                core.trajectory.controls[:5],
                                core.trajectory.states[1:6],
                            )
                        ]
                    ),
                    "hypothetical_decision": {
                        **asdict(decision),
                        "control": (
                            None
                            if decision.control is None
                            else asdict(decision.control)
                        ),
                    },
                    "published_advise": [0, 0],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        global_data.updateThrottleRudderOutput(0, 0, 0.0, 0, 0.0)
        ros = USVRosbridgeClient.ros
        if ros is not None:
            ros.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
