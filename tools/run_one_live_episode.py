"""Run one bounded live National_Test episode with fail-closed shutdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from usvlib4ros.msg.global_data import GlobalData
from usvlib4ros.navigation.device_action_bridge import (
    create_ros_device_action_bridge,
)
from usvlib4ros.navigation.fixed_map_service import (
    FixedMapNavigationService,
)
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.usvRosUtil import USVRosbridgeClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Use the current active Unity pose for a bounded smoke run.",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not 0.0 < args.max_seconds <= 300.0:
        raise ValueError("max-seconds must be in (0, 300]")
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
    bridge = create_ros_device_action_bridge(config["deviceId"])
    service = FixedMapNavigationService(
        controller,
        global_data,
        action_bridge=bridge,
    )
    bridge.start(publish_hz=30.0)
    try:
        controller.initParameterList()
        if not args.no_reset:
            reset_request_time = float(
                getattr(
                    global_data.device_data,
                    "reset_request_time",
                    0.0,
                )
                or 0.0
            )
            if not controller.reset_unity():
                raise RuntimeError("Unity reset request failed")
            if not service._wait_for_reset(
                initial_request_time=reset_request_time,
            ):
                raise TimeoutError("Unity reset did not complete")
            if not controller.set_auto_work():
                raise RuntimeError("automatic work-mode request failed")
            if not service._wait_for_auto():
                raise TimeoutError("automatic work mode did not activate")
        route = controller.getRoute()
        if not getattr(route, "points", None):
            raise RuntimeError("National_Test route is unavailable")
        completed = service._run_episode(
            route,
            0,
            max_seconds=args.max_seconds,
        )
        print(
            json.dumps(
                {
                    "completed": completed,
                    "max_seconds": args.max_seconds,
                    "reset": not args.no_reset,
                },
                ensure_ascii=False,
            )
        )
        return 0 if completed else 2
    finally:
        service._publish_zero()
        bridge.close()
        ros = USVRosbridgeClient.ros
        if ros is not None:
            ros.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
