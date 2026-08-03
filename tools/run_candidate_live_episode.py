"""Run one offline-ready v10 candidate and write promotable Unity evidence."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from usvlib4ros.msg.global_data import GlobalData
from usvlib4ros.navigation.device_action_bridge import (
    create_ros_device_action_bridge,
)
from usvlib4ros.navigation.fixed_map_service import FixedMapNavigationService
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.policy.checkpoint_promotion import UNITY_LOG_SCHEMA
from usvlib4ros.usvRosUtil import USVRosbridgeClient


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "logs"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    return parser.parse_args()


def _manifest_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".json")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _candidate_manifest(checkpoint: Path) -> dict:
    manifest = json.loads(
        _manifest_path(checkpoint).read_text(encoding="utf-8")
    )
    if manifest.get("offline_ready") is not True:
        raise ValueError("candidate is not offline_ready")
    if manifest.get("live_ready") is not False:
        raise ValueError("candidate is not a restricted pre-promotion model")
    if _digest(checkpoint) != manifest.get("checkpoint_sha256"):
        raise ValueError("candidate checkpoint hash does not match manifest")
    return manifest


def _passed(metrics: dict, completed: bool, max_seconds: float) -> bool:
    distances = metrics.get("waypoint_min_distances_m")
    return (
        completed
        and float(metrics.get("duration_s", float("inf"))) <= max_seconds
        and metrics.get("completed_waypoints") == 13
        and isinstance(distances, list)
        and len(distances) == 13
        and all(
            isinstance(value, (int, float))
            and isfinite(float(value))
            and float(value) <= 0.5
            for value in distances
        )
        and metrics.get("collisions") == 0
        and metrics.get("laser_emergency_stops") == 0
        and metrics.get("unrecovered_unsafe_events") == 0
    )


def main() -> int:
    args = _arguments()
    checkpoint = args.checkpoint.resolve()
    if not 0.0 < args.max_seconds <= 300.0:
        raise ValueError("max-seconds must be in (0, 300]")
    manifest = _candidate_manifest(checkpoint)

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
    bridge = create_ros_device_action_bridge(config["deviceId"])
    service = FixedMapNavigationService(
        controller,
        data,
        checkpoint_path=checkpoint,
        action_bridge=bridge,
        allow_offline_candidate=True,
    )
    completed = False
    final_zero_samples = 0
    run_error = None
    bridge.start(publish_hz=30.0)
    try:
        controller.initParameterList()
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
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        for _ in range(2):
            service._publish_zero()
            final_zero_samples += 1
            time.sleep(0.05)
        bridge.close()
        ros = USVRosbridgeClient.ros
        if ros is not None:
            ros.terminate()

    metrics = dict(service.last_episode_metrics or {})
    evidence = {
        "schema_version": UNITY_LOG_SCHEMA,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "map_payload_hash": manifest["map_payload_hash"],
        "calibration_hash": manifest["calibration_hash"],
        "passed": _passed(metrics, completed, args.max_seconds),
        **metrics,
        "run_error": run_error,
        "final_zero_control_samples": final_zero_samples,
        "contains_sensitive_connection_data": False,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / (
        "unity-candidate-" + time.strftime("%Y%m%d-%H%M%S") + ".json"
    )
    target.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(str(target))
    if run_error is not None:
        print(run_error)
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
