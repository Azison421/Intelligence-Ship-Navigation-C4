import hashlib
import json

from usvlib4ros.policy.checkpoint_promotion import promote_checkpoint


def test_promotion_requires_three_matching_passed_unity_validation_logs(tmp_path):
    checkpoint = tmp_path / "national_test_sac_live_v10.pt"
    checkpoint.write_bytes(b"candidate")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest_path = checkpoint.with_suffix(".pt.json")
    manifest = {
        "schema_version": "national-test-sac-checkpoint-v4",
        "checkpoint_sha256": checkpoint_hash,
        "map_payload_hash": "map-hash",
        "geometry_version": "geometry-v1",
        "calibration_hash": "calibration-hash",
        "action_schema": "five-discrete-forward-bias-v2",
        "offline_ready": True,
        "live_ready": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    logs = []
    for run in range(3):
        path = tmp_path / f"unity-{run}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "national-test-unity-validation-v1",
                    "checkpoint_sha256": checkpoint_hash,
                    "map_payload_hash": "map-hash",
                    "calibration_hash": "calibration-hash",
                    "passed": True,
                    "duration_s": 299.0,
                    "completed_waypoints": 13,
                    "waypoint_min_distances_m": [0.5] * 13,
                    "collisions": 0,
                    "laser_emergency_stops": 0,
                    "unrecovered_unsafe_events": 0,
                    "final_zero_control_samples": 2,
                }
            ),
            encoding="utf-8",
        )
        logs.append(path)

    promoted = promote_checkpoint(manifest_path, logs)

    assert promoted["live_ready"] is True
    assert len(promoted["unity_validation_log_hashes"]) == 3
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == promoted

