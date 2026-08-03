"""Fail-closed promotion of an offline v10 candidate after Unity evidence."""

from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Iterable


MANIFEST_SCHEMA = "national-test-sac-checkpoint-v4"
UNITY_LOG_SCHEMA = "national-test-unity-validation-v1"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def promote_checkpoint(
    manifest_path: Path,
    unity_log_paths: Iterable[Path],
) -> dict:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("candidate manifest schema is incompatible")
    if manifest.get("offline_ready") is not True:
        raise ValueError("candidate has not passed offline evaluation")
    if manifest.get("live_ready") is not False:
        raise ValueError("candidate is not in promotable state")
    checkpoint = manifest_file.with_suffix("")
    if not checkpoint.is_file() or _digest(checkpoint) != manifest.get(
        "checkpoint_sha256"
    ):
        raise ValueError("candidate checkpoint hash is invalid")

    logs = tuple(Path(path) for path in unity_log_paths)
    if len(logs) < 3 or len(set(logs)) != len(logs):
        raise ValueError("three distinct Unity validation logs are required")
    log_hashes = []
    for path in logs:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("schema_version") != UNITY_LOG_SCHEMA:
            raise ValueError("Unity validation log schema is incompatible")
        for key in (
            "checkpoint_sha256",
            "map_payload_hash",
            "calibration_hash",
        ):
            if evidence.get(key) != manifest.get(key):
                raise ValueError(f"Unity validation {key} does not match")
        distances = evidence.get("waypoint_min_distances_m")
        if (
            evidence.get("passed") is not True
            or not isinstance(evidence.get("duration_s"), (int, float))
            or not isfinite(float(evidence["duration_s"]))
            or not 0.0 <= float(evidence["duration_s"]) <= 300.0
            or evidence.get("completed_waypoints") != 13
            or not isinstance(distances, list)
            or len(distances) != 13
            or any(
                not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) > 0.5
                for value in distances
            )
            or evidence.get("collisions") != 0
            or evidence.get("laser_emergency_stops") != 0
            or evidence.get("unrecovered_unsafe_events") != 0
            or int(evidence.get("final_zero_control_samples", 0)) < 2
        ):
            raise ValueError("Unity validation acceptance criteria were not met")
        log_hashes.append(_digest(path))

    promoted = dict(manifest)
    promoted["unity_validation_log_hashes"] = log_hashes
    promoted["live_ready"] = True
    temporary = manifest_file.with_suffix(manifest_file.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            promoted,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_file)
    return promoted


__all__ = ["promote_checkpoint"]
