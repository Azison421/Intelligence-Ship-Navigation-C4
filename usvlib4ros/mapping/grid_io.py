"""Serialization for the compiled Beihu planning grid.

The JSON payload mirrors the fields needed to construct a ROS
``nav_msgs/OccupancyGrid`` without requiring ROS at build time.  PGM/YAML are
provided as a convenience for map tooling; the JSON payload remains the
canonical row-major representation because it preserves the explicit unknown
mask and build/hash metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

from .sidecar_compiler import CompiledSidecarMap
from usvlib4ros.storage import resolve_project_storage


OCCUPANCY_GRID_SCHEMA_VERSION = "navalg-occupancy-grid-v1"


def occupancy_grid_payload(compiled: CompiledSidecarMap) -> dict[str, object]:
    """Return a deterministic, ROS-compatible occupancy-grid payload."""

    snapshot = compiled.snapshot
    manifest = compiled.manifest
    values = {
        ".": 0,
        "#": 100,
        "?": -1,
    }
    data = [values[cell] for row in snapshot.rows for cell in row]
    return {
        "schema_version": OCCUPANCY_GRID_SCHEMA_VERSION,
        "frame_id": snapshot.map_frame,
        "resolution": snapshot.resolution,
        "width": snapshot.width,
        "height": snapshot.height,
        "origin": [manifest.origin_enu[0], manifest.origin_enu[1], 0.0],
        "row_order": "y_index_ascending_from_origin",
        "data_encoding": "int8_occupancy_(-1_unknown_0_free_100_occupied)",
        "data": data,
        "coverage_status": snapshot.coverage_status,
        "snapshot_id": snapshot.snapshot_id,
        "session_id": snapshot.session_id,
        "source_version": snapshot.source_version,
        "stamp_sim": snapshot.stamp_sim,
        "source_artifact_hash": snapshot.source_artifact_hash,
        "payload_content_hash": snapshot.payload_content_hash,
        "compiler_config_hash": snapshot.compiler_config_hash,
        "route_scene_id": manifest.route_scene_id,
        "route_name": manifest.route_name,
        "route_id": manifest.route_id,
        "route_version": manifest.route_version,
        "transform_model": manifest.transform_model,
        "compiler_version": manifest.compiler_version,
    }


def _pgm_byte(marker: str) -> int:
    # map_server's trinary convention: black occupied, white free, grey
    # unknown.  PGM is written top-to-bottom, so rows are reversed below while
    # the JSON payload remains y-index ascending from the map origin.
    return {"#": 0, ".": 254, "?": 205}[marker]


def _write_pgm(path: Path, rows: tuple[str, ...]) -> None:
    width = len(rows[0])
    height = len(rows)
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        for row in reversed(rows):
            handle.write(bytes(_pgm_byte(marker) for marker in row))


def write_occupancy_grid(compiled: CompiledSidecarMap, output_dir: str | Path) -> dict[str, Path]:
    """Write canonical JSON plus standard PGM/YAML views to ``output_dir``."""

    directory = resolve_project_storage(output_dir, category="maps")
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = compiled.snapshot
    payload = occupancy_grid_payload(compiled)
    json_path = directory / "beihu_planning_grid.json"
    pgm_path = directory / "beihu_planning_grid.pgm"
    yaml_path = directory / "beihu_planning_grid.yaml"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_pgm(pgm_path, snapshot.rows)
    yaml_path.write_text(
        "\n".join(
            (
                "image: beihu_planning_grid.pgm",
                f"resolution: {snapshot.resolution:.17g}",
                "origin: [%.17g, %.17g, 0.0]" % compiled.manifest.origin_enu,
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "mode: trinary",
                "# JSON sidecar is canonical; PGM rows are vertically flipped for map-server visuals.",
                f"# source_artifact_hash: {snapshot.source_artifact_hash}",
                f"# compiler_config_hash: {snapshot.compiler_config_hash}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return {"json": json_path, "pgm": pgm_path, "yaml": yaml_path}


__all__ = [
    "OCCUPANCY_GRID_SCHEMA_VERSION",
    "occupancy_grid_payload",
    "write_occupancy_grid",
]
