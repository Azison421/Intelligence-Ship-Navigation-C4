"""Pure-Python compiler for the fixed Beihu build-bound StaticWorld sidecar.

Consumes the extracted sidecar artifact (water mesh, fixed buoys, Buoy
collider envelope, GPS anchors, build hashes) and produces a versioned
``PlanningMapSnapshot`` plus an audit manifest.

Boundaries (ADR-001, P0-17/P0-18):

- the water mesh only defines the candidate navigable boundary; fixed buoys
  come from route.txt instances with the Buoy collider envelope;
- random training instances are never synthesized here;
- coverage is ``candidate_complete_prior`` by default.  Promoting a compiled
  snapshot to ``complete_prior`` requires an explicit, versioned operator
  authorization recorded in the compile report (see ADR-004); without it the
  snapshot stays a candidate and the planning gates reject it;
- no ROS, Unity or MATLAB dependency: this module is pure Python + numpy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from usvlib4ros.planning import CircularObstacle, PlanningMapSnapshot

from .coordinates import (
    AffineTransform2D,
    GpsProjector,
    SimilarityTransform2D,
    fit_affine_unity_to_enu,
    fit_unity_to_enu,
)

SIDECAR_SCHEMA_VERSION = "build-bound-static-world-sidecar-v1"
COMPILER_VERSION = "beihu-sidecar-compiler-v3"
DEFAULT_RESOLUTION_M = 0.2
DEFAULT_MARGIN_M = 5.0
DEFAULT_FOOTPRINT_RADIUS_M = 0.4
DEFAULT_REQUIRED_CLEARANCE_M = 0.2
COVERAGE_CANDIDATE = "candidate_complete_prior"
COVERAGE_COMPLETE = "complete_prior"
PROMOTION_NOTE_PREFIX = "operator-authorization:"
TRANSFORM_MODELS = ("axis_affine", "similarity", "similarity_reflected", "route_fitted_affine")


@dataclass(frozen=True)
class SidecarCompilerConfig:
    """Versioned knobs that shape the compiled snapshot; all enter its hash."""

    resolution_m: float = DEFAULT_RESOLUTION_M
    margin_m: float = DEFAULT_MARGIN_M
    footprint_radius_m: float = DEFAULT_FOOTPRINT_RADIUS_M
    required_clearance_m: float = DEFAULT_REQUIRED_CLEARANCE_M
    geometry_version: str = "circle-v1"
    transform_model: str = "axis_affine"
    coverage_status: str = COVERAGE_CANDIDATE
    promotion_note: str = ""
    fitted_affine: tuple[float, ...] = ()

    def is_valid(self) -> bool:
        fitted_ok = self.transform_model != "route_fitted_affine" or (
            len(self.fitted_affine) == 6 and all(math.isfinite(v) for v in self.fitted_affine)
        )
        return (
            min(self.resolution_m, self.margin_m) > 0.0
            and self.footprint_radius_m >= 0.0
            and self.required_clearance_m >= 0.0
            and isinstance(self.geometry_version, str)
            and bool(self.geometry_version)
            and self.transform_model in TRANSFORM_MODELS
            and fitted_ok
            and self.coverage_status in (COVERAGE_CANDIDATE, COVERAGE_COMPLETE)
            and (self.coverage_status != COVERAGE_COMPLETE or self.promotion_note.startswith(PROMOTION_NOTE_PREFIX))
        )

    def config_hash(self) -> str:
        payload = "|".join(
            (
                COMPILER_VERSION,
                f"{self.resolution_m:.17g}",
                f"{self.margin_m:.17g}",
                f"{self.footprint_radius_m:.17g}",
                f"{self.required_clearance_m:.17g}",
                f"geometry={self.geometry_version}",
                f"transform={self.transform_model}",
                f"coverage={self.coverage_status}",
                "fitted=" + ",".join(f"{v:.17g}" for v in self.fitted_affine),
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FixedBuoy:
    name: str
    x: float
    y: float
    radius_m: float


@dataclass(frozen=True)
class SidecarMapManifest:
    """Audit metadata the runtime adapter needs alongside the snapshot."""

    snapshot_id: str
    origin_enu: tuple[float, float]
    resolution_m: float
    grid_width: int
    grid_height: int
    water_cells: int
    buoy_cells: int
    buoys: tuple[FixedBuoy, ...]
    route_points_enu: tuple[tuple[float, float], ...]
    cruise_speeds: tuple[float, ...]
    route_scene_id: str
    route_name: str
    route_id: str
    route_version: int
    transform_model: str
    coverage_status: str
    promotion_note: str
    source_artifact_hash: str
    compiler_config_hash: str
    compiler_version: str
    gps_origin: tuple[float, float]


@dataclass(frozen=True)
class CompiledSidecarMap:
    snapshot: PlanningMapSnapshot
    manifest: SidecarMapManifest


def load_sidecar_artifact(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    artifact = json.loads(raw.decode("utf-8"))
    if artifact.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ValueError("sidecar artifact schema mismatch")
    return artifact, sha256(raw).hexdigest()


def _unity_world_vertices(artifact: dict) -> np.ndarray:
    mesh = artifact["water_mesh"]
    world = np.array(mesh["world_matrix"], dtype=np.float64)
    local = np.array(mesh["vertices_local"], dtype=np.float64)
    homogeneous = np.hstack([local, np.ones((local.shape[0], 1))])
    return (world @ homogeneous.T).T[:, :3]


def unity_point_in_water(artifact: dict, ux: float, uz: float) -> bool:
    """True if the Unity-world point lies inside any water triangle.

    Independent geometric gate for the live converter: the ship's GPS,
    mapped back to Unity coordinates, must land in the extracted water mesh
    (see ADR-004 amendment, 2026-07-30 evidence).
    """

    world = _unity_world_vertices(artifact)
    triangles = np.array(artifact["water_mesh"]["triangles"], dtype=np.int64)
    for tri in triangles:
        a, b, c = world[tri[0]], world[tri[1]], world[tri[2]]
        v0x, v0z = c[0] - a[0], c[2] - a[2]
        v1x, v1z = b[0] - a[0], b[2] - a[2]
        v2x, v2z = ux - a[0], uz - a[2]
        den = v0x * v1z - v1x * v0z
        if abs(den) < 1e-12:
            continue
        u = (v2x * v1z - v1x * v2z) / den
        v = (v0x * v2z - v2x * v0z) / den
        if u >= 0.0 and v >= 0.0 and (u + v) <= 1.0:
            return True
    return False


def _rasterize_water(triangles_enu: np.ndarray, origin: tuple[float, float], width: int, height: int, resolution: float) -> np.ndarray:
    """Mark cells whose centre lies inside any water triangle (conservative)."""

    water = np.zeros((height, width), dtype=bool)
    ox, oy = origin
    for triangle in triangles_enu:
        min_x = max(0, int((triangle[:, 0].min() - ox) // resolution))
        max_x = min(width - 1, int((triangle[:, 0].max() - ox) // resolution))
        min_y = max(0, int((triangle[:, 1].min() - oy) // resolution))
        max_y = min(height - 1, int((triangle[:, 1].max() - oy) // resolution))
        if min_x > max_x or min_y > max_y:
            continue
        xs = ox + (np.arange(min_x, max_x + 1) + 0.5) * resolution
        ys = oy + (np.arange(min_y, max_y + 1) + 0.5) * resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        a, b, c = triangle[0], triangle[1], triangle[2]
        v0x, v0y = c[0] - a[0], c[1] - a[1]
        v1x, v1y = b[0] - a[0], b[1] - a[1]
        v2x, v2y = grid_x - a[0], grid_y - a[1]
        denominator = v0x * v1y - v1x * v0y
        if abs(denominator) < 1e-12:
            continue
        u = (v2x * v1y - v1x * v2y) / denominator
        v = (v0x * v2y - v2x * v0y) / denominator
        inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        water[min_y : max_y + 1, min_x : max_x + 1] |= inside
    return water


def compile_beihu_sidecar(
    artifact: dict,
    *,
    source_artifact_hash: str,
    session_id: str,
    source_version: int = 1,
    stamp_sim: float = 0.0,
    config: Optional[SidecarCompilerConfig] = None,
    snapshot_id: Optional[str] = None,
) -> CompiledSidecarMap:
    """Compile the fixed sidecar into a PlanningMapSnapshot + audit manifest."""

    config = config or SidecarCompilerConfig()
    if not config.is_valid():
        raise ValueError("sidecar compiler config is invalid or promotion note missing")
    for key in ("water_mesh", "buoy_collider", "route", "gps_anchors", "source_hashes"):
        if key not in artifact:
            raise ValueError(f"sidecar artifact is missing {key!r}")

    anchors = artifact["gps_anchors"]
    projector = GpsProjector(float(anchors["latitude1"]), float(anchors["longitude1"]))
    if config.transform_model == "axis_affine":
        unity_to_enu = fit_affine_unity_to_enu(anchors, projector)
        unity_scale = max(abs(unity_to_enu.scale_x), abs(unity_to_enu.scale_z))
    elif config.transform_model == "similarity":
        unity_to_enu = fit_unity_to_enu(anchors, projector, reflected=False)
        unity_scale = unity_to_enu.scale
    elif config.transform_model == "route_fitted_affine":
        unity_to_enu = AffineTransform2D(*config.fitted_affine)
        unity_scale = unity_to_enu.max_scale()
    else:
        unity_to_enu = fit_unity_to_enu(anchors, projector, reflected=True)
        unity_scale = unity_to_enu.scale

    world_vertices = _unity_world_vertices(artifact)
    enu_xy = np.array(
        [unity_to_enu.unity_to_enu(float(x), float(z)) for x, _, z in world_vertices],
        dtype=np.float64,
    )
    triangles = np.array(artifact["water_mesh"]["triangles"], dtype=np.int64)
    triangles_enu = enu_xy[triangles]

    route = artifact["route"]
    buoy_collider = artifact["buoy_collider"]
    buoys = tuple(
        FixedBuoy(
            name=str(obstacle.get("name", "")),
            x=unity_to_enu.unity_to_enu(float(obstacle["unity_position"][0]), float(obstacle["unity_position"][2]))[0],
            y=unity_to_enu.unity_to_enu(float(obstacle["unity_position"][0]), float(obstacle["unity_position"][2]))[1],
            radius_m=float(buoy_collider["radius"])
            * max(1.0, float(obstacle.get("scale", [1.0])[0]))
            * unity_scale,
        )
        for obstacle in route["obstacles"]
    )
    route_points_enu = tuple(
        unity_to_enu.unity_to_enu(float(point["unity_position"][0]), float(point["unity_position"][2]))
        for point in route["points"]
    )
    cruise_speeds = tuple(float(point.get("cruise_speed", 0.0)) for point in route["points"])

    margin = config.margin_m
    min_x = float(enu_xy[:, 0].min()) - margin
    min_y = float(enu_xy[:, 1].min()) - margin
    max_x = float(enu_xy[:, 0].max()) + margin
    max_y = float(enu_xy[:, 1].max()) + margin
    resolution = config.resolution_m
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))
    origin = (min_x, min_y)

    water = _rasterize_water(triangles_enu, origin, width, height, resolution)
    occupied = np.zeros((height, width), dtype=bool)
    ox, oy = origin
    half_diagonal = resolution * float(np.sqrt(2.0)) / 2.0
    for buoy in buoys:
        cell_x = (buoy.x - ox) / resolution - 0.5
        cell_y = (buoy.y - oy) / resolution - 0.5
        reach = int(np.ceil((buoy.radius_m + half_diagonal) / resolution)) + 1
        cx = int(round(cell_x))
        cy = int(round(cell_y))
        for gy in range(max(0, cy - reach), min(height, cy + reach + 1)):
            for gx in range(max(0, cx - reach), min(width, cx + reach + 1)):
                center_x = ox + (gx + 0.5) * resolution
                center_y = oy + (gy + 0.5) * resolution
                # Cell-disc intersection: a buoy disc can never fall between
                # cells unmarked; the half-diagonal makes the mark conservative.
                if (center_x - buoy.x) ** 2 + (center_y - buoy.y) ** 2 <= (buoy.radius_m + half_diagonal) ** 2:
                    occupied[gy, gx] = True

    rows = []
    for gy in range(height):
        row_chars = []
        for gx in range(width):
            if water[gy, gx]:
                row_chars.append(".")
            else:
                row_chars.append("?")
        rows.append("".join(row_chars))

    resolved_snapshot_id = snapshot_id or f"beihu-national-test-sidecar-v{source_version}"
    snapshot = PlanningMapSnapshot.from_rows(
        tuple(rows),
        snapshot_id=resolved_snapshot_id,
        session_id=session_id,
        source_version=source_version,
        map_frame="map",
        resolution=resolution,
        footprint_radius=config.footprint_radius_m,
        required_clearance=config.required_clearance_m,
        coverage_status=config.coverage_status,
        stamp_sim=stamp_sim,
        source_artifact_hash=source_artifact_hash,
        compiler_config_hash=config.config_hash(),
        circular_obstacles=tuple(
            CircularObstacle(
                x=buoy.x - origin[0],
                y=buoy.y - origin[1],
                radius=buoy.radius_m,
            )
            for buoy in buoys
        ),
        geometry_version=config.geometry_version,
    )
    snapshot.precompute_clearance()
    manifest = SidecarMapManifest(
        snapshot_id=resolved_snapshot_id,
        origin_enu=origin,
        resolution_m=resolution,
        grid_width=width,
        grid_height=height,
        water_cells=int(water.sum()),
        buoy_cells=int(occupied.sum()),
        buoys=buoys,
        route_points_enu=route_points_enu,
        cruise_speeds=cruise_speeds,
        route_scene_id=str(route.get("scene_id", "")),
        route_name=str(route.get("route_name", "")),
        route_id=str(route.get("route_id", "")),
        route_version=int(route.get("route_version", 0)),
        transform_model=config.transform_model,
        coverage_status=config.coverage_status,
        promotion_note=config.promotion_note,
        source_artifact_hash=source_artifact_hash,
        compiler_config_hash=config.config_hash(),
        compiler_version=COMPILER_VERSION,
        gps_origin=(float(anchors["latitude1"]), float(anchors["longitude1"])),
    )
    return CompiledSidecarMap(snapshot=snapshot, manifest=manifest)


def enu_to_grid(manifest: SidecarMapManifest, x_enu: float, y_enu: float) -> tuple[float, float]:
    """Map ENU metres -> planner grid frame (snapshot rows start at (0, 0))."""

    return x_enu - manifest.origin_enu[0], y_enu - manifest.origin_enu[1]


def grid_to_enu(manifest: SidecarMapManifest, x_grid: float, y_grid: float) -> tuple[float, float]:
    return x_grid + manifest.origin_enu[0], y_grid + manifest.origin_enu[1]
