"""Fixed National_Test route compilation and kinodynamic planning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from usvlib4ros.mapping import (
    CompiledSidecarMap,
    SidecarCompilerConfig,
    compile_beihu_sidecar,
    load_sidecar_artifact,
)

from .kinodynamic_informed_rrtstar import (
    CostConfig,
    GoalRegion,
    KinodynamicInformedRRTStarPlanner,
    PlannerConfig,
    PlanningRequest,
    PrototypeReducedDynamics,
    Trajectory,
    VesselState,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "mapping" / "data"
SIDECAR_PATH = DATA_DIR / "beihu_static_world_sidecar.json"
LIVE_PROFILE_PATH = DATA_DIR / "national_test_live_profile.json"
FIXED_ROUTE_TOLERANCE_M = 0.5
MAX_FIXED_ROUTE_TOLERANCE_M = FIXED_ROUTE_TOLERANCE_M
DISPLACED_GATE_TOLERANCE_M = 0.3
SAFE_GATE_CLEARANCE_M = 0.3
ROUTE_GUIDANCE_VERSION = "national-test-exact-buoys-waypoints-v4"
_ROUTE_GATE_CACHE: dict[tuple[str, int], tuple[float, float]] = {}


def _validate_route_index(point_count: int, mission_index: int) -> None:
    """Validate a published fixed-route index."""

    if point_count <= 0 or not 0 <= mission_index < point_count:
        raise ValueError("fixed route tolerance index is invalid")


def fixed_route_goal_xy(manifest, mission_index: int) -> tuple[float, float]:
    """Return one unchanged published National_Test waypoint."""

    point_count = len(manifest.route_points_enu)
    _validate_route_index(point_count, mission_index)
    point = manifest.route_points_enu[mission_index]
    x = point[0] - manifest.origin_enu[0]
    y = point[1] - manifest.origin_enu[1]
    return x, y


def fixed_route_tolerance(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
) -> float:
    """Required ship-centre radius around the unchanged published target."""

    manifest = compiled_map.manifest
    point_count = len(manifest.route_points_enu)
    _validate_route_index(point_count, mission_index)
    return FIXED_ROUTE_TOLERANCE_M


def fixed_route_waypoint_reached(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
    state: VesselState,
) -> bool:
    """Whether the ship centre entered the unchanged waypoint region."""

    if not state.is_finite():
        return False
    goal_x, goal_y = fixed_route_goal_xy(
        compiled_map.manifest,
        mission_index,
    )
    return (
        math.hypot(state.x - goal_x, state.y - goal_y)
        <= FIXED_ROUTE_TOLERANCE_M + 1e-9
    )


def fixed_route_planning_gate(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
) -> tuple[float, float]:
    """Map-derived safe pass gate; published waypoint coordinates stay fixed."""

    manifest = compiled_map.manifest
    snapshot = compiled_map.snapshot
    point_count = len(manifest.route_points_enu)
    _validate_route_index(point_count, mission_index)
    cache_key = (snapshot.payload_content_hash, mission_index)
    cached = _ROUTE_GATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    goal = fixed_route_goal_xy(manifest, mission_index)
    goal_state = VesselState(
        x=goal[0],
        y=goal[1],
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=snapshot.stamp_sim,
    )
    if (
        snapshot.is_state_valid(goal_state)
        and snapshot.clearance_at(goal_state)
        >= SAFE_GATE_CLEARANCE_M
    ):
        _ROUTE_GATE_CACHE[cache_key] = goal
        return goal

    resolution = snapshot.resolution
    safe_cells: list[tuple[float, float, tuple[int, int]]] = []
    min_cell_x = max(
        0,
        int((goal[0] - FIXED_ROUTE_TOLERANCE_M) // resolution),
    )
    max_cell_x = min(
        snapshot.width - 1,
        int((goal[0] + FIXED_ROUTE_TOLERANCE_M) // resolution),
    )
    min_cell_y = max(
        0,
        int((goal[1] - FIXED_ROUTE_TOLERANCE_M) // resolution),
    )
    max_cell_y = min(
        snapshot.height - 1,
        int((goal[1] + FIXED_ROUTE_TOLERANCE_M) // resolution),
    )
    for cell_y in range(min_cell_y, max_cell_y + 1):
        for cell_x in range(min_cell_x, max_cell_x + 1):
            x = (cell_x + 0.5) * resolution
            y = (cell_y + 0.5) * resolution
            distance = math.hypot(x - goal[0], y - goal[1])
            if distance > MAX_FIXED_ROUTE_TOLERANCE_M:
                continue
            state = VesselState(
                x=x,
                y=y,
                yaw=0.0,
                speed=0.0,
                yaw_rate=0.0,
                stamp_sim=snapshot.stamp_sim,
            )
            clearance = snapshot.clearance_at(state)
            if (
                snapshot.is_state_valid(state)
                and clearance >= SAFE_GATE_CLEARANCE_M
            ):
                safe_cells.append(
                    (distance, -clearance, (cell_x, cell_y))
                )
    if not safe_cells:
        raise ValueError(
            "fixed route point has no safe pass gate within 0.5 m"
        )
    gate_cell = min(safe_cells)[2]
    gate = (
        (gate_cell[0] + 0.5) * resolution,
        (gate_cell[1] + 0.5) * resolution,
    )
    _ROUTE_GATE_CACHE[cache_key] = gate
    return gate


def fixed_route_gate_region(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
) -> tuple[float, float, float]:
    manifest = compiled_map.manifest
    published = fixed_route_goal_xy(manifest, mission_index)
    gate = fixed_route_planning_gate(compiled_map, mission_index)
    displaced = math.hypot(
        gate[0] - published[0],
        gate[1] - published[1],
    ) > 1e-9
    tolerance = (
        DISPLACED_GATE_TOLERANCE_M
        if displaced
        else FIXED_ROUTE_TOLERANCE_M
    )
    return gate[0], gate[1], tolerance


def fixed_route_continuations(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
) -> tuple[tuple[float, float, float], ...]:
    """Return every remaining fixed waypoint region for global lookahead."""

    manifest = compiled_map.manifest
    point_count = len(manifest.route_points_enu)
    _validate_route_index(point_count, mission_index)
    return tuple(
        (
            *fixed_route_gate_region(compiled_map, next_index),
        )
        for next_index in range(
            mission_index + 1,
            min(point_count, mission_index + 3),
        )
    )


@dataclass(frozen=True)
class FixedRoutePlan:
    """A sequentially certified plan over every fixed task waypoint."""

    compiled_map: CompiledSidecarMap
    trajectories: tuple[Trajectory, ...]
    start_mission_index: int
    final_state: VesselState


def _route_planner(
    *,
    optimize_with_rrtstar: bool,
) -> KinodynamicInformedRRTStarPlanner:
    return KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=1_200,
            edge_durations=(0.2, 0.5, 1.0, 2.0),
            goal_bias=0.25,
            global_sample_ratio=0.3,
            rewire_radius=2.5,
            connect_tolerance=1.2,
            stop_on_first_solution=not optimize_with_rrtstar,
            grid_seed_enabled=True,
            max_request_age_s=60.0,
            max_map_age_s=1.0e9,
            max_throttle=0.1,
            max_abs_rudder=0.1,
        )
    )


def compile_offline_national_map(
    *,
    session_id: str,
    stamp_sim: float = 0.0,
) -> CompiledSidecarMap:
    """Compile the current verified live affine profile without ROS access."""

    artifact, artifact_hash = load_sidecar_artifact(SIDECAR_PATH)
    profile = json.loads(LIVE_PROFILE_PATH.read_text(encoding="utf-8"))
    if profile.get("schema_version") != "national-test-live-affine-v1":
        raise ValueError("National_Test live profile schema is incompatible")
    if profile.get("source_artifact_sha256") != artifact_hash:
        raise ValueError("National_Test profile and sidecar hash do not match")
    if profile.get("route_id") != artifact["route"]["route_id"]:
        raise ValueError("National_Test profile route id does not match")
    coefficients = tuple(float(value) for value in profile["fitted_affine"])
    if len(coefficients) != 6 or not all(
        math.isfinite(value) for value in coefficients
    ):
        raise ValueError("National_Test affine coefficients are invalid")
    return compile_beihu_sidecar(
        artifact,
        source_artifact_hash=artifact_hash,
        session_id=session_id,
        stamp_sim=stamp_sim,
        config=SidecarCompilerConfig(
            transform_model="route_fitted_affine",
            coverage_status="complete_prior",
            promotion_note=(
                "operator-authorization:verified-live-route-offline-profile"
            ),
            fitted_affine=coefficients,
        ),
    )


def plan_fixed_leg(
    compiled_map: CompiledSidecarMap,
    *,
    start_state: VesselState,
    mission_index: int,
    dynamics: Optional[PrototypeReducedDynamics] = None,
    cost_config: Optional[CostConfig] = None,
    time_budget_ms: float = 5_000.0,
    optimize_with_rrtstar: bool = False,
    seed: int = 31,
    _allow_retry: bool = True,
) -> Trajectory:
    """Plan one task leg from the latest measured or simulated state."""

    manifest = compiled_map.manifest
    snapshot = compiled_map.snapshot
    dynamics = dynamics or PrototypeReducedDynamics()
    cost_config = cost_config or CostConfig()
    if not 0 <= mission_index < len(manifest.route_points_enu):
        raise ValueError("mission_index is outside the fixed route")
    if not snapshot.is_state_valid(start_state):
        raise ValueError("fixed leg start state is not valid")
    if not math.isfinite(time_budget_ms) or time_budget_ms <= 0.0:
        raise ValueError("time_budget_ms must be positive and finite")
    goal_x, goal_y = fixed_route_goal_xy(manifest, mission_index)
    goal = GoalRegion(
        x=goal_x,
        y=goal_y,
        position_tolerance=fixed_route_tolerance(
            compiled_map,
            mission_index,
        ),
        heading_tolerance=math.pi,
        speed_limit=1.2,
        yaw_rate_limit=1.2,
    )
    continuations = fixed_route_continuations(
        compiled_map,
        mission_index,
    )
    result = None
    for lookahead_count in range(len(continuations), -1, -1):
        request = PlanningRequest(
            request_id=(
                f"fixed-route-live-leg-{mission_index}"
                f"-lookahead-{lookahead_count}"
            ),
            session_id=snapshot.session_id,
            start_state=start_state,
            goal_region=goal,
            map_snapshot_id=snapshot.snapshot_id,
            dynamics_version=dynamics.version,
            cost_config_version=cost_config.version,
            time_budget_ms=time_budget_ms,
            seed=seed + mission_index,
            mission_index=mission_index,
            stamp_sim=start_state.stamp_sim,
            mission_version=f"route-v{manifest.route_version}",
            route_gate=fixed_route_gate_region(
                compiled_map,
                mission_index,
            ),
            continuation_targets=continuations[:lookahead_count],
        )
        result = _route_planner(
            optimize_with_rrtstar=optimize_with_rrtstar,
        ).plan(
            request,
            snapshot,
            dynamics,
            cost_config,
            now_sim=start_state.stamp_sim,
        )
        if (
            result.trajectory is not None
            and result.trajectory.controls
        ):
            return result.trajectory
    for recovery_attempt in range(2):
        fallback_request = PlanningRequest(
            request_id=(
                f"fixed-route-live-leg-{mission_index}"
                f"-recovery-{recovery_attempt}"
            ),
            session_id=snapshot.session_id,
            start_state=start_state,
            goal_region=goal,
            map_snapshot_id=snapshot.snapshot_id,
            dynamics_version=dynamics.version,
            cost_config_version=cost_config.version,
            time_budget_ms=time_budget_ms,
            seed=seed + mission_index + 1_009 * recovery_attempt,
            mission_index=mission_index,
            stamp_sim=start_state.stamp_sim,
            mission_version=f"route-v{manifest.route_version}",
        )
        result = _route_planner(
            optimize_with_rrtstar=optimize_with_rrtstar,
        ).plan(
            fallback_request,
            snapshot,
            dynamics,
            cost_config,
            now_sim=start_state.stamp_sim,
        )
        if (
            result.trajectory is not None
            and result.trajectory.controls
        ):
            return result.trajectory
    if _allow_retry:
        return plan_fixed_leg(
            compiled_map,
            start_state=start_state,
            mission_index=mission_index,
            dynamics=dynamics,
            cost_config=cost_config,
            time_budget_ms=time_budget_ms,
            optimize_with_rrtstar=optimize_with_rrtstar,
            seed=seed + 10_009,
            _allow_retry=False,
        )
    assert result is not None
    raise RuntimeError(
        f"fixed route leg {mission_index} failed: "
        f"{result.status.value} {result.reason}"
    )


def plan_fixed_route(
    compiled_map: CompiledSidecarMap,
    *,
    dynamics: Optional[PrototypeReducedDynamics] = None,
    cost_config: Optional[CostConfig] = None,
    start_state: Optional[VesselState] = None,
    start_mission_index: int = 1,
    time_budget_ms: float = 5_000.0,
    optimize_with_rrtstar: bool = False,
    seed: int = 31,
) -> FixedRoutePlan:
    """Plan and independently validate every remaining fixed route leg.

    The deterministic grid/lattice rollout supplies a feasible kinodynamic
    warm start.  With ``optimize_with_rrtstar`` enabled, the planner spends
    the remaining per-leg budget on informed sampling and rewiring.
    """

    manifest = compiled_map.manifest
    snapshot = compiled_map.snapshot
    dynamics = dynamics or PrototypeReducedDynamics()
    cost_config = cost_config or CostConfig()
    if not 0 <= start_mission_index < len(manifest.route_points_enu):
        raise ValueError("start_mission_index is outside the fixed route")
    if not math.isfinite(time_budget_ms) or time_budget_ms <= 0.0:
        raise ValueError("time_budget_ms must be positive and finite")

    if start_state is None:
        current_enu = manifest.route_points_enu[start_mission_index - 1]
        goal_enu = manifest.route_points_enu[start_mission_index]
        x = current_enu[0] - manifest.origin_enu[0]
        y = current_enu[1] - manifest.origin_enu[1]
        goal_x = goal_enu[0] - manifest.origin_enu[0]
        goal_y = goal_enu[1] - manifest.origin_enu[1]
        start_state = VesselState(
            x=x,
            y=y,
            yaw=math.atan2(goal_y - y, goal_x - x),
            speed=0.0,
            yaw_rate=0.0,
            stamp_sim=snapshot.stamp_sim,
        )
    if not snapshot.is_state_valid(start_state):
        raise ValueError("fixed route start state is not valid")

    trajectories = []
    state = start_state
    for mission_index in range(
        start_mission_index,
        len(manifest.route_points_enu),
    ):
        trajectory = plan_fixed_leg(
            compiled_map,
            start_state=state,
            mission_index=mission_index,
            dynamics=dynamics,
            cost_config=cost_config,
            time_budget_ms=time_budget_ms,
            optimize_with_rrtstar=optimize_with_rrtstar,
            seed=seed,
        )
        trajectories.append(trajectory)
        state = trajectory.states[-1]

    return FixedRoutePlan(
        compiled_map=compiled_map,
        trajectories=tuple(trajectories),
        start_mission_index=start_mission_index,
        final_state=state,
    )


__all__ = [
    "FIXED_ROUTE_TOLERANCE_M",
    "MAX_FIXED_ROUTE_TOLERANCE_M",
    "ROUTE_GUIDANCE_VERSION",
    "SAFE_GATE_CLEARANCE_M",
    "FixedRoutePlan",
    "compile_offline_national_map",
    "fixed_route_continuations",
    "fixed_route_gate_region",
    "fixed_route_goal_xy",
    "fixed_route_planning_gate",
    "fixed_route_tolerance",
    "fixed_route_waypoint_reached",
    "plan_fixed_leg",
    "plan_fixed_route",
]
