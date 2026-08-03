"""Fixed National_Test route compilation and kinodynamic planning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Optional

from usvlib4ros.mapping import (
    CompiledSidecarMap,
    SidecarCompilerConfig,
    compile_beihu_sidecar,
    load_sidecar_artifact,
)

from .kinodynamic_informed_rrtstar import (
    Control,
    CostConfig,
    GoalRegion,
    KinodynamicInformedRRTStarPlanner,
    PlannerConfig,
    PlanningRequest,
    PrototypeReducedDynamics,
    Trajectory,
    VesselState,
)
from .forward_control_profile import (
    diagnostic_forward_control_profile,
    reduced_dynamics_from_profile,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "mapping" / "data"
SIDECAR_PATH = DATA_DIR / "beihu_static_world_sidecar.json"
LIVE_PROFILE_PATH = DATA_DIR / "national_test_live_profile.json"
FIXED_ROUTE_TOLERANCE_M = 0.5
MAX_FIXED_ROUTE_TOLERANCE_M = FIXED_ROUTE_TOLERANCE_M
DISPLACED_GATE_TOLERANCE_M = 0.3
SAFE_GATE_CLEARANCE_M = 0.3
ROUTE_GUIDANCE_VERSION = "national-test-reversible-composite-v16"
NARROW_ROUTE_INDEX = 10
NARROW_ESCAPE_XY = (31.6, 99.5)
NARROW_ESCAPE_TOLERANCE_M = 0.3
NARROW_ESCAPE_RELEASE_X_M = 31.3
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


def narrow_escape_released(
    compiled_map: CompiledSidecarMap,
    state: VesselState,
) -> bool:
    """Release reverse once the vessel has crossed the safe east plane."""

    snapshot = compiled_map.snapshot
    exact_escape = (
        math.hypot(
            state.x - NARROW_ESCAPE_XY[0],
            state.y - NARROW_ESCAPE_XY[1],
        )
        <= NARROW_ESCAPE_TOLERANCE_M + 1e-9
    )
    safe_plane = (
        state.x >= NARROW_ESCAPE_RELEASE_X_M
        and state.speed <= 0.05
        and snapshot.is_state_valid(state)
        and snapshot.clearance_at(state)
        >= snapshot.required_clearance
    )
    return exact_escape or safe_plane


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


def fixed_route_guidance_hash(compiled_map: CompiledSidecarMap) -> str:
    payload = {
        "version": ROUTE_GUIDANCE_VERSION,
        "published_goals": [
            fixed_route_goal_xy(compiled_map.manifest, index)
            for index in range(
                len(compiled_map.manifest.route_points_enu)
            )
        ],
        "planning_gates": [
            fixed_route_gate_region(compiled_map, index)
            for index in range(
                len(compiled_map.manifest.route_points_enu)
            )
        ],
        "narrow_route_index": NARROW_ROUTE_INDEX,
        "narrow_escape_xy": NARROW_ESCAPE_XY,
        "narrow_escape_tolerance_m": NARROW_ESCAPE_TOLERANCE_M,
        "narrow_escape_release_x_m": NARROW_ESCAPE_RELEASE_X_M,
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def fixed_route_geometry_candidates(
    compiled_map: CompiledSidecarMap,
) -> tuple[CompiledSidecarMap, ...]:
    """Ordered collision-model evidence gate; never weakens below 0.1 m."""

    snapshot = compiled_map.snapshot
    configurations = (
        {
            "footprint_radius": 0.4,
            "required_clearance": 0.2,
            "vessel_capsule_length": 0.0,
            "vessel_capsule_width": 0.0,
            "geometry_version": "circle-0.4-margin-0.2-v1",
        },
        {
            "footprint_radius": 0.0,
            "required_clearance": 0.2,
            "vessel_capsule_length": 1.3,
            "vessel_capsule_width": 0.64,
            "geometry_version": (
                "official-capsule-1.3x0.64-margin-0.2-v1"
            ),
        },
        {
            "footprint_radius": 0.0,
            "required_clearance": 0.1,
            "vessel_capsule_length": 1.3,
            "vessel_capsule_width": 0.64,
            "geometry_version": (
                "official-capsule-1.3x0.64-margin-0.1-v1"
            ),
        },
    )
    return tuple(
        replace(
            compiled_map,
            snapshot=replace(
                snapshot,
                payload_content_hash="",
                **configuration,
            ),
        )
        for configuration in configurations
    )


def build_fixed_leg_request(
    compiled_map: CompiledSidecarMap,
    *,
    start_state: VesselState,
    mission_index: int,
    dynamics: PrototypeReducedDynamics,
    cost_config: CostConfig,
    time_budget_ms: float,
    seed: int,
    lookahead_count: int,
    narrow_visit_completed: bool = False,
) -> PlanningRequest:
    """Compile one ordinary or narrow composite leg without changing task points."""

    manifest = compiled_map.manifest
    snapshot = compiled_map.snapshot
    _validate_route_index(len(manifest.route_points_enu), mission_index)
    continuations = fixed_route_continuations(compiled_map, mission_index)
    if not 0 <= lookahead_count <= len(continuations):
        raise ValueError("lookahead_count is outside the available route")
    goal_x, goal_y = fixed_route_goal_xy(manifest, mission_index)
    published_goal = GoalRegion(
        x=goal_x,
        y=goal_y,
        position_tolerance=FIXED_ROUTE_TOLERANCE_M,
        heading_tolerance=math.pi,
        speed_limit=1.8,
        yaw_rate_limit=1.2,
    )
    route_gate = fixed_route_gate_region(compiled_map, mission_index)
    required_visit_regions: tuple[GoalRegion, ...] = ()
    goal = published_goal
    continuation_targets = continuations[:lookahead_count]
    if mission_index == NARROW_ROUTE_INDEX:
        goal = GoalRegion(
            x=NARROW_ESCAPE_XY[0],
            y=NARROW_ESCAPE_XY[1],
            position_tolerance=NARROW_ESCAPE_TOLERANCE_M,
            heading_tolerance=math.pi,
            speed_limit=1.8,
            yaw_rate_limit=1.2,
        )
        if not narrow_visit_completed:
            required_visit_regions = (
                GoalRegion(
                    x=route_gate[0],
                    y=route_gate[1],
                    position_tolerance=route_gate[2],
                    heading_tolerance=math.pi,
                    speed_limit=1.8,
                    yaw_rate_limit=1.2,
                ),
                published_goal,
            )
        route_gate = None
        continuation_targets = ()
    return PlanningRequest(
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
        route_gate=route_gate,
        continuation_targets=continuation_targets,
        required_visit_regions=required_visit_regions,
    )


@dataclass(frozen=True)
class FixedRoutePlan:
    """A sequentially certified plan over every fixed task waypoint."""

    compiled_map: CompiledSidecarMap
    trajectories: tuple[Trajectory, ...]
    start_mission_index: int
    final_state: VesselState


@dataclass(frozen=True)
class GeometryGateEvidence:
    geometry_version: str
    map_payload_hash: str
    required_clearance_m: float
    feasible: bool
    reason: str


class NarrowCompositeInfeasibleError(RuntimeError):
    def __init__(self, evidence: tuple[GeometryGateEvidence, ...]) -> None:
        self.evidence = evidence
        summary = "; ".join(
            f"{item.geometry_version}:{item.reason}" for item in evidence
        )
        super().__init__(
            "narrow composite is infeasible under all approved geometry "
            f"gates: {summary}"
        )


def _route_planner(
    *,
    optimize_with_rrtstar: bool,
    forward_action_controls: tuple[Control, ...] = (),
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
            forward_action_controls=forward_action_controls,
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
            required_clearance_m=0.2,
            geometry_version=(
                "circle-0.4-margin-0.2-live-recovery-v1"
            ),
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
    forward_action_controls: tuple[Control, ...] = (),
    narrow_visit_completed: bool = False,
    _allow_retry: bool = True,
) -> Trajectory:
    """Plan one task leg from the latest measured or simulated state."""

    manifest = compiled_map.manifest
    snapshot = compiled_map.snapshot
    if dynamics is None and not forward_action_controls:
        profile = diagnostic_forward_control_profile()
        dynamics = reduced_dynamics_from_profile(profile)
        forward_action_controls = profile.action_controls
    else:
        dynamics = dynamics or PrototypeReducedDynamics()
    cost_config = cost_config or CostConfig()
    if not 0 <= mission_index < len(manifest.route_points_enu):
        raise ValueError("mission_index is outside the fixed route")
    if not snapshot.is_state_valid(start_state):
        raise ValueError("fixed leg start state is not valid")
    if not math.isfinite(time_budget_ms) or time_budget_ms <= 0.0:
        raise ValueError("time_budget_ms must be positive and finite")
    continuations = fixed_route_continuations(
        compiled_map,
        mission_index,
    )
    result = None
    lookahead_counts = (
        (0,)
        if mission_index == NARROW_ROUTE_INDEX
        else range(len(continuations), -1, -1)
    )
    for lookahead_count in lookahead_counts:
        request = build_fixed_leg_request(
            compiled_map,
            start_state=start_state,
            mission_index=mission_index,
            dynamics=dynamics,
            cost_config=cost_config,
            time_budget_ms=time_budget_ms,
            seed=seed,
            lookahead_count=lookahead_count,
            narrow_visit_completed=narrow_visit_completed,
        )
        result = _route_planner(
            optimize_with_rrtstar=optimize_with_rrtstar,
            forward_action_controls=forward_action_controls,
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
    if mission_index == NARROW_ROUTE_INDEX:
        assert result is not None
        raise RuntimeError(
            f"fixed route leg {mission_index} failed: "
            f"{result.status.value} {result.reason}"
        )
    for recovery_attempt in range(2):
        goal_x, goal_y = fixed_route_goal_xy(
            manifest,
            mission_index,
        )
        fallback_request = PlanningRequest(
            request_id=(
                f"fixed-route-live-leg-{mission_index}"
                f"-recovery-{recovery_attempt}"
            ),
            session_id=snapshot.session_id,
            start_state=start_state,
            goal_region=GoalRegion(
                x=goal_x,
                y=goal_y,
                position_tolerance=fixed_route_tolerance(
                    compiled_map,
                    mission_index,
                ),
                heading_tolerance=math.pi,
                speed_limit=1.2,
                yaw_rate_limit=1.2,
            ),
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
            forward_action_controls=forward_action_controls,
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
            forward_action_controls=forward_action_controls,
            narrow_visit_completed=narrow_visit_completed,
            _allow_retry=False,
        )
    assert result is not None
    raise RuntimeError(
        f"fixed route leg {mission_index} failed: "
        f"{result.status.value} {result.reason}"
    )


def plan_narrow_with_geometry_evidence(
    compiled_map: CompiledSidecarMap,
    *,
    start_state: VesselState,
    dynamics: Optional[PrototypeReducedDynamics] = None,
    cost_config: Optional[CostConfig] = None,
    time_budget_ms: float = 5_000.0,
    seed: int = 31,
    forward_action_controls: tuple[Control, ...] = (),
) -> tuple[
    CompiledSidecarMap,
    Trajectory,
    tuple[GeometryGateEvidence, ...],
]:
    """Try the three approved geometry gates in order and fail explicitly."""

    if dynamics is None and not forward_action_controls:
        profile = diagnostic_forward_control_profile()
        dynamics = reduced_dynamics_from_profile(profile)
        forward_action_controls = profile.action_controls
    else:
        dynamics = dynamics or PrototypeReducedDynamics()
    cost_config = cost_config or CostConfig()
    evidence = []
    for candidate in fixed_route_geometry_candidates(compiled_map):
        try:
            trajectory = plan_fixed_leg(
                candidate,
                start_state=start_state,
                mission_index=NARROW_ROUTE_INDEX,
                dynamics=dynamics,
                cost_config=cost_config,
                time_budget_ms=time_budget_ms,
                seed=seed,
                forward_action_controls=forward_action_controls,
                _allow_retry=False,
            )
        except RuntimeError as exc:
            evidence.append(
                GeometryGateEvidence(
                    candidate.snapshot.geometry_version,
                    candidate.snapshot.payload_content_hash,
                    candidate.snapshot.required_clearance,
                    False,
                    str(exc),
                )
            )
            continue
        evidence.append(
            GeometryGateEvidence(
                candidate.snapshot.geometry_version,
                candidate.snapshot.payload_content_hash,
                candidate.snapshot.required_clearance,
                True,
                "FEASIBLE",
            )
        )
        return candidate, trajectory, tuple(evidence)
    raise NarrowCompositeInfeasibleError(tuple(evidence))


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
    forward_action_controls: tuple[Control, ...] = (),
) -> FixedRoutePlan:
    """Plan and independently validate every remaining fixed route leg.

    The deterministic grid/lattice rollout supplies a feasible kinodynamic
    warm start.  With ``optimize_with_rrtstar`` enabled, the planner spends
    the remaining per-leg budget on informed sampling and rewiring.
    """

    manifest = compiled_map.manifest
    snapshot = compiled_map.snapshot
    if dynamics is None and not forward_action_controls:
        profile = diagnostic_forward_control_profile()
        dynamics = reduced_dynamics_from_profile(profile)
        forward_action_controls = profile.action_controls
    else:
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
            forward_action_controls=forward_action_controls,
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
    "NARROW_ESCAPE_TOLERANCE_M",
    "NARROW_ESCAPE_RELEASE_X_M",
    "NARROW_ESCAPE_XY",
    "NARROW_ROUTE_INDEX",
    "FixedRoutePlan",
    "GeometryGateEvidence",
    "NarrowCompositeInfeasibleError",
    "compile_offline_national_map",
    "fixed_route_continuations",
    "fixed_route_geometry_candidates",
    "fixed_route_guidance_hash",
    "fixed_route_gate_region",
    "fixed_route_goal_xy",
    "fixed_route_planning_gate",
    "fixed_route_tolerance",
    "fixed_route_waypoint_reached",
    "narrow_escape_released",
    "plan_fixed_leg",
    "plan_narrow_with_geometry_evidence",
    "plan_fixed_route",
    "build_fixed_leg_request",
]
