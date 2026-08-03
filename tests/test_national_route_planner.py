"""Fixed National_Test route regressions.

The affine profile was captured from the live, read-only Get_Route response
for route version 46.  It contains no host, device id, or GPS coordinates.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest

from usvlib4ros.mapping import (
    SidecarCompilerConfig,
    compile_beihu_sidecar,
    load_sidecar_artifact,
)
from usvlib4ros.planning import (
    Control,
    CostConfig,
    GoalRegion,
    KinodynamicInformedRRTStarPlanner,
    PlannerConfig,
    PlanningRequest,
    PrototypeReducedDynamics,
    VesselState,
)
from usvlib4ros.planning.fixed_route import (
    NARROW_ESCAPE_RELEASE_X_M,
    NARROW_ESCAPE_XY,
    NARROW_ROUTE_INDEX,
    NarrowCompositeInfeasibleError,
    build_fixed_leg_request,
    compile_offline_national_map,
    fixed_route_goal_xy,
    fixed_route_geometry_candidates,
    narrow_escape_released,
    fixed_route_planning_gate,
    fixed_route_tolerance,
    fixed_route_waypoint_reached,
    plan_fixed_leg,
    plan_narrow_with_geometry_evidence,
)
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    diagnostic_forward_control_profile,
    reduced_dynamics_from_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR = (
    PROJECT_ROOT
    / "usvlib4ros"
    / "mapping"
    / "data"
    / "beihu_static_world_sidecar.json"
)
LIVE_ROUTE_AFFINE = (
    0.63215819453121191,
    -0.14629506646630225,
    0.19025639089919313,
    0.82212003241410347,
    -191.1658098488378,
    -299.80159718108752,
)


def test_live_non_collision_recovery_pose_keeps_point_two_planning_margin():
    compiled = compile_offline_national_map(
        session_id="live-point-one-margin-evidence",
    )
    recovery = VesselState(
        x=38.57,
        y=73.66,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
    )
    clearance = compiled.snapshot.clearance_at(recovery)

    assert 0.1 < clearance < 0.2
    assert compiled.snapshot.required_clearance == 0.2
    assert compiled.snapshot.geometry_version == (
        "circle-0.4-margin-0.2-live-recovery-v1"
    )
    assert not compiled.snapshot.is_state_valid(recovery)


def test_narrow_escape_releases_after_crossing_the_safe_east_plane():
    compiled = compile_offline_national_map(
        session_id="narrow-safe-release-plane",
    )
    released = VesselState(
        x=NARROW_ESCAPE_RELEASE_X_M,
        y=99.25,
        yaw=3.0,
        speed=-0.12,
        yaw_rate=0.0,
    )
    not_released = replace(
        released,
        x=NARROW_ESCAPE_RELEASE_X_M - 0.01,
    )

    assert narrow_escape_released(compiled, released)
    assert not narrow_escape_released(compiled, not_released)


def _compiled_live_route():
    artifact, artifact_hash = load_sidecar_artifact(SIDECAR)
    return compile_beihu_sidecar(
        artifact,
        source_artifact_hash=artifact_hash,
        session_id="national-route-regression",
        stamp_sim=0.0,
        config=SidecarCompilerConfig(
            transform_model="route_fitted_affine",
            coverage_status="complete_prior",
            promotion_note="operator-authorization:offline-live-route-regression",
            fitted_affine=LIVE_ROUTE_AFFINE,
        ),
    )


def _route_planner():
    return KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=600,
            edge_durations=(0.2, 0.5, 1.0, 2.0),
            goal_bias=0.25,
            global_sample_ratio=0.3,
            rewire_radius=2.5,
            connect_tolerance=1.2,
            stop_on_first_solution=True,
            grid_seed_enabled=True,
            max_request_age_s=60.0,
            max_map_age_s=1.0e9,
        )
    )


def _formal_profile_shape() -> ForwardControlProfile:
    """Deterministic unit fixture matching the approved live profile shape."""

    return ForwardControlProfile(
        calibration_hash="0" * 64,
        minimum_steerage_throttle=0.1,
        cruise_throttle=0.4,
        action_controls=(
            Control(0.1, -0.5),
            Control(0.4, -0.2),
            Control(0.4, 0.0),
            Control(0.4, 0.2),
            Control(0.4, 0.5),
        ),
        throttle_speed_gain=1.2681317113395243,
        positive_rudder_yaw_rate_gain=1.962635624471142,
        negative_rudder_yaw_rate_gain=2.048615259634089,
    )


def test_fixed_route_preserves_published_yellow_waypoint_coordinates():
    """Safety inflation may constrain approach, but must not move the task."""

    compiled = compile_offline_national_map(
        session_id="national-route-waypoint-semantics",
    )
    manifest = compiled.manifest
    snapshot = compiled.snapshot

    for mission_index, point_enu in enumerate(manifest.route_points_enu):
        expected = (
            point_enu[0] - manifest.origin_enu[0],
            point_enu[1] - manifest.origin_enu[1],
        )
        assert fixed_route_goal_xy(manifest, mission_index) == expected

        cell = snapshot._cell_for(*expected)
        assert cell is not None
        assert snapshot.rows[cell[1]][cell[0]] == "."
        gate = fixed_route_planning_gate(compiled, mission_index)
        gate_state = VesselState(
            x=gate[0],
            y=gate[1],
            yaw=0.0,
            speed=0.0,
            yaw_rate=0.0,
            stamp_sim=snapshot.stamp_sim,
        )
        assert snapshot.is_state_valid(gate_state)
        assert snapshot.clearance_at(gate_state) >= 0.3
        assert math.hypot(
            gate[0] - expected[0],
            gate[1] - expected[1],
        ) <= 0.5
        assert fixed_route_tolerance(compiled, mission_index) == 0.5


def test_waypoint_reach_requires_ship_centre_within_point_five_metres():
    compiled = compile_offline_national_map(
        session_id="national-route-reach-boundary",
    )
    goal_x, goal_y = fixed_route_goal_xy(compiled.manifest, 0)
    on_boundary = VesselState(
        x=goal_x + 0.5,
        y=goal_y,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=compiled.snapshot.stamp_sim,
    )

    assert fixed_route_waypoint_reached(compiled, 0, on_boundary)
    assert not fixed_route_waypoint_reached(
        compiled,
        0,
        VesselState(
            x=goal_x + 0.5001,
            y=goal_y,
            yaw=0.0,
            speed=0.0,
            yaw_rate=0.0,
            stamp_sim=compiled.snapshot.stamp_sim,
        ),
    )


def test_narrow_point_is_one_visit_then_east_escape_planning_request():
    compiled = compile_offline_national_map(
        session_id="national-route-narrow-composite",
    )
    dynamics = PrototypeReducedDynamics()
    previous = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX - 1,
    )
    original = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX,
    )
    next_goal = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX + 1,
    )
    start = VesselState(
        x=previous[0],
        y=previous[1],
        yaw=math.atan2(original[1] - previous[1], original[0] - previous[0]),
        speed=0.3,
        yaw_rate=0.0,
    )
    request = build_fixed_leg_request(
        compiled,
        start_state=start,
        mission_index=NARROW_ROUTE_INDEX,
        dynamics=dynamics,
        cost_config=CostConfig(),
        time_budget_ms=5_000.0,
        seed=31,
        lookahead_count=0,
    )

    assert (request.goal_region.x, request.goal_region.y) == NARROW_ESCAPE_XY
    assert len(request.required_visit_regions) == 2
    assert (
        request.required_visit_regions[1].x,
        request.required_visit_regions[1].y,
        request.required_visit_regions[1].position_tolerance,
    ) == (*original, 0.5)
    direct = compiled.snapshot.check_motion(
        (
            VesselState(
                x=request.required_visit_regions[0].x,
                y=request.required_visit_regions[0].y,
                yaw=0.0,
                speed=0.3,
                yaw_rate=0.0,
            ),
            VesselState(
                x=next_goal[0],
                y=next_goal[1],
                yaw=0.0,
                speed=0.3,
                yaw_rate=0.0,
            ),
        )
    )
    assert not direct.valid


def test_escape_replan_does_not_require_revisiting_completed_narrow_point():
    compiled = compile_offline_national_map(
        session_id="national-route-escape-replan",
    )
    dynamics = PrototypeReducedDynamics()
    original = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX,
    )
    request = build_fixed_leg_request(
        compiled,
        start_state=VesselState(
            x=original[0],
            y=original[1],
            yaw=0.0,
            speed=0.3,
            yaw_rate=0.0,
        ),
        mission_index=NARROW_ROUTE_INDEX,
        dynamics=dynamics,
        cost_config=CostConfig(),
        time_budget_ms=5_000.0,
        seed=31,
        lookahead_count=0,
        narrow_visit_completed=True,
    )

    assert (request.goal_region.x, request.goal_region.y) == NARROW_ESCAPE_XY
    assert request.required_visit_regions == ()


def test_geometry_evidence_gate_is_circle_then_capsule_then_point_one_margin():
    compiled = compile_offline_national_map(
        session_id="national-route-geometry-gate",
    )

    candidates = fixed_route_geometry_candidates(compiled)

    assert tuple(
        candidate.snapshot.geometry_version for candidate in candidates
    ) == (
        "circle-0.4-margin-0.2-v1",
        "official-capsule-1.3x0.64-margin-0.2-v1",
        "official-capsule-1.3x0.64-margin-0.1-v1",
    )
    assert tuple(
        candidate.snapshot.required_clearance for candidate in candidates
    ) == (0.2, 0.2, 0.1)
    assert len(
        {candidate.snapshot.payload_content_hash for candidate in candidates}
    ) == 3


def test_narrow_composite_fails_closed_after_all_approved_geometry_gates():
    compiled = compile_offline_national_map(
        session_id="national-route-narrow-trajectory",
    )
    previous = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX - 1,
    )
    gate = fixed_route_planning_gate(compiled, NARROW_ROUTE_INDEX)
    start = VesselState(
        x=previous[0],
        y=previous[1],
        yaw=math.atan2(gate[1] - previous[1], gate[0] - previous[0]),
        speed=0.3,
        yaw_rate=0.0,
    )

    with pytest.raises(NarrowCompositeInfeasibleError) as captured:
        plan_narrow_with_geometry_evidence(
            compiled,
            start_state=start,
            time_budget_ms=400.0,
            seed=71,
            forward_action_controls=(
                diagnostic_forward_control_profile().action_controls
            ),
        )

    evidence = captured.value.evidence
    assert len(evidence) == 3
    assert not any(item.feasible for item in evidence)
    assert tuple(item.geometry_version for item in evidence) == (
        "circle-0.4-margin-0.2-v1",
        "official-capsule-1.3x0.64-margin-0.2-v1",
        "official-capsule-1.3x0.64-margin-0.1-v1",
    )


def test_narrow_composite_can_reverse_through_its_single_entry():
    compiled = compile_offline_national_map(
        session_id="national-route-reverse-escape",
    )
    profile = _formal_profile_shape()
    base_dynamics = reduced_dynamics_from_profile(profile)
    dynamics = replace(
        base_dynamics,
        version=f"{base_dynamics.version}-reverse-v1",
        allow_reverse=True,
        max_reverse_speed=0.2,
        reverse_throttle_speed_gain=0.306,
    )
    previous = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX - 1,
    )
    gate = fixed_route_planning_gate(compiled, NARROW_ROUTE_INDEX)
    start = VesselState(
        x=previous[0],
        y=previous[1],
        yaw=math.atan2(gate[1] - previous[1], gate[0] - previous[0]),
        speed=0.3,
        yaw_rate=0.0,
    )
    reverse = Control(-0.4, 0.0)

    trajectory = plan_fixed_leg(
        compiled,
        start_state=start,
        mission_index=NARROW_ROUTE_INDEX,
        dynamics=dynamics,
        forward_action_controls=(*profile.action_controls, reverse),
        time_budget_ms=5_000.0,
        seed=71,
        _allow_retry=False,
    )

    original = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX,
    )
    assert reverse in trajectory.controls
    assert any(
        math.hypot(state.x - original[0], state.y - original[1]) <= 0.5
        for rollout in trajectory.edge_rollouts
        for state in rollout
    )
    assert math.hypot(
        trajectory.states[-1].x - NARROW_ESCAPE_XY[0],
        trajectory.states[-1].y - NARROW_ESCAPE_XY[1],
    ) <= 0.3


@pytest.mark.parametrize(
    "entry_state",
    (
        (
            32.95171342055915,
            99.26520585035696,
            2.819942972022514,
            0.44881696654570113,
            0.6233649894781259,
            0.1,
            -0.4508191167693636,
        ),
        (
            32.92501609984386,
            99.23545963004646,
            2.896899542842471,
            0.5072523199396324,
            0.34187009513995664,
            0.4,
            -0.2,
        ),
    ),
)
def test_narrow_composite_recovers_a_safe_high_yaw_rate_entry_state(
    entry_state,
):
    compiled = compile_offline_national_map(
        session_id="national-route-reverse-entry-recovery",
    )
    profile = _formal_profile_shape()
    base_dynamics = reduced_dynamics_from_profile(profile)
    dynamics = replace(
        base_dynamics,
        version=f"{base_dynamics.version}-reverse-v1",
        allow_reverse=True,
        max_reverse_speed=0.2,
        reverse_throttle_speed_gain=0.306,
    )
    start = VesselState(
        x=entry_state[0],
        y=entry_state[1],
        yaw=entry_state[2],
        speed=entry_state[3],
        yaw_rate=entry_state[4],
        throttle_state=entry_state[5],
        rudder_state=entry_state[6],
    )
    reverse = Control(-0.4, 0.0)

    trajectory = plan_fixed_leg(
        compiled,
        start_state=start,
        mission_index=NARROW_ROUTE_INDEX,
        dynamics=dynamics,
        forward_action_controls=(*profile.action_controls, reverse),
        time_budget_ms=5_000.0,
        seed=100,
        _allow_retry=False,
    )

    original = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX,
    )
    assert compiled.snapshot.check_motion(
        trajectory.edge_rollouts[0]
    ).valid
    assert any(
        math.hypot(state.x - original[0], state.y - original[1]) <= 0.5
        for rollout in trajectory.edge_rollouts
        for state in rollout
    )
    assert reverse in trajectory.controls
    assert math.hypot(
        trajectory.states[-1].x - NARROW_ESCAPE_XY[0],
        trajectory.states[-1].y - NARROW_ESCAPE_XY[1],
    ) <= 0.3


def test_planner_solves_first_buoy_gate_with_kinodynamic_seed():
    """The first fixed leg needs a bend around a buoy, not a straight edge."""

    compiled = _compiled_live_route()
    snapshot = compiled.snapshot
    manifest = compiled.manifest
    dynamics = PrototypeReducedDynamics()
    cost = CostConfig()
    (start_e, goal_e) = manifest.route_points_enu[:2]
    sx = start_e[0] - manifest.origin_enu[0]
    sy = start_e[1] - manifest.origin_enu[1]
    gx = goal_e[0] - manifest.origin_enu[0]
    gy = goal_e[1] - manifest.origin_enu[1]
    yaw = math.atan2(gy - sy, gx - sx)
    start = VesselState(
        x=sx,
        y=sy,
        yaw=yaw,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    request = PlanningRequest(
        request_id="national-leg-0-1",
        session_id=snapshot.session_id,
        start_state=start,
        goal_region=GoalRegion(
            x=gx,
            y=gy,
            position_tolerance=2.5,
            speed_limit=1.2,
            yaw_rate_limit=1.2,
        ),
        map_snapshot_id=snapshot.snapshot_id,
        dynamics_version=dynamics.version,
        cost_config_version=cost.version,
        time_budget_ms=2_000.0,
        seed=31,
        mission_index=1,
        stamp_sim=0.0,
        mission_version=f"route-v{manifest.route_version}",
    )
    planner = _route_planner()

    result = planner.plan(
        request,
        snapshot,
        dynamics,
        cost,
        now_sim=0.0,
    )

    assert result.trajectory is not None, (
        f"fixed route leg failed: {result.status.value} {result.reason}"
    )
    assert result.trajectory.validation_status == "VALID"
    assert request.goal_region.contains(result.trajectory.states[-1])
    assert result.trajectory.min_clearance > snapshot.required_clearance


def test_fixed_leg_uses_only_calibrated_forward_motion_primitives():
    compiled = compile_offline_national_map(
        session_id="national-route-profile-controls",
    )
    profile = _formal_profile_shape()
    dynamics = reduced_dynamics_from_profile(profile)
    start_xy = fixed_route_goal_xy(compiled.manifest, 0)
    goal_xy = fixed_route_goal_xy(compiled.manifest, 1)
    start = VesselState(
        x=start_xy[0],
        y=start_xy[1],
        yaw=math.atan2(
            goal_xy[1] - start_xy[1],
            goal_xy[0] - start_xy[0],
        ),
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )

    trajectory = plan_fixed_leg(
        compiled,
        start_state=start,
        mission_index=1,
        dynamics=dynamics,
        forward_action_controls=profile.action_controls,
        time_budget_ms=5_000.0,
        optimize_with_rrtstar=False,
    )

    assert trajectory.controls
    assert set(trajectory.controls) <= set(profile.action_controls)
    assert trajectory.times[-1] <= 300.0


def test_planner_chains_ordinary_legs_until_narrow_geometry_gate():
    """Ordinary legs must use calibrated primitives before the known blocker."""

    compiled = compile_offline_national_map(
        session_id="national-route-chain",
    )
    profile = _formal_profile_shape()
    dynamics = reduced_dynamics_from_profile(profile)
    start_xy = fixed_route_goal_xy(compiled.manifest, 0)
    goal_xy = fixed_route_goal_xy(compiled.manifest, 1)
    state = VesselState(
        x=start_xy[0],
        y=start_xy[1],
        yaw=math.atan2(
            goal_xy[1] - start_xy[1],
            goal_xy[0] - start_xy[0],
        ),
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    trajectories = []
    for mission_index in range(1, NARROW_ROUTE_INDEX):
        trajectory = plan_fixed_leg(
            compiled,
            start_state=state,
            mission_index=mission_index,
            dynamics=dynamics,
            forward_action_controls=profile.action_controls,
            time_budget_ms=5_000.0,
            optimize_with_rrtstar=False,
        )
        trajectories.append(trajectory)
        state = trajectory.states[-1]

    assert len(trajectories) == NARROW_ROUTE_INDEX - 1
    assert all(
        trajectory.validation_status == "VALID"
        for trajectory in trajectories
    )
    assert all(
        trajectory.min_clearance
        > compiled.snapshot.required_clearance
        for trajectory in trajectories
    )
    assert all(
        control in profile.action_controls
        for trajectory in trajectories
        for control in trajectory.controls
    )
    assert sum(trajectory.times[-1] for trajectory in trajectories) < 300.0


def test_fixed_leg_recovers_after_safe_policy_exploration():
    compiled = compile_offline_national_map(
        session_id="national-route-recovery",
    )
    off_path_state = VesselState(
        x=34.992242240145565,
        y=99.4049267448261,
        yaw=2.2196044059495414,
        speed=0.12,
        yaw_rate=0.0,
        throttle_state=0.05,
        rudder_state=0.0,
        stamp_sim=76.8,
    )
    profile = _formal_profile_shape()

    trajectory = plan_fixed_leg(
        compiled,
        start_state=off_path_state,
        mission_index=11,
        dynamics=reduced_dynamics_from_profile(profile),
        forward_action_controls=profile.action_controls,
        time_budget_ms=5_000.0,
        seed=809,
    )

    assert trajectory.validation_status == "VALID"
    assert trajectory.controls
    assert (
        trajectory.min_clearance
        > compiled.snapshot.required_clearance
    )
