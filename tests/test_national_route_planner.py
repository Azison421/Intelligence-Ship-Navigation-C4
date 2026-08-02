"""Fixed National_Test route regressions.

The affine profile was captured from the live, read-only Get_Route response
for route version 46.  It contains no host, device id, or GPS coordinates.
"""

from __future__ import annotations

import math
from pathlib import Path

from usvlib4ros.mapping import (
    SidecarCompilerConfig,
    compile_beihu_sidecar,
    load_sidecar_artifact,
)
from usvlib4ros.planning import (
    CostConfig,
    GoalRegion,
    KinodynamicInformedRRTStarPlanner,
    PlannerConfig,
    PlanningRequest,
    PrototypeReducedDynamics,
    VesselState,
)
from usvlib4ros.planning.fixed_route import (
    compile_offline_national_map,
    fixed_route_goal_xy,
    fixed_route_planning_gate,
    fixed_route_tolerance,
    fixed_route_waypoint_reached,
    plan_fixed_leg,
    plan_fixed_route,
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


def test_planner_chains_all_fixed_route_waypoints():
    """Use each certified terminal state as the next leg's live start state."""

    compiled = compile_offline_national_map(
        session_id="national-route-chain",
    )
    route_plan = plan_fixed_route(
        compiled,
        optimize_with_rrtstar=False,
    )

    assert len(route_plan.trajectories) == 12
    assert all(
        trajectory.validation_status == "VALID"
        for trajectory in route_plan.trajectories
    )
    assert all(
        trajectory.min_clearance
        > route_plan.compiled_map.snapshot.required_clearance
        for trajectory in route_plan.trajectories
    )
    assert all(
        control.throttle <= 0.1 + 1e-9
        and abs(control.rudder) <= 0.1 + 1e-9
        for trajectory in route_plan.trajectories
        for control in trajectory.controls
    )


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

    trajectory = plan_fixed_leg(
        compiled,
        start_state=off_path_state,
        mission_index=11,
        time_budget_ms=2_000.0,
        seed=809,
    )

    assert trajectory.validation_status == "VALID"
    assert trajectory.controls
    assert (
        trajectory.min_clearance
        > compiled.snapshot.required_clearance
    )
