from __future__ import annotations

from dataclasses import replace
from math import atan2, hypot, pi, sqrt
from random import Random
import subprocess
import sys
from time import perf_counter

import pytest

from usvlib4ros.planning import (
    CircularObstacle,
    Control,
    CostConfig,
    GoalRegion,
    KinodynamicInformedRRTStarPlanner,
    PlanStatus,
    PlanningMapSnapshot,
    PlanningRequest,
    PlannerConfig,
    PrototypeReducedDynamics,
    SteeringFeasibilityConfig,
    SteeringResult,
    TrajectoryValidator,
    VesselState,
)


def _world(rows: tuple[str, ...] | None = None) -> PlanningMapSnapshot:
    return PlanningMapSnapshot.from_rows(
        rows
        or (
            "............",
            "............",
            "............",
            "............",
            "............",
            "............",
            "............",
            "............",
        ),
        snapshot_id="map-v1",
        session_id="session-v1",
        source_version=1,
        resolution=1.0,
        footprint_radius=0.15,
        stamp_sim=10.0,
    )


def test_circular_obstacle_uses_exact_surface_clearance():
    world = PlanningMapSnapshot.from_rows(
        ("..........",) * 10,
        snapshot_id="circle-map",
        session_id="circle-session",
        source_version=1,
        resolution=1.0,
        footprint_radius=0.4,
        required_clearance=0.2,
        circular_obstacles=(CircularObstacle(x=5.0, y=5.0, radius=1.0),),
    )
    safe = VesselState(
        x=7.0,
        y=5.0,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )

    assert world.clearance_at(safe) == pytest.approx(0.6)
    assert world.is_state_valid(safe)
    assert not world.is_state_valid(replace(safe, x=6.5))


def test_circular_obstacles_are_bound_into_payload_hash():
    kwargs = {
        "snapshot_id": "circle-hash-map",
        "session_id": "circle-hash-session",
        "source_version": 1,
    }
    empty = PlanningMapSnapshot.from_rows(("....",) * 4, **kwargs)
    occupied = PlanningMapSnapshot.from_rows(
        ("....",) * 4,
        circular_obstacles=(CircularObstacle(x=2.0, y=2.0, radius=0.5),),
        **kwargs,
    )

    assert occupied.payload_content_hash != empty.payload_content_hash


def test_oriented_capsule_clearance_changes_with_heading_and_hash():
    kwargs = {
        "rows": ("..........",) * 10,
        "snapshot_id": "capsule-map",
        "session_id": "capsule-session",
        "source_version": 1,
        "resolution": 1.0,
        "required_clearance": 0.1,
        "circular_obstacles": (
            CircularObstacle(x=5.0, y=5.0, radius=0.1),
        ),
    }
    circle = PlanningMapSnapshot.from_rows(
        footprint_radius=0.4,
        **kwargs,
    )
    capsule = PlanningMapSnapshot.from_rows(
        footprint_radius=0.0,
        vessel_capsule_length=1.3,
        vessel_capsule_width=0.64,
        geometry_version="official-capsule-v1",
        **kwargs,
    )
    lengthwise = VesselState(
        x=4.3,
        y=5.0,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
    )
    crosswise = replace(lengthwise, yaw=pi / 2.0)

    assert capsule.payload_content_hash != circle.payload_content_hash
    assert not capsule.is_state_valid(lengthwise)
    assert capsule.is_state_valid(crosswise)
    assert capsule.clearance_at(crosswise) > capsule.clearance_at(lengthwise)


def _request(dynamics: PrototypeReducedDynamics) -> PlanningRequest:
    return PlanningRequest(
        request_id="request-v1",
        session_id="session-v1",
        start_state=VesselState(
            x=1.0,
            y=3.0,
            yaw=0.0,
            speed=0.2,
            yaw_rate=0.0,
            frame_id="map",
            stamp_sim=10.0,
        ),
        goal_region=GoalRegion(
            x=9.0,
            y=3.0,
            position_tolerance=0.7,
            speed_limit=2.0,
            yaw_rate_limit=1.2,
        ),
        map_snapshot_id="map-v1",
        dynamics_version=dynamics.version,
        cost_config_version="cost-v1",
        time_budget_ms=1200.0,
        seed=7,
        stamp_sim=10.0,
    )


def test_open_water_returns_independently_validated_dynamic_trajectory():
    dynamics = PrototypeReducedDynamics()
    request = _request(dynamics)
    result = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(max_nodes=500, goal_bias=0.5, stop_on_first_solution=True)
    ).plan(request, _world(), dynamics, CostConfig(), now_sim=10.0)

    assert result.status == PlanStatus.SUCCESS
    assert result.trajectory is not None
    assert result.trajectory.edge_rollouts
    validation = TrajectoryValidator().validate(
        result.trajectory, request, _world(), dynamics, CostConfig()
    )
    assert validation.valid
    assert validation.min_clearance >= _world().required_clearance


def test_required_visit_regions_are_sampled_in_order_before_terminal_goal():
    dynamics = PrototypeReducedDynamics()
    request = _request(dynamics)
    world = _world()
    result = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(max_nodes=500, goal_bias=0.5, stop_on_first_solution=True)
    ).plan(request, world, dynamics, CostConfig(), now_sim=10.0)
    assert result.trajectory is not None
    sampled = result.trajectory.edge_rollouts[0][1]
    required = GoalRegion(
        x=sampled.x,
        y=sampled.y,
        position_tolerance=1e-6,
        speed_limit=2.0,
        yaw_rate_limit=2.0,
    )
    visited_request = replace(request, required_visit_regions=(required,))
    validation = TrajectoryValidator().validate(
        result.trajectory,
        visited_request,
        world,
        dynamics,
        CostConfig(),
    )
    assert validation.valid

    missed_request = replace(
        request,
        required_visit_regions=(
            GoalRegion(
                x=1.0,
                y=7.0,
                position_tolerance=0.1,
                speed_limit=2.0,
                yaw_rate_limit=2.0,
            ),
        ),
    )
    missed = TrajectoryValidator().validate(
        result.trajectory,
        missed_request,
        world,
        dynamics,
        CostConfig(),
    )
    assert not missed.valid
    assert missed.reason == "REQUIRED_VISIT_NOT_MET"


def test_required_visit_uses_forward_lattice_seed_then_keeps_rrtstar_budget():
    dynamics = PrototypeReducedDynamics()
    request = replace(
        _request(dynamics),
        required_visit_regions=(
            GoalRegion(
                x=5.0,
                y=3.0,
                position_tolerance=0.5,
                speed_limit=2.0,
                yaw_rate_limit=2.0,
            ),
        ),
        time_budget_ms=250.0,
    )
    controls = (
        Control(0.3, -0.5),
        Control(0.3, -0.3),
        Control(0.3, 0.0),
        Control(0.3, 0.3),
        Control(0.3, 0.5),
    )
    result = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=100,
            stop_on_first_solution=False,
            forward_action_controls=controls,
        )
    ).plan(request, _world(), dynamics, CostConfig(), now_sim=10.0)

    assert result.trajectory is not None
    assert result.reason != "VALIDATED_FORWARD_LATTICE_SEED"
    assert sum(result.sample_counts.values()) > 0
    assert TrajectoryValidator().validate(
        result.trajectory,
        request,
        _world(),
        dynamics,
        CostConfig(),
    ).valid


def test_live_calibrated_rudder_needs_headway_and_uses_unity_sign():
    dynamics = PrototypeReducedDynamics()
    rest = VesselState(
        x=2.0,
        y=3.0,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=10.0,
    )

    stationary = dynamics.propagate(rest, Control(0.0, 1.0), 1.0)
    positive_rudder = dynamics.propagate(
        replace(rest, speed=0.36),
        Control(0.3, 0.3),
        0.5,
    )
    negative_rudder = dynamics.propagate(
        replace(rest, speed=0.36),
        Control(0.3, -0.3),
        0.5,
    )

    assert stationary[-1].x == pytest.approx(rest.x)
    assert stationary[-1].y == pytest.approx(rest.y)
    assert stationary[-1].yaw == pytest.approx(rest.yaw)
    assert stationary[-1].yaw_rate == pytest.approx(0.0)
    assert positive_rudder[-1].yaw_rate < 0.0
    assert negative_rudder[-1].yaw_rate > 0.0


def test_reduced_dynamics_can_replay_calibrated_reverse_control():
    dynamics = PrototypeReducedDynamics(
        version="reverse-test-v1",
        allow_reverse=True,
        max_reverse_speed=0.2,
        reverse_throttle_speed_gain=0.306,
    )
    start = VesselState(
        x=5.0,
        y=5.0,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
    )

    rollout = dynamics.propagate(
        start,
        Control(throttle=-0.4, rudder=0.0),
        2.0,
    )

    assert rollout[-1].speed < -0.05
    assert rollout[-1].x < start.x
    assert all(state.is_finite() for state in rollout)


def test_official_live_speed_envelope_accepts_reported_transient_speed():
    dynamics = PrototypeReducedDynamics()
    transient = VesselState(
        x=2.0,
        y=3.0,
        yaw=0.0,
        speed=1.5,
        yaw_rate=0.0,
        stamp_sim=10.0,
    )

    assert dynamics.max_speed == pytest.approx(1.8)
    assert dynamics.is_state_valid(transient)


def test_live_low_action_calibration_matches_bounded_speed_and_yaw_rate():
    dynamics = PrototypeReducedDynamics()
    rest = VesselState(
        x=2.0,
        y=3.0,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=10.0,
    )

    end = dynamics.propagate(rest, Control(0.1, 0.1), 4.0)[-1]

    assert 0.45 <= end.speed <= 0.65
    assert -0.30 <= end.yaw_rate <= -0.15


def test_grid_seed_uses_propulsion_and_physical_rudder_sign_to_recover_heading():
    dynamics = PrototypeReducedDynamics()
    request = replace(
        _request(dynamics),
        start_state=replace(
            _request(dynamics).start_state,
            x=4.0,
            y=4.0,
            yaw=pi,
            speed=0.0,
        ),
        goal_region=replace(
            _request(dynamics).goal_region,
            x=9.0,
            y=4.0,
            position_tolerance=1.0,
        ),
    )
    planner = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=24,
            grid_seed_enabled=True,
            stop_on_first_solution=True,
        )
    )

    result = planner.plan(
        request,
        _world(),
        dynamics,
        CostConfig(),
        now_sim=10.0,
    )

    assert result.trajectory is not None
    first = result.trajectory.controls[0]
    assert 0.0 < first.throttle <= 0.1
    assert 0.0 < first.rudder <= 0.1


def test_public_steering_reports_endpoint_metrics_for_reachable_target():
    dynamics = PrototypeReducedDynamics()
    source = VesselState(
        x=2.0,
        y=2.0,
        yaw=0.0,
        speed=0.2,
        yaw_rate=0.0,
        frame_id="map",
        stamp_sim=10.0,
    )
    target = dynamics.propagate(source, Control(0.5, 0.0), 0.5)[-1]
    planner = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(edge_durations=(0.5,), connect_tolerance=0.05)
    )

    result = planner.steer(source, target, _world(), dynamics, CostConfig())

    assert isinstance(result, SteeringResult)
    assert result.success
    assert result.end_state is not None
    assert result.position_error <= 0.05
    assert result.heading_error <= 2.0 * pi / 180.0
    assert result.speed_error <= 0.1
    assert result.constraint_violations == ()


def test_steering_feasibility_report_does_not_pass_below_sample_gate():
    dynamics = PrototypeReducedDynamics()
    source = VesselState(
        x=2.0,
        y=2.0,
        yaw=0.0,
        speed=0.2,
        yaw_rate=0.0,
        frame_id="map",
        stamp_sim=10.0,
    )
    target = dynamics.propagate(source, Control(0.5, 0.0), 0.5)[-1]
    planner = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(edge_durations=(0.5,), connect_tolerance=0.05)
    )

    report = planner.evaluate_steering_feasibility(
        ((source, target),),
        _world(),
        dynamics,
        CostConfig(),
        config=SteeringFeasibilityConfig(minimum_pairs=2, require_reverse=False),
    )

    assert report.forward.attempted == 1
    assert report.forward.successes == 1
    assert not report.passed
    assert "INSUFFICIENT_FORWARD_PAIRS" in report.reasons


def test_public_steering_reports_unavailable_when_no_candidate_reaches_target():
    dynamics = PrototypeReducedDynamics()
    source = VesselState(
        x=1.5,
        y=3.0,
        yaw=0.0,
        speed=0.2,
        yaw_rate=0.0,
        frame_id="map",
        stamp_sim=10.0,
    )
    target = replace(source, x=10.0)
    planner = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(edge_durations=(0.5,), connect_tolerance=0.05)
    )

    result = planner.steer(source, target, _world(), dynamics, CostConfig())

    assert not result.success
    assert result.reason == "STEERING_UNAVAILABLE"
    assert result.attempts > 0


def test_forward_steering_feasibility_spike_covers_1000_state_pairs():
    rng = Random(20260730)
    world = PlanningMapSnapshot.from_rows(
        tuple("." * 64 for _ in range(64)),
        snapshot_id="steering-spike-map-v1",
        session_id="steering-spike-session-v1",
        source_version=1,
        resolution=1.0,
        footprint_radius=0.0,
        stamp_sim=0.0,
    )
    dynamics = PrototypeReducedDynamics()
    pairs = []
    for _ in range(1000):
        source = VesselState(
            x=rng.uniform(16.0, 48.0),
            y=rng.uniform(16.0, 48.0),
            yaw=rng.uniform(-pi, pi),
            speed=rng.uniform(0.1, 1.0),
            yaw_rate=0.0,
            frame_id="map",
            stamp_sim=0.0,
        )
        target = dynamics.propagate(source, Control(0.5, 0.0), 0.5)[-1]
        pairs.append((source, target))

    report = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(edge_durations=(0.5,), connect_tolerance=0.05)
    ).evaluate_steering_feasibility(
        pairs,
        world,
        dynamics,
        CostConfig(),
        reverse_pairs=(),
        config=SteeringFeasibilityConfig(
            minimum_pairs=1000,
            p95_time_limit_ms=1000.0,
            require_reverse=False,
        ),
    )

    assert report.forward.attempted == 1000
    assert report.forward.successes == 1000
    assert report.forward.success_rate == 1.0
    assert report.passed


def test_planning_import_does_not_require_roslibpy():
    script = """
import sys

class _BlockRoslibpy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'roslibpy' or fullname.startswith('roslibpy.'):
            raise ModuleNotFoundError('blocked for pure-Python import test')
        return None

sys.meta_path.insert(0, _BlockRoslibpy())
from usvlib4ros.planning import VesselState
assert VesselState.__name__ == 'VesselState'
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_wall_case_uses_global_and_informed_sampling_and_rewire_phase():
    dynamics = PrototypeReducedDynamics()
    request = replace(
        _request(dynamics),
        time_budget_ms=5000.0,
        goal_region=GoalRegion(
            x=9.0,
            y=3.0,
            position_tolerance=0.8,
            speed_limit=2.0,
            yaw_rate_limit=1.2,
        ),
    )
    world = _world(
        (
            "............",
            "............",
            ".....#......",
            ".....#......",
            ".....#......",
            ".....#......",
            ".....#......",
            "............",
        )
    )
    result = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=900,
            goal_bias=0.3,
            global_sample_ratio=0.3,
            stop_on_first_solution=False,
            rewire_radius=2.0,
            edge_durations=(0.3, 0.5, 1.0),
        )
    ).plan(request, world, dynamics, CostConfig(), now_sim=10.0)

    assert result.status in (PlanStatus.SUCCESS, PlanStatus.TIME_BUDGET_WITH_VALID_SOLUTION)
    assert result.trajectory is not None
    assert result.sample_counts["global"] > 0
    assert result.sample_counts["informed"] > 0
    assert result.rewire_attempts > 0
    assert TrajectoryValidator().validate(
        result.trajectory, request, world, dynamics, CostConfig()
    ).valid


def test_sealed_wall_returns_no_validated_trajectory():
    dynamics = PrototypeReducedDynamics()
    request = replace(
        _request(dynamics),
        time_budget_ms=250.0,
        goal_region=GoalRegion(
            x=9.0,
            y=3.0,
            position_tolerance=0.6,
            speed_limit=2.0,
            yaw_rate_limit=1.2,
        ),
    )
    sealed = _world(tuple(".....#......" for _ in range(8)))

    result = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=120,
            goal_bias=0.3,
            global_sample_ratio=0.3,
            stop_on_first_solution=False,
            edge_durations=(0.3, 0.5, 1.0),
        )
    ).plan(request, sealed, dynamics, CostConfig(), now_sim=10.0)

    assert result.status in (PlanStatus.TIMEOUT_NO_SOLUTION, PlanStatus.NO_PATH)
    assert result.trajectory is None


def test_rewire_connector_requires_full_dynamic_state_endpoint():
    dynamics = PrototypeReducedDynamics()
    planner = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(edge_durations=(0.5,), connect_tolerance=5.0)
    )
    source = VesselState(
        x=2.0,
        y=3.0,
        yaw=0.0,
        speed=0.2,
        yaw_rate=0.0,
        stamp_sim=10.0,
    )
    target = replace(
        source,
        x=2.1,
        yaw=pi / 2.0,
        speed=0.0,
        stamp_sim=10.5,
    )

    connection, _ = planner._connect(
        source,
        target,
        GoalRegion(x=2.1, y=3.0),
        _world(),
        dynamics,
        CostConfig(),
        require_endpoint=True,
        require_exact_endpoint=True,
    )

    assert connection is None


def test_tree_duplicate_filter_keeps_same_position_distinct_dynamic_states():
    planner = KinodynamicInformedRRTStarPlanner()
    base = VesselState(
        x=2.0,
        y=3.0,
        yaw=0.0,
        speed=0.2,
        yaw_rate=0.0,
        throttle_state=0.1,
        rudder_state=0.0,
        stamp_sim=10.0,
    )

    assert planner._states_equivalent(base, base)
    assert not planner._states_equivalent(base, replace(base, yaw=0.25))
    assert not planner._states_equivalent(base, replace(base, speed=0.5))
    assert not planner._states_equivalent(base, replace(base, yaw_rate=0.2))
    assert not planner._states_equivalent(base, replace(base, throttle_state=0.4))
    assert not planner._states_equivalent(base, replace(base, rudder_state=-0.3))


def test_connector_honors_deadline_before_propagating_an_edge():
    dynamics = PrototypeReducedDynamics()
    planner = KinodynamicInformedRRTStarPlanner()
    source = VesselState(2.0, 3.0, 0.0, 0.2, 0.0, stamp_sim=10.0)
    target = replace(source, x=4.0, stamp_sim=10.5)

    connection, attempts = planner._connect(
        source,
        target,
        GoalRegion(x=4.0, y=3.0),
        _world(),
        dynamics,
        CostConfig(),
        deadline=perf_counter() - 1.0,
    )

    assert connection is None
    assert attempts == 0


def test_default_planner_continues_after_direct_goal_connection():
    dynamics = PrototypeReducedDynamics()
    request = replace(
        _request(dynamics),
        time_budget_ms=200.0,
        goal_region=GoalRegion(
            x=4.0,
            y=3.0,
            position_tolerance=0.7,
            speed_limit=2.0,
            yaw_rate_limit=1.2,
        ),
    )
    result = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=24,
            goal_bias=0.0,
            global_sample_ratio=0.25,
            edge_durations=(0.3, 0.5, 1.0, 2.0, 3.0),
        )
    ).plan(request, _world(), dynamics, CostConfig(), now_sim=10.0)

    assert result.trajectory is not None
    assert result.sample_counts["informed"] > 0
    assert result.status == PlanStatus.TIME_BUDGET_WITH_VALID_SOLUTION


def test_informed_sample_keeps_goal_tolerance_boundary_inside_the_search_set():
    dynamics = PrototypeReducedDynamics()
    cost = CostConfig()
    planner = KinodynamicInformedRRTStarPlanner()
    start = VesselState(0.0, 0.0, 0.0, 0.0, 0.0, stamp_sim=10.0)
    goal = GoalRegion(
        x=10.0,
        y=0.0,
        position_tolerance=1.0,
        speed_limit=dynamics.max_speed,
        yaw_rate_limit=dynamics.max_yaw_rate,
    )
    goal_state = VesselState(10.0, 0.0, 0.0, 0.0, 0.0, stamp_sim=10.0)
    geometric_cost_bound = 10.2
    expanded_bound = geometric_cost_bound + goal.position_tolerance
    major = expanded_bound / 2.0
    minor = sqrt(expanded_bound * expanded_bound - 100.0) / 2.0
    local_x = 5.0
    local_y = 1.0
    radius = hypot(local_x / major, local_y / minor)
    theta = atan2(local_y / minor, local_x / major)

    class FixedRandom:
        def __init__(self) -> None:
            self.values = iter((theta, 0.0, 0.0, 0.0))

        def random(self) -> float:
            return radius * radius

        def uniform(self, _low: float, _high: float) -> float:
            return next(self.values)

    sample = planner._informed_sample(
        FixedRandom(),
        start,
        goal_state,
        goal,
        _world(),
        dynamics,
        geometric_cost_bound * (cost.w_time / dynamics.max_speed + cost.w_length),
        cost,
    )

    assert sample.x == pytest.approx(10.0)
    assert sample.y == pytest.approx(1.0)


def test_planner_rejects_observed_local_map_and_non_finite_state():
    dynamics = PrototypeReducedDynamics()
    request = _request(dynamics)
    observed = replace(_world(), coverage_status="observed_local")
    result = KinodynamicInformedRRTStarPlanner().plan(
        request, observed, dynamics, CostConfig(), now_sim=10.0
    )
    assert result.status == PlanStatus.INVALID_MAP
    assert result.reason == "MAP_COVERAGE_INSUFFICIENT"

    bad_request = replace(
        request,
        start_state=replace(request.start_state, x=float("nan")),
    )
    bad = KinodynamicInformedRRTStarPlanner().plan(
        bad_request, _world(), dynamics, CostConfig(), now_sim=10.0
    )
    assert bad.status == PlanStatus.INVALID_START
    assert bad.trajectory is None


def test_map_rejects_payload_hash_that_does_not_match_its_content():
    world = _world()
    changed_rows = list(world.rows)
    changed_rows[0] = "#" + changed_rows[0][1:]

    with pytest.raises(ValueError, match="payload hash"):
        PlanningMapSnapshot.from_rows(
            tuple(changed_rows),
            snapshot_id=world.snapshot_id,
            session_id=world.session_id,
            source_version=world.source_version,
            map_frame=world.map_frame,
            resolution=world.resolution,
            footprint_radius=world.footprint_radius,
            required_clearance=world.required_clearance,
            stamp_sim=world.stamp_sim,
            source_artifact_hash=world.source_artifact_hash,
            payload_content_hash=world.payload_content_hash,
            compiler_config_hash=world.compiler_config_hash,
        )


def test_validator_rejects_tampered_trajectory_version_and_edge_rollout():
    dynamics = PrototypeReducedDynamics()
    request = _request(dynamics)
    world = _world()
    result = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(max_nodes=500, goal_bias=0.5)
    ).plan(request, world, dynamics, CostConfig(), now_sim=10.0)
    assert result.trajectory is not None

    wrong_version = replace(result.trajectory, dynamics_version="tampered")
    assert not TrajectoryValidator().validate(
        wrong_version, request, world, dynamics, CostConfig()
    ).valid

    for tampered in (
        replace(result.trajectory, state_version="tampered"),
        replace(result.trajectory, mission_index=request.mission_index + 1),
        replace(result.trajectory, mission_version="tampered"),
        replace(result.trajectory, map_source_artifact_hash="tampered"),
        replace(result.trajectory, map_compiler_config_hash="tampered"),
        replace(result.trajectory, frame_id="other-frame"),
    ):
        assert not TrajectoryValidator().validate(
            tampered, request, world, dynamics, CostConfig()
        ).valid

    rollouts = list(result.trajectory.edge_rollouts)
    first = list(rollouts[0])
    first[-1] = replace(first[-1], x=first[-1].x + 0.75)
    rollouts[0] = tuple(first)
    tampered_rollout = replace(result.trajectory, edge_rollouts=tuple(rollouts))
    validation = TrajectoryValidator().validate(
        tampered_rollout, request, world, dynamics, CostConfig()
    )
    assert not validation.valid
    assert validation.reason in {"EDGE_ROLLOUT_MISMATCH", "EDGE_ENDPOINT_MISMATCH"}


def test_planner_returns_structured_failure_for_expired_request():
    dynamics = PrototypeReducedDynamics()
    result = KinodynamicInformedRRTStarPlanner().plan(
        _request(dynamics), _world(), dynamics, CostConfig(), now_sim=16.0
    )
    assert result.status == PlanStatus.STALE_REQUEST
    assert result.reason == "REQUEST_EXPIRED"


def test_planner_reports_dynamics_error_when_every_propagation_fails():
    class BrokenDynamics(PrototypeReducedDynamics):
        def propagate(self, *_args: object, **_kwargs: object) -> tuple[VesselState, ...]:
            raise ValueError("synthetic propagation failure")

    dynamics = BrokenDynamics()
    request = replace(_request(dynamics), time_budget_ms=20.0)
    result = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(max_nodes=4, edge_durations=(0.5,), goal_bias=0.0)
    ).plan(request, _world(), dynamics, CostConfig(), now_sim=10.0)

    assert result.status == PlanStatus.DYNAMICS_ERROR
    assert result.reason == "DYNAMICS_PROPAGATION_FAILED"
