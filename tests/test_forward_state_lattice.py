from time import perf_counter

from usvlib4ros.planning.forward_state_lattice import ForwardStateLatticePlanner
from usvlib4ros.planning.kinodynamic_informed_rrtstar import (
    Control,
    GoalRegion,
    PlanningMapSnapshot,
    PlanningRequest,
    PrototypeReducedDynamics,
    VesselState,
)


def test_forward_state_lattice_visits_required_region_before_terminal_goal():
    world = PlanningMapSnapshot.from_rows(
        ("........",) * 8,
        snapshot_id="lattice-map",
        session_id="lattice-session",
        source_version=1,
        resolution=1.0,
        footprint_radius=0.2,
        required_clearance=0.1,
    )
    dynamics = PrototypeReducedDynamics()
    request = PlanningRequest(
        request_id="lattice-request",
        session_id="lattice-session",
        start_state=VesselState(
            x=1.5,
            y=4.5,
            yaw=0.0,
            speed=0.3,
            yaw_rate=0.0,
        ),
        goal_region=GoalRegion(
            x=6.0,
            y=4.5,
            position_tolerance=0.6,
            speed_limit=1.8,
            yaw_rate_limit=1.2,
        ),
        required_visit_regions=(
            GoalRegion(
                x=4.0,
                y=4.5,
                position_tolerance=0.5,
                speed_limit=1.8,
                yaw_rate_limit=1.2,
            ),
        ),
        map_snapshot_id="lattice-map",
        dynamics_version=dynamics.version,
        cost_config_version="cost-v1",
        time_budget_ms=2_000.0,
        seed=1,
    )
    controls = (
        Control(0.3, -0.5),
        Control(0.3, -0.3),
        Control(0.3, 0.0),
        Control(0.3, 0.3),
        Control(0.3, 0.5),
    )

    seed = ForwardStateLatticePlanner().plan(
        request,
        world,
        dynamics,
        controls,
        deadline=perf_counter() + 2.0,
    )

    assert seed is not None
    assert all(control.throttle >= 0.0 for control in seed.controls)
    samples = tuple(
        state for rollout in seed.edge_rollouts for state in rollout
    )
    required = request.required_visit_regions[0]
    assert any(required.contains(state) for state in samples)
    assert request.goal_region.contains(seed.states[-1])

