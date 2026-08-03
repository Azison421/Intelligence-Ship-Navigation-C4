import hashlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from usvlib4ros.mapping import GpsProjector
from usvlib4ros.navigation.fixed_map_runtime import (
    DEFAULT_CHECKPOINT,
    FixedMapControllerCore,
    RuntimeInput,
    build_live_route_context,
    load_live_ready_policy,
)
from usvlib4ros.navigation.reverse_control_calibration import (
    ReverseControlProfile,
    enable_reverse_dynamics,
)
from usvlib4ros.planning import Control, VesselState
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    reduced_dynamics_from_profile,
)
from usvlib4ros.planning.fixed_route import (
    NARROW_ROUTE_INDEX,
    compile_offline_national_map,
    fixed_route_goal_xy,
    fixed_route_guidance_hash,
    fixed_route_planning_gate,
)
from usvlib4ros.policy import RecurrentDiscreteSAC


def _live_route_and_pose():
    compiled = compile_offline_national_map(session_id="route-fixture")
    manifest = compiled.manifest
    projector = GpsProjector(*manifest.gps_origin)
    points = []
    for x, y in manifest.route_points_enu:
        lat, lng = projector.enu_to_gps(x, y)
        points.append(SimpleNamespace(lat=lat, lng=lng))
    route = SimpleNamespace(
        id=manifest.route_id,
        name=manifest.route_name,
        version=1785568402934,
        start_index=0,
        points=points,
    )
    pose = SimpleNamespace(
        lat=points[0].lat,
        lng=points[0].lng,
        yaw=0.0,
        speed=0.0,
        rotate_speed=0.0,
    )
    return route, pose


def test_runtime_defaults_to_final_map_guidance_checkpoint():
    assert DEFAULT_CHECKPOINT.name == "national_test_sac_live_v10.pt"


def _runtime_state(context):
    manifest = context.compiled_map.manifest
    first, second = manifest.route_points_enu[:2]
    x = first[0] - manifest.origin_enu[0]
    y = first[1] - manifest.origin_enu[1]
    return VesselState(
        x=x,
        y=y,
        yaw=math.atan2(second[1] - first[1], second[0] - first[0]),
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )


def _sample(context, **changes):
    values = {
        "vessel_state": _runtime_state(context),
        "laser_ranges": (20.0,) * 72,
        "laser_valid_mask": (False,) * 72,
        "pose_age_s": 0.0,
        "scan_age_s": 0.0,
        "device_age_s": 0.0,
        "work_model": 2,
        "task_status": 1,
    }
    values.update(changes)
    return RuntimeInput(**values)


def test_live_route_context_accepts_only_approved_national_route():
    route, pose = _live_route_and_pose()

    context = build_live_route_context(
        route,
        pose,
        session_id="live-route-test",
    )

    assert context.fit_residual_m < 0.05
    assert context.route_version == route.version
    assert context.compiled_map.manifest.route_id == route.id

    route.id = "wrong-route"
    with pytest.raises(ValueError, match="route id"):
        build_live_route_context(
            route,
            pose,
            session_id="wrong-route-test",
        )


def test_runtime_core_plans_then_emits_only_fresh_safe_control():
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-core-test",
    )
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=16,
        seed=31,
    )

    core = FixedMapControllerCore(context, policy)
    assert core.supervisor.prediction_horizon_s == 2.0
    hold = core.step(_sample(context))
    decision = core.step(_sample(context))

    assert hold.stop
    assert hold.reason == "PLANNING_HOLD"
    assert not decision.stop
    assert decision.mission_index == 1
    assert decision.action is not None
    assert any(decision.safe_mask)
    assert decision.replanned

    running_core = FixedMapControllerCore(context, policy)
    running_core.step(_sample(context, task_status=2))
    running = running_core.step(_sample(context, task_status=2))
    assert not running.stop

    inactive_core = FixedMapControllerCore(context, policy)
    inactive = inactive_core.step(_sample(context, task_status=0))
    assert inactive.stop
    assert inactive.reason == "TASK_INACTIVE"

    stale_core = FixedMapControllerCore(context, policy)
    stale = stale_core.step(_sample(context, scan_age_s=1.1))
    assert stale.stop
    assert stale.reason == "SCAN_STALE"

    laser_core = FixedMapControllerCore(context, policy)
    laser = laser_core.step(
        _sample(
            context,
            laser_ranges=(0.5,) + (20.0,) * 71,
            laser_valid_mask=(True,) + (False,) * 71,
        )
    )
    assert laser.stop
    assert laser.reason == "LASER_EMERGENCY_STOP"


def test_runtime_recovers_live_laser_safe_pose_without_relaxing_global_map():
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-clearance-recovery-test",
    )
    state = VesselState(
        x=38.56912943606869,
        y=73.6600965399968,
        yaw=1.102951261381084,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    assert 0.1 < context.compiled_map.snapshot.clearance_at(state) < 0.2
    assert not context.compiled_map.snapshot.is_state_valid(state)
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )

    decision = core.step(
        _sample(
            context,
            vessel_state=state,
            laser_ranges=(1.8,) * 72,
            laser_valid_mask=(True,) * 72,
        )
    )

    assert not decision.stop
    assert decision.reason == "CLEARANCE_RECOVERY"
    assert decision.control == Control(0.1, 0.0)


def test_runtime_does_not_recover_map_invalid_pose_with_emergency_laser():
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-clearance-recovery-laser-test",
    )
    state = VesselState(
        x=38.56912943606869,
        y=73.6600965399968,
        yaw=1.102951261381084,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )

    decision = core.step(
        _sample(
            context,
            vessel_state=state,
            laser_ranges=(0.6,) * 72,
            laser_valid_mask=(True,) * 72,
        )
    )

    assert decision.stop
    assert decision.reason == "STATE_INVALID"


def test_runtime_defers_initial_planning_failure_with_zero_control(monkeypatch):
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-planning-deferred-test",
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )
    monkeypatch.setattr(
        "usvlib4ros.navigation.fixed_map_runtime.plan_fixed_leg",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("forced planner timeout")
        ),
    )

    hold = core.step(_sample(context))
    decision = core.step(_sample(context))

    assert hold.stop
    assert hold.reason == "PLANNING_HOLD"
    assert decision.stop
    assert decision.reason == "PLANNING_DEFERRED"
    assert decision.control is None


def test_runtime_keeps_last_safe_trajectory_when_replan_times_out(
    monkeypatch,
):
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-replan-recovery-test",
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )
    hold = core.step(_sample(context))
    assert hold.reason == "PLANNING_HOLD"
    first = core.step(_sample(context))
    assert not first.stop
    old_trajectory = core.trajectory
    assert old_trajectory is not None
    core.trajectory_index = 0
    reference = old_trajectory.states[0]
    candidates = (
        replace(reference, x=reference.x + 1.0),
        replace(reference, x=reference.x - 1.0),
        replace(reference, y=reference.y + 1.0),
        replace(reference, y=reference.y - 1.0),
    )
    displaced = next(
        state
        for state in candidates
        if context.compiled_map.snapshot.is_state_valid(state)
    )
    monkeypatch.setattr(
        "usvlib4ros.navigation.fixed_map_runtime.plan_fixed_leg",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("forced planner timeout")
        ),
    )

    decision = core.step(_sample(context, vessel_state=displaced))

    assert core.trajectory is None
    assert decision.stop
    assert decision.reason == "REPLANNING_HOLD"

    deferred = core.step(_sample(context, vessel_state=displaced))

    assert deferred.stop
    assert deferred.reason == "PLANNING_DEFERRED"


def test_runtime_brakes_before_planning_after_a_fast_waypoint_crossing():
    route, pose = _live_route_and_pose()
    context = replace(
        build_live_route_context(
            route,
            pose,
            session_id="runtime-transition-brake-test",
        ),
        start_index=5,
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )
    gate = fixed_route_planning_gate(context.compiled_map, 5)
    previous = fixed_route_planning_gate(context.compiled_map, 4)
    fast = VesselState(
        x=gate[0],
        y=gate[1],
        yaw=math.atan2(gate[1] - previous[1], gate[0] - previous[0]),
        speed=0.4,
        yaw_rate=0.0,
        stamp_sim=context.compiled_map.snapshot.stamp_sim,
    )
    core.trajectory = object()

    braking = core.step(_sample(context, vessel_state=fast))

    assert braking.reason == "PLANNING_BRAKE"
    assert braking.control == Control(-0.4, 0.0)
    assert core.mission_index == 6
    assert core.trajectory is None

    stopped = replace(fast, speed=0.1)
    hold = core.step(_sample(context, vessel_state=stopped))

    assert hold.stop
    assert hold.reason == "PLANNING_HOLD"


def test_runtime_uses_short_horizon_for_safe_overspeed_braking():
    route, pose = _live_route_and_pose()
    context = replace(
        build_live_route_context(
            route,
            pose,
            session_id="runtime-short-brake-fallback-test",
        ),
        start_index=1,
    )
    forward = ForwardControlProfile(
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
    reverse = ReverseControlProfile(
        source_log_sha256="1" * 64,
        command_throttle=-0.4,
        command_signed_speed_mps=-0.12256225167642798,
        reverse_throttle_speed_gain=0.3064056291910699,
        max_reverse_speed_mps=0.2,
    )
    dynamics = enable_reverse_dynamics(
        reduced_dynamics_from_profile(forward),
        reverse,
    )
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=16,
        seed=31,
    )
    policy.forward_control_profile = forward
    policy.reverse_control_profile = reverse
    policy.reduced_dynamics = dynamics
    state = VesselState(
        x=38.422363707454906,
        y=72.75204248409835,
        yaw=1.5879662589748555,
        speed=0.35749474692653355,
        yaw_rate=0.021398576733705245,
        stamp_sim=27.73399999999674,
    )
    future_states = tuple(
        replace(
            state,
            x=state.x,
            y=state.y + 0.2 * index,
            stamp_sim=state.stamp_sim + 0.5 * index,
        )
        for index in range(4)
    )
    core = FixedMapControllerCore(context, policy)
    core.trajectory = SimpleNamespace(
        states=future_states,
        controls=(Control(0.22, 0.0),) * 3,
        durations=(0.5,) * 3,
        times=(0.0, 0.5, 1.0, 1.5),
    )

    decision = core.step(_sample(context, vessel_state=state))

    assert not decision.stop
    assert decision.reason == "OVERSPEED_REVERSE_BRAKE"
    assert decision.control == Control(-0.4, 0.0)


def test_runtime_records_waypoint_only_after_reaching_its_safe_gate():
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-original-waypoint-test",
    )
    context = replace(context, start_index=5)
    compiled = context.compiled_map
    goal = fixed_route_goal_xy(compiled.manifest, 5)
    gate = fixed_route_planning_gate(compiled, 5)
    candidates = []
    resolution = compiled.snapshot.resolution
    min_x = max(0, int((goal[0] - 0.5) // resolution))
    max_x = min(
        compiled.snapshot.width - 1,
        int((goal[0] + 0.5) // resolution),
    )
    min_y = max(0, int((goal[1] - 0.5) // resolution))
    max_y = min(
        compiled.snapshot.height - 1,
        int((goal[1] + 0.5) // resolution),
    )
    for cell_y in range(min_y, max_y + 1):
        for cell_x in range(min_x, max_x + 1):
            state = VesselState(
                x=(cell_x + 0.5) * compiled.snapshot.resolution,
                y=(cell_y + 0.5) * compiled.snapshot.resolution,
                yaw=0.0,
                speed=0.0,
                yaw_rate=0.0,
                stamp_sim=compiled.snapshot.stamp_sim,
            )
            if (
                math.hypot(state.x - goal[0], state.y - goal[1]) <= 0.5
                and compiled.snapshot.is_state_valid(state)
            ):
                candidates.append(
                    (math.hypot(state.x - gate[0], state.y - gate[1]), state)
                )
    gate_distance, reached_state = max(candidates, key=lambda item: item[0])
    assert gate_distance > 0.2

    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )
    assert not core._advance_reached_goals(reached_state)
    assert core.mission_index == 5

    gate_state = replace(
        reached_state,
        x=gate[0],
        y=gate[1],
    )
    assert math.hypot(
        gate_state.x - goal[0],
        gate_state.y - goal[1],
    ) <= 0.5
    assert not core._advance_reached_goals(gate_state)
    assert core.mission_index == 6


def test_narrow_waypoint_completion_preserves_composite_trajectory_and_hidden_state():
    route, pose = _live_route_and_pose()
    context = replace(
        build_live_route_context(
            route,
            pose,
            session_id="runtime-narrow-escape-test",
        ),
        start_index=NARROW_ROUTE_INDEX,
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )
    original = fixed_route_goal_xy(
        context.compiled_map.manifest,
        NARROW_ROUTE_INDEX,
    )
    gate = fixed_route_planning_gate(
        context.compiled_map,
        NARROW_ROUTE_INDEX,
    )
    reached = VesselState(
        x=gate[0],
        y=gate[1],
        yaw=0.0,
        speed=0.3,
        yaw_rate=0.0,
    )
    assert math.hypot(
        reached.x - original[0],
        reached.y - original[1],
    ) <= 0.5
    trajectory = object()
    hidden = object()
    core.trajectory = trajectory
    core.trajectory_index = 7
    core.hidden = hidden
    core.hidden_reset = False

    assert not core._advance_reached_goals(reached)
    assert core.mission_index == NARROW_ROUTE_INDEX + 1
    assert core.maneuver_phase == "ESCAPE_PENDING"
    assert core.trajectory is trajectory
    assert core.trajectory_index == 7
    assert core.hidden is hidden
    assert not core.hidden_reset


def test_runtime_executes_safe_reverse_escape_without_calling_sac():
    route, pose = _live_route_and_pose()
    context = replace(
        build_live_route_context(
            route,
            pose,
            session_id="runtime-reverse-escape-test",
        ),
        start_index=NARROW_ROUTE_INDEX + 1,
    )
    forward = ForwardControlProfile(
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
    reverse = ReverseControlProfile(
        source_log_sha256="1" * 64,
        command_throttle=-0.4,
        command_signed_speed_mps=-0.12256225167642798,
        reverse_throttle_speed_gain=0.3064056291910699,
        max_reverse_speed_mps=0.2,
    )
    dynamics = enable_reverse_dynamics(
        reduced_dynamics_from_profile(forward),
        reverse,
    )
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=16,
        seed=31,
    )
    policy.forward_control_profile = forward
    policy.reverse_control_profile = reverse
    policy.reduced_dynamics = dynamics
    policy.act = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("SAC must not replace the reverse escape primitive")
    )
    core = FixedMapControllerCore(context, policy)
    state = VesselState(
        x=31.0,
        y=99.5,
        yaw=math.pi,
        speed=-0.1,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    rollout = dynamics.propagate(state, reverse.control, 0.8)
    core.maneuver_phase = "ESCAPE_PENDING"
    core.trajectory = SimpleNamespace(
        states=(state, rollout[-1]),
        controls=(reverse.control,),
        durations=(0.8,),
        times=(0.0, 0.8),
    )
    hidden = object()
    core.hidden = hidden
    core.hidden_reset = False

    decision = core.step(_sample(context, vessel_state=state))

    assert not decision.stop
    assert decision.reason == "REVERSE_ESCAPE_NOMINAL"
    assert decision.control == reverse.control
    assert decision.action == 2
    assert core.hidden is hidden
    assert not core.hidden_reset


def test_live_policy_loader_rejects_checkpoint_without_evaluation(
    tmp_path,
):
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="checkpoint-gate-test",
    )
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    checkpoint.with_suffix(".pt.json").write_text(
        '{"schema_version":"national-test-sac-checkpoint-v4",'
        '"offline_ready":false,'
        '"live_ready":false}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="offline and Unity"):
        load_live_ready_policy(checkpoint, context)


def test_live_policy_loader_rejects_checkpoint_from_other_dynamics(tmp_path):
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="checkpoint-dynamics-gate-test",
    )
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    compiled = context.compiled_map
    checkpoint.with_suffix(".pt.json").write_text(
        json.dumps(
            {
                "schema_version": "national-test-sac-checkpoint-v4",
                "offline_ready": True,
                "live_ready": True,
                "unity_validation_log_hashes": ["a", "b", "c"],
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "route_id": compiled.manifest.route_id,
                "map_source_artifact_hash": (
                    compiled.snapshot.source_artifact_hash
                ),
                "map_payload_hash": compiled.snapshot.payload_content_hash,
                "observation_schema": "local-observation-v2-reduced",
                "observation_dim": 162,
                "action_schema": "five-discrete-forward-bias-v2",
                "action_dim": 5,
                "dynamics_version": "obsolete-turn-in-place-model",
                "route_guidance_version": (
                    "national-test-reversible-composite-v16"
                ),
                "route_guidance_hash": fixed_route_guidance_hash(
                    compiled
                ),
                "geometry_version": (
                    compiled.snapshot.geometry_version
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dynamics_version"):
        load_live_ready_policy(checkpoint, context)
