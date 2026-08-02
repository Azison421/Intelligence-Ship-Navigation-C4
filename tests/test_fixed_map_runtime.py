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
from usvlib4ros.planning import VesselState
from usvlib4ros.planning.fixed_route import (
    compile_offline_national_map,
    fixed_route_goal_xy,
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
    assert DEFAULT_CHECKPOINT.name == "national_test_sac_live_v9.pt"


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
    decision = core.step(_sample(context))

    assert not decision.stop
    assert decision.mission_index == 1
    assert decision.action is not None
    assert any(decision.safe_mask)
    assert decision.replanned

    running_core = FixedMapControllerCore(context, policy)
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


def test_runtime_advances_displaced_gate_from_original_waypoint_region():
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
    assert core.mission_index == 6


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
        '{"schema_version":"national-test-sac-checkpoint-v3",'
        '"live_ready":false}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="policy-only evaluation"):
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
                "schema_version": "national-test-sac-checkpoint-v3",
                "live_ready": True,
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
                "action_schema": "five-discrete-rudder-v1",
                "action_dim": 5,
                "dynamics_version": "obsolete-turn-in-place-model",
                "route_guidance_version": (
                    "national-test-safe-gates-feedback-v3"
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dynamics_version"):
        load_live_ready_policy(checkpoint, context)
