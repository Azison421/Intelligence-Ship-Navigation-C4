import hashlib
import inspect
import json
from types import SimpleNamespace

from usvlib4ros.navigation.reverse_control_calibration import (
    ReverseControlProfile,
)
from usvlib4ros.planning import Control
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
)
from usvlib4ros.policy.fixed_map_trainer import (
    FixedMapSACTrainer,
    LIVE_RESET_SPAWN_X_M,
    LIVE_RESET_SPAWN_Y_M,
    LIVE_RESET_SPAWN_YAW_RAD,
    advance_training_maneuver,
)
from usvlib4ros.policy.fixed_map_features import TrajectoryPreview


def test_offline_evaluation_uses_competition_step_limit():
    assert (
        inspect.signature(FixedMapSACTrainer.evaluate)
        .parameters["max_steps"]
        .default
        == 3_000
    )


def _forward_profile():
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


def _reverse_profile():
    return ReverseControlProfile(
        source_log_sha256="1" * 64,
        command_throttle=-0.4,
        command_signed_speed_mps=-0.12256225167642798,
        reverse_throttle_speed_gain=0.3064056291910699,
        max_reverse_speed_mps=0.2,
    )


def test_training_keeps_composite_state_when_narrow_target_is_reached():
    transition = advance_training_maneuver(
        mission_index=10,
        maneuver_phase="NORMAL",
        reached=True,
        route_point_count=13,
    )

    assert transition.mission_index == 11
    assert transition.maneuver_phase == "ESCAPE_PENDING"
    assert transition.task_point_advanced
    assert not transition.needs_new_plan
    assert not transition.completed

    escaped = advance_training_maneuver(
        mission_index=11,
        maneuver_phase="ESCAPE_PENDING",
        reached=True,
        route_point_count=13,
    )
    assert escaped.mission_index == 11
    assert escaped.maneuver_phase == "NORMAL"
    assert not escaped.task_point_advanced
    assert escaped.needs_new_plan
    assert not escaped.completed


def test_nominal_future_starts_after_candidate_prefix():
    first = Control(0.4, 0.2)
    second = Control(0.4, 0.0)
    preview = TrajectoryPreview(
        state_index=0,
        nominal_control_index=0,
        cross_track_error_m=0.0,
        remaining_arc_length_m=1.0,
        progress=0.0,
        lookahead_x=1.0,
        lookahead_y=0.0,
        heading_error=0.0,
    )

    future = FixedMapSACTrainer._nominal_future_controls(
        SimpleNamespace(
            controls=(first, second),
            durations=(0.8, 0.8),
        ),
        preview,
    )

    assert future[:2] == ((first, 0.5), (second, 0.8))
    assert sum(duration for _, duration in future) == 1.7


def test_fixed_map_sac_trains_complete_safe_episode_and_saves_checkpoint(
    tmp_path,
):
    trainer = FixedMapSACTrainer(
        seed=31,
        hidden_dim=16,
        forward_profile=_forward_profile(),
        reverse_profile=_reverse_profile(),
        calibration_status="calibrated",
        reverse_calibration_status="calibrated",
    )

    assert trainer.dynamics.allow_reverse
    assert trainer.planning_controls[-1] == _reverse_profile().control
    starts = tuple(
        trainer._initial_state(100_000 + index)
        for index in range(20)
    )
    assert len({(state.x, state.y, state.yaw) for state in starts}) == 20
    assert all(
        abs(state.x - LIVE_RESET_SPAWN_X_M) <= 1.0
        and abs(state.y - LIVE_RESET_SPAWN_Y_M) <= 1.0
        and abs(
            (
                state.yaw
                - LIVE_RESET_SPAWN_YAW_RAD
                + 3.141592653589793
            )
            % (2.0 * 3.141592653589793)
            - 3.141592653589793
        )
        <= 3.141592653589793 / 6.0
        and trainer.compiled_map.snapshot.clearance_at(state) >= 3.0
        for state in starts
    )

    training, episodes = trainer.train(
        episodes=1,
        updates_per_episode=1,
        batch_size=1,
        burn_in=1,
        unroll=2,
    )

    assert trainer.observation_dim == 162
    assert training.completed_episodes == 1
    assert training.safety_stops == 0
    assert training.updates == 1
    assert training.total_steps == episodes[0].steps
    assert episodes[0].completed
    assert not episodes[0].timeout
    assert episodes[0].mission_index == 13
    assert (
        episodes[0].minimum_clearance_m
        > trainer.compiled_map.snapshot.required_clearance
    )

    checkpoint, manifest_path = trainer.save_checkpoint(
        tmp_path / "national_test_sac.pt",
        training,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == (
        "national-test-sac-checkpoint-v4"
    )
    assert manifest["route_guidance_version"] == (
        "national-test-reversible-composite-v16"
    )
    assert manifest["algorithm"] == "discrete-recurrent-sac"
    assert manifest["dynamics_version"] == trainer.dynamics.version
    assert manifest["reverse_control_profile"]["profile_hash"] == (
        _reverse_profile().profile_hash
    )
    assert manifest["reverse_calibration_status"] == "calibrated"
    assert manifest["route_id"] == trainer.compiled_map.manifest.route_id
    assert manifest["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert manifest["evaluation_summary"] is None
    assert not manifest["offline_ready"]
    assert not manifest["live_ready"]
    serialized = json.dumps(manifest).lower()
    assert "device_id" not in serialized
    assert '"host"' not in serialized
