import hashlib
import json

from usvlib4ros.policy.fixed_map_trainer import FixedMapSACTrainer


def test_fixed_map_sac_trains_complete_safe_episode_and_saves_checkpoint(
    tmp_path,
):
    trainer = FixedMapSACTrainer(seed=31, hidden_dim=16)

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
        "national-test-sac-checkpoint-v3"
    )
    assert manifest["route_guidance_version"] == (
        "national-test-safe-gates-feedback-v3"
    )
    assert manifest["algorithm"] == "discrete-recurrent-sac"
    assert manifest["dynamics_version"] == trainer.dynamics.version
    assert manifest["route_id"] == trainer.compiled_map.manifest.route_id
    assert manifest["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert manifest["evaluation_summary"] is None
    assert not manifest["live_ready"]
    serialized = json.dumps(manifest).lower()
    assert "device_id" not in serialized
    assert '"host"' not in serialized
