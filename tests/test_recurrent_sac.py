from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from usvlib4ros.policy.recurrent_sac import (
    LocalObservationV2,
    RecurrentHiddenState,
    RecurrentDiscreteSAC,
    SequenceReplay,
    SequenceTransition,
    _RecurrentHead,
)


class _UntrustedCheckpointObject:
    pass


def _observation(stamp: float = 0.0, session_id: str = "session-a") -> LocalObservationV2:
    return LocalObservationV2(
        laser_ranges=(0.8,) * 72,
        laser_valid_mask=(True,) * 72,
        scan_age_s=0.0,
        ego_features=(0.4, 0.0, 0.0, 0.0),
        path_features=(0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 8.0, 0.1),
        target_features=(0.0,) * 6,
        target_mask=(False,),
        safety_features=(1.0, 0.0, 0.0),
        event_features=(0.0, 0.0, 0.0),
        session_id=session_id,
        stamp_sim=stamp,
    )


def _transition(
    index: int,
    *,
    session_id: str = "session-a",
    action: int | None = 2,
    safety_truncation: bool = False,
    terminal: bool = False,
    timeout: bool = False,
    next_safe_action_mask: tuple[bool, ...] | None = None,
) -> SequenceTransition:
    current = _observation(float(index), session_id)
    following = _observation(float(index + 1), session_id)
    return SequenceTransition(
        observation=current,
        next_observation=following,
        executed_action=action,
        reward=1.0 if not terminal else -2.0,
        terminated=terminal,
        timeout=timeout,
        safety_truncation=safety_truncation,
        safe_action_mask=(True,) * 5 if not safety_truncation else (False,) * 5,
        hidden_reset=index == 0,
        next_safe_action_mask=next_safe_action_mask,
    )


def test_local_observation_rejects_non_finite_values_and_wrong_laser_shape():
    with pytest.raises(ValueError, match="72"):
        LocalObservationV2(
            laser_ranges=(0.8,) * 71,
            laser_valid_mask=(True,) * 71,
            scan_age_s=0.0,
        )

    with pytest.raises(ValueError, match="finite"):
        LocalObservationV2(
            laser_ranges=(math.nan,) + (0.8,) * 71,
            laser_valid_mask=(True,) * 72,
            scan_age_s=0.0,
        )


def test_sequence_replay_keeps_episode_boundaries_and_padding_masks():
    replay = SequenceReplay(capacity=8, seed=11)
    replay.add_episode([_transition(i) for i in range(3)])
    replay.add_episode([_transition(i, session_id="session-b") for i in range(2)])

    batch = replay.sample(batch_size=2, burn_in=1, unroll=3)

    assert batch.observations.shape == (2, 4, replay.observation_dim)
    assert batch.safe_action_mask.shape == (2, 4, 5)
    assert torch.all(batch.learning_mask[:, 0] == 0)
    assert torch.all(batch.padding_mask >= 0)
    assert set(batch.session_ids).issubset({"session-a", "session-b"})
    assert all(first == second for first, second in zip(batch.episode_ids, batch.episode_ids))


def test_recurrent_next_state_advance_uses_hidden_after_each_current_observation():
    torch.manual_seed(47)
    observation_dim = 6
    head = _RecurrentHead(observation_dim, hidden_dim=8, output_dim=5)
    current = torch.randn(1, 3, observation_dim)
    following = torch.randn(1, 3, observation_dim)
    reset = torch.tensor([[True, False, False]], dtype=torch.bool)

    _, _, history = head.forward_with_history(current, reset)
    continued = head.advance_from_history(following, history)
    next_reset = torch.ones((1, 3), dtype=torch.bool)
    continued_after_reset = head.advance_from_history(
        following,
        history,
        next_reset,
    )
    independently_reset, _ = head(following, next_reset)

    assert continued.shape == independently_reset.shape
    assert not torch.allclose(continued, independently_reset)
    assert torch.allclose(continued_after_reset, independently_reset)


def test_replay_preserves_next_observation_hidden_reset():
    transition = replace(
        _transition(0),
        next_observation=replace(_observation(1.0), hidden_reset=True),
    )
    replay = SequenceReplay(capacity=4, seed=5)
    replay.add_episode([transition])

    batch = replay.sample(batch_size=1, burn_in=0, unroll=1)

    assert batch.next_hidden_reset[0, 0].item() is True


def test_masked_policy_renormalizes_and_empty_mask_is_stop_candidate():
    observation = _observation()
    agent = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=7,
    )

    proposal, hidden = agent.act(
        observation,
        safe_action_mask=(True, False, True, False, False),
        deterministic=True,
    )
    assert proposal.action in (0, 2)
    assert proposal.masked_probabilities[1] == 0.0
    assert proposal.masked_probabilities[3] == 0.0
    assert proposal.masked_probabilities[4] == 0.0
    assert math.isclose(sum(proposal.masked_probabilities), 1.0, rel_tol=1e-6)
    assert hidden is not None

    stop_proposal, _ = agent.act(
        observation,
        safe_action_mask=(False,) * 5,
        deterministic=True,
    )
    assert stop_proposal.action is None
    assert stop_proposal.masked_probabilities == (0.0,) * 5


def test_act_returns_independent_actor_and_critic_hidden_streams():
    observation = _observation()
    agent = RecurrentDiscreteSAC(observation_dim=observation.feature_dim, hidden_dim=16, seed=29)

    _, hidden = agent.act(observation, (True,) * 5, deterministic=True)

    assert isinstance(hidden, RecurrentHiddenState)
    assert hidden.actor is not None
    assert hidden.critic1 is not None
    assert hidden.critic2 is not None
    assert hidden.actor is not hidden.critic1
    assert hidden.actor is not hidden.critic2
    assert hidden.critic1 is not hidden.critic2
    assert hidden.actor.shape == hidden.critic1.shape == hidden.critic2.shape


def test_policy_rejects_non_boolean_safe_action_mask_instead_of_coercing_nan():
    observation = _observation()
    agent = RecurrentDiscreteSAC(observation_dim=observation.feature_dim, hidden_dim=16, seed=31)

    with pytest.raises(ValueError, match="boolean"):
        agent.act(observation, (float("nan"), False, False, False, False))


def test_sequence_transition_rejects_actions_that_are_not_safe_or_are_safety_truncated():
    normal = _transition(0)
    with pytest.raises(ValueError, match="safe"):
        replace(normal, safe_action_mask=(False,) * 5)

    with pytest.raises(ValueError, match="safety"):
        replace(normal, safety_truncation=True)


def test_mask_aware_recurrent_sac_update_is_finite_and_excludes_stop_transition():
    observation = _observation()
    agent = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=13,
    )
    replay = SequenceReplay(capacity=16, seed=13)
    replay.add_episode(
        [_transition(i) for i in range(5)]
        + [_transition(5, action=None, safety_truncation=True)]
    )
    replay.add_episode([_transition(i) for i in range(6)])

    batch = replay.sample(batch_size=2, burn_in=1, unroll=4)
    metrics = agent.update(batch)

    assert metrics["updated"] is True
    assert metrics["bootstrap_count"] >= 0
    assert all(math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float)))
    assert all(torch.isfinite(parameter).all() for parameter in agent.parameters())


def test_recurrent_sac_rejects_non_finite_batch_before_optimizer_step():
    observation = _observation()
    agent = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=17,
    )
    replay = SequenceReplay(capacity=8, seed=17)
    replay.add_episode([_transition(i) for i in range(5)])
    batch = replay.sample(batch_size=1, burn_in=1, unroll=3)
    batch.rewards[0, -1] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        agent.update(batch)


def test_sequence_replay_rejects_transition_after_terminal_boundary():
    replay = SequenceReplay(capacity=8, seed=37)

    with pytest.raises(ValueError, match="last"):
        replay.add_episode([_transition(0, terminal=True), _transition(1)])

    with pytest.raises(ValueError, match="last"):
        replay.add_episode([_transition(0, timeout=True), _transition(1)])


def test_final_timeout_can_bootstrap_only_with_explicit_next_safe_action_mask():
    observation = _observation()
    replay = SequenceReplay(capacity=8, seed=41)
    replay.add_episode(
        [
            _transition(
                0,
                timeout=True,
                next_safe_action_mask=(True,) * 5,
            )
        ]
    )
    batch = replay.sample(batch_size=1, burn_in=0, unroll=1)
    assert torch.equal(batch.next_safe_action_mask[0, 0], torch.ones(5, dtype=torch.bool))

    agent = RecurrentDiscreteSAC(observation_dim=observation.feature_dim, hidden_dim=16, seed=43)
    metrics = agent.update(batch)
    assert metrics["updated"] is True
    assert metrics["bootstrap_count"] == 1


def test_recurrent_sac_checkpoint_round_trip_is_schema_checked_and_non_overwriting(tmp_path):
    observation = _observation()
    agent = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=19,
        observation_schema="local-observation-v2-reduced",
    )
    checkpoint = tmp_path / "rsac_v0.pt"
    agent.save_checkpoint(checkpoint)
    before = checkpoint.read_bytes()

    restored = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=23,
        observation_schema="local-observation-v2-reduced",
    )
    restored.load_checkpoint(checkpoint)
    first, _ = agent.act(observation, (True,) * 5, deterministic=True)
    second, _ = restored.act(observation, (True,) * 5, deterministic=True)
    assert first.probabilities == second.probabilities
    assert checkpoint.read_bytes() == before

    with pytest.raises(FileExistsError):
        agent.save_checkpoint(checkpoint)

    incompatible = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=29,
        observation_schema="local-observation-v2-3dof",
    )
    with pytest.raises(ValueError, match="schema"):
        incompatible.load_checkpoint(checkpoint)


def test_recurrent_sac_checkpoint_rejection_does_not_mutate_existing_agent(tmp_path):
    observation = _observation()
    source = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=47,
    )
    checkpoint = source.save_checkpoint(tmp_path / "valid.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["critic1"].pop(next(iter(payload["critic1"])))
    corrupted = tmp_path / "missing-critic-key.pt"
    torch.save(payload, corrupted)

    recipient = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=53,
    )
    before_actor = {
        name: value.detach().clone()
        for name, value in recipient.actor.state_dict().items()
    }
    assert any(
        not torch.equal(before_actor[name], value)
        for name, value in source.actor.state_dict().items()
    )

    with pytest.raises(RuntimeError, match="Missing key"):
        recipient.load_checkpoint(corrupted)

    assert all(
        torch.equal(before_actor[name], value)
        for name, value in recipient.actor.state_dict().items()
    )


def test_recurrent_sac_rejects_unsafe_checkpoint_before_schema_processing(tmp_path):
    observation = _observation()
    agent = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=57,
    )
    checkpoint = tmp_path / "unsafe.pt"
    torch.save(_UntrustedCheckpointObject(), checkpoint)
    before_actor = {
        name: value.detach().clone()
        for name, value in agent.actor.state_dict().items()
    }

    with pytest.raises(ValueError, match="safely loaded"):
        agent.load_checkpoint(checkpoint)

    assert all(
        torch.equal(before_actor[name], value)
        for name, value in agent.actor.state_dict().items()
    )


def test_recurrent_sac_rolls_back_a_non_finite_optimizer_step(monkeypatch):
    observation = _observation()
    agent = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=59,
    )
    replay = SequenceReplay(capacity=8, seed=59)
    replay.add_episode([_transition(index) for index in range(5)])
    batch = replay.sample(batch_size=1, burn_in=1, unroll=3)
    modules = (
        agent.actor,
        agent.critic1,
        agent.critic2,
        agent.target_critic1,
        agent.target_critic2,
    )
    before_states = [
        {name: value.detach().clone() for name, value in module.state_dict().items()}
        for module in modules
    ]
    before_log_alpha = agent.log_alpha.detach().clone()
    original_step = agent.critic1_optimizer.step

    def corrupting_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            next(agent.critic1.parameters()).fill_(float("nan"))
        return result

    monkeypatch.setattr(agent.critic1_optimizer, "step", corrupting_step)

    with pytest.raises(ValueError, match="finite"):
        agent.update(batch)

    for module, before_state in zip(modules, before_states):
        assert all(
            torch.equal(before_state[name], value)
            for name, value in module.state_dict().items()
        )
    assert torch.equal(agent.log_alpha.detach(), before_log_alpha)
    assert agent.training_step == 0
    assert agent.critic1_optimizer.state_dict()["state"] == {}
    assert agent.critic2_optimizer.state_dict()["state"] == {}
    assert agent.actor_optimizer.state_dict()["state"] == {}
    assert agent.alpha_optimizer.state_dict()["state"] == {}


def test_recurrent_sac_rejects_and_rolls_back_a_non_finite_optimizer_moment(monkeypatch):
    observation = _observation()
    agent = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=61,
    )
    replay = SequenceReplay(capacity=8, seed=61)
    replay.add_episode([_transition(index) for index in range(5)])
    batch = replay.sample(batch_size=1, burn_in=1, unroll=3)
    original_step = agent.critic1_optimizer.step

    def corrupting_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        optimizer_state = next(iter(agent.critic1_optimizer.state.values()))
        optimizer_state["exp_avg"].fill_(float("nan"))
        return result

    monkeypatch.setattr(agent.critic1_optimizer, "step", corrupting_step)

    with pytest.raises(ValueError, match="finite"):
        agent.update(batch)

    assert agent.training_step == 0
    assert agent.critic1_optimizer.state_dict()["state"] == {}
    assert agent.critic2_optimizer.state_dict()["state"] == {}
    assert agent.actor_optimizer.state_dict()["state"] == {}
    assert agent.alpha_optimizer.state_dict()["state"] == {}


def test_recurrent_sac_rejects_non_finite_checkpoint_optimizer_state_without_mutation(tmp_path):
    observation = _observation()
    source = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=67,
    )
    replay = SequenceReplay(capacity=8, seed=67)
    replay.add_episode([_transition(index) for index in range(5)])
    source.update(replay.sample(batch_size=1, burn_in=1, unroll=3))
    checkpoint = source.save_checkpoint(tmp_path / "valid-optimizer.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    optimizer_state = next(iter(payload["critic1_optimizer"]["state"].values()))
    optimizer_state["exp_avg"].fill_(float("nan"))
    corrupted = tmp_path / "non-finite-optimizer.pt"
    torch.save(payload, corrupted)

    recipient = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=71,
    )
    before_actor = {
        name: value.detach().clone()
        for name, value in recipient.actor.state_dict().items()
    }

    with pytest.raises(ValueError, match="finite"):
        recipient.load_checkpoint(corrupted)

    assert all(
        torch.equal(before_actor[name], value)
        for name, value in recipient.actor.state_dict().items()
    )
    assert recipient.critic1_optimizer.state_dict()["state"] == {}


def test_recurrent_sac_refuses_to_save_non_finite_optimizer_hyperparameters(tmp_path):
    observation = _observation()
    agent = RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=16,
        seed=73,
    )
    agent.critic1_optimizer.param_groups[0]["lr"] = float("nan")
    checkpoint = tmp_path / "non-finite-hyperparameter.pt"

    with pytest.raises(ValueError, match="finite"):
        agent.save_checkpoint(checkpoint)

    assert not checkpoint.exists()
