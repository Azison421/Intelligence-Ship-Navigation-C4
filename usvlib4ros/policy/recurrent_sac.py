"""Discrete mask-aware Recurrent SAC for the PRD reduced observation schema.

The implementation is deliberately self-contained and testable without ROS,
Unity or MATLAB.  It keeps policy proposal, sequence replay, SAC updates and
checkpoint manifests separate from the control/output adapter.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ACTION_COUNT = 5
LASER_COUNT = 72
CHECKPOINT_FORMAT = "recurrent-sac-v1"


def _finite_tensor(tensor: Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite")


def _finite_modules(modules: Iterable[nn.Module], name: str = "parameters") -> None:
    for module in modules:
        for parameter in module.parameters():
            _finite_tensor(parameter.detach(), name)


def _finite_value_tree(value: object, name: str) -> None:
    if isinstance(value, Tensor):
        _finite_tensor(value.detach(), name)
    elif isinstance(value, dict):
        for nested_value in value.values():
            _finite_value_tree(nested_value, name)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            _finite_value_tree(nested_value, name)
    elif isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _as_tuple(values: Sequence[float] | Iterable[float]) -> tuple[float, ...]:
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation features must be numeric") from exc


@dataclass(frozen=True)
class LocalObservationV2:
    """Reduced LocalObservationV2 with explicit laser and version fields."""

    laser_ranges: tuple[float, ...]
    laser_valid_mask: tuple[bool, ...]
    scan_age_s: float
    ego_features: tuple[float, ...] = ()
    path_features: tuple[float, ...] = ()
    target_features: tuple[float, ...] = ()
    target_mask: tuple[bool, ...] = ()
    safety_features: tuple[float, ...] = ()
    event_features: tuple[float, ...] = ()
    pose_age_s: float = 0.0
    schema_version: str = "local-observation-v2-reduced"
    session_id: str = "session-v0"
    stamp_sim: float = 0.0
    hidden_reset: bool = False

    def __post_init__(self) -> None:
        laser = _as_tuple(self.laser_ranges)
        mask = tuple(self.laser_valid_mask)
        object.__setattr__(self, "laser_ranges", laser)
        object.__setattr__(self, "laser_valid_mask", mask)
        if len(laser) != LASER_COUNT or len(mask) != LASER_COUNT:
            raise ValueError("LocalObservationV2 requires 72 laser values and masks")
        if any(not isinstance(value, bool) for value in mask):
            raise ValueError("laser_valid_mask must contain booleans")
        for field_name in (
            "ego_features",
            "path_features",
            "target_features",
            "safety_features",
            "event_features",
        ):
            object.__setattr__(self, field_name, _as_tuple(getattr(self, field_name)))
        target_mask = tuple(self.target_mask)
        object.__setattr__(self, "target_mask", target_mask)
        if any(not isinstance(value, bool) for value in target_mask):
            raise ValueError("target_mask must contain booleans")
        numeric = (
            *laser,
            self.scan_age_s,
            *self.ego_features,
            *self.path_features,
            *self.target_features,
            *self.safety_features,
            *self.event_features,
            self.pose_age_s,
            self.stamp_sim,
        )
        if not all(torch.isfinite(torch.tensor(value, dtype=torch.float32)).item() for value in numeric):
            raise ValueError("LocalObservationV2 values must be finite")
        if self.scan_age_s < 0.0 or self.pose_age_s < 0.0:
            raise ValueError("observation ages must be non-negative")
        if not self.schema_version or not self.session_id:
            raise ValueError("observation schema and session are required")

    @property
    def feature_dim(self) -> int:
        return len(self.to_vector())

    def to_vector(self) -> tuple[float, ...]:
        return (
            *self.laser_ranges,
            *(1.0 if valid else 0.0 for valid in self.laser_valid_mask),
            float(self.scan_age_s),
            *self.ego_features,
            *self.path_features,
            *self.target_features,
            *(1.0 if valid else 0.0 for valid in self.target_mask),
            *self.safety_features,
            *self.event_features,
            float(self.pose_age_s),
        )

    def tensor(self) -> Tensor:
        return torch.tensor(self.to_vector(), dtype=torch.float32)


@dataclass(frozen=True)
class PolicyProposal:
    probabilities: tuple[float, ...]
    masked_probabilities: tuple[float, ...]
    action: Optional[int]
    q1_values: tuple[float, ...]
    q2_values: tuple[float, ...]
    entropy: float
    hidden_state_input_id: str
    hidden_state_output_id: str
    observation_schema: str
    action_schema: str
    inference_ms: float

    @property
    def q_values(self) -> tuple[float, ...]:
        return tuple(min(first, second) for first, second in zip(self.q1_values, self.q2_values))


@dataclass(frozen=True)
class RecurrentHiddenState:
    """Separate recurrent streams for actor and the two critics."""

    actor: Optional[Tensor]
    critic1: Optional[Tensor]
    critic2: Optional[Tensor]


@dataclass(frozen=True)
class SequenceTransition:
    observation: LocalObservationV2
    next_observation: LocalObservationV2
    executed_action: Optional[int]
    reward: float
    terminated: bool
    timeout: bool
    safety_truncation: bool
    safe_action_mask: tuple[bool, ...]
    hidden_reset: bool = False
    next_safe_action_mask: Optional[tuple[bool, ...]] = None

    def __post_init__(self) -> None:
        if self.observation.session_id != self.next_observation.session_id:
            raise ValueError("a transition cannot cross sessions")
        for name in ("terminated", "timeout", "safety_truncation", "hidden_reset"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.terminated and self.timeout:
            raise ValueError("terminated and timeout cannot both be true")
        if self.executed_action is not None and not 0 <= self.executed_action < ACTION_COUNT:
            raise ValueError("executed_action must be one of five actions or None")
        if self.executed_action is None and not self.safety_truncation:
            raise ValueError("a missing executed action must be a safety truncation")
        safe_action_mask = tuple(self.safe_action_mask)
        object.__setattr__(self, "safe_action_mask", safe_action_mask)
        if len(safe_action_mask) != ACTION_COUNT or any(type(value) is not bool for value in safe_action_mask):
            raise ValueError("safe_action_mask must contain five booleans")
        if self.safety_truncation and self.executed_action is not None:
            raise ValueError("safety truncation cannot carry an executed action")
        if self.executed_action is not None and not safe_action_mask[self.executed_action]:
            raise ValueError("executed_action must be allowed by safe_action_mask")
        if self.next_safe_action_mask is not None:
            next_safe_action_mask = tuple(self.next_safe_action_mask)
            object.__setattr__(self, "next_safe_action_mask", next_safe_action_mask)
            if len(next_safe_action_mask) != ACTION_COUNT or any(
                type(value) is not bool for value in next_safe_action_mask
            ):
                raise ValueError("next_safe_action_mask must contain five booleans")
        if not torch.isfinite(torch.tensor(self.reward, dtype=torch.float32)).item():
            raise ValueError("reward must be finite")


@dataclass
class ReplaySequenceBatch:
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    rewards: Tensor
    terminated: Tensor
    timeout: Tensor
    safety_truncation: Tensor
    safe_action_mask: Tensor
    next_safe_action_mask: Tensor
    learning_mask: Tensor
    padding_mask: Tensor
    hidden_reset: Tensor
    next_hidden_reset: Tensor
    session_ids: tuple[str, ...]
    episode_ids: tuple[int, ...]

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.safe_action_mask.shape[-1])


class SequenceReplay:
    """Episode-aware replay that never samples across a session boundary."""

    def __init__(self, capacity: int = 1024, seed: int = 0) -> None:
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = Random(seed)
        self._episodes: list[tuple[SequenceTransition, ...]] = []
        self._next_episode_id = 0
        self.observation_dim: Optional[int] = None

    def __len__(self) -> int:
        return len(self._episodes)

    def add_episode(self, transitions: Sequence[SequenceTransition]) -> None:
        episode = tuple(transitions)
        if not episode:
            raise ValueError("episode must contain at least one transition")
        session = episode[0].observation.session_id
        dimension = episode[0].observation.feature_dim
        if any(
            transition.observation.session_id != session
            or transition.next_observation.session_id != session
            or transition.observation.feature_dim != dimension
            or transition.next_observation.feature_dim != dimension
            for transition in episode
        ):
            raise ValueError("episode observations must share session and schema")
        if self.observation_dim is None:
            self.observation_dim = dimension
        if self.observation_dim != dimension:
            raise ValueError("replay observation dimensions cannot change")
        for index, transition in enumerate(episode):
            if (transition.terminated or transition.timeout or transition.safety_truncation) and index != len(episode) - 1:
                raise ValueError("terminal or safety boundary must be the last transition")
        self._episodes.append(episode)
        if len(self._episodes) > self.capacity:
            self._episodes.pop(0)

    def sample(self, batch_size: int, burn_in: int, unroll: int) -> ReplaySequenceBatch:
        if not self._episodes:
            raise ValueError("replay is empty")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(burn_in, int) or burn_in < 0 or not isinstance(unroll, int) or unroll <= 0:
            raise ValueError("burn_in must be non-negative and unroll positive")
        assert self.observation_dim is not None
        total = burn_in + unroll
        rows: list[dict[str, object]] = []
        for _ in range(batch_size):
            episode_id = self._rng.randrange(len(self._episodes))
            episode = self._episodes[episode_id]
            start = self._rng.randrange(len(episode))
            row: dict[str, object] = {
                "obs": [],
                "next_obs": [],
                "actions": [],
                "rewards": [],
                "terminated": [],
                "timeout": [],
                "safety": [],
                "safe": [],
                "next_safe": [],
                "learning": [],
                "padding": [],
                "reset": [],
                "next_reset": [],
                "session": episode[0].observation.session_id,
                "episode_id": episode_id,
            }
            for offset in range(total):
                index = start + offset
                if index >= len(episode):
                    row["obs"].append((0.0,) * self.observation_dim)
                    row["next_obs"].append((0.0,) * self.observation_dim)
                    row["actions"].append(-1)
                    row["rewards"].append(0.0)
                    row["terminated"].append(True)
                    row["timeout"].append(False)
                    row["safety"].append(False)
                    row["safe"].append((False,) * ACTION_COUNT)
                    row["next_safe"].append((False,) * ACTION_COUNT)
                    row["learning"].append(0.0)
                    row["padding"].append(True)
                    row["reset"].append(True)
                    row["next_reset"].append(True)
                    continue
                transition = episode[index]
                next_safe = transition.next_safe_action_mask
                if next_safe is None:
                    next_safe = (
                        episode[index + 1].safe_action_mask
                        if index + 1 < len(episode)
                        else (False,) * ACTION_COUNT
                    )
                row["obs"].append(transition.observation.to_vector())
                row["next_obs"].append(transition.next_observation.to_vector())
                row["actions"].append(-1 if transition.executed_action is None else transition.executed_action)
                row["rewards"].append(float(transition.reward))
                row["terminated"].append(bool(transition.terminated))
                row["timeout"].append(bool(transition.timeout))
                row["safety"].append(bool(transition.safety_truncation))
                row["safe"].append(transition.safe_action_mask)
                row["next_safe"].append(next_safe)
                row["learning"].append(0.0 if offset < burn_in else 1.0)
                row["padding"].append(False)
                row["reset"].append(bool(transition.hidden_reset or index == 0))
                row["next_reset"].append(bool(transition.next_observation.hidden_reset))
            rows.append(row)

        def stack(name: str, dtype: torch.dtype) -> Tensor:
            return torch.tensor([row[name] for row in rows], dtype=dtype)

        return ReplaySequenceBatch(
            observations=stack("obs", torch.float32),
            next_observations=stack("next_obs", torch.float32),
            actions=stack("actions", torch.long),
            rewards=stack("rewards", torch.float32),
            terminated=stack("terminated", torch.bool),
            timeout=stack("timeout", torch.bool),
            safety_truncation=stack("safety", torch.bool),
            safe_action_mask=stack("safe", torch.bool),
            next_safe_action_mask=stack("next_safe", torch.bool),
            learning_mask=stack("learning", torch.float32),
            padding_mask=stack("padding", torch.bool),
            hidden_reset=stack("reset", torch.bool),
            next_hidden_reset=stack("next_reset", torch.bool),
            session_ids=tuple(str(row["session"]) for row in rows),
            episode_ids=tuple(int(row["episode_id"]) for row in rows),
        )


class _RecurrentHead(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(observation_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def encode(
        self,
        observations: Tensor,
        reset_mask: Optional[Tensor] = None,
        hidden: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if reset_mask is None:
            return self.gru(observations, hidden)
        outputs: list[Tensor] = []
        current_hidden = hidden
        for index in range(observations.shape[1]):
            if current_hidden is not None:
                reset = reset_mask[:, index].to(dtype=observations.dtype).view(1, -1, 1)
                current_hidden = current_hidden * (1.0 - reset)
            output, current_hidden = self.gru(
                observations[:, index : index + 1], current_hidden
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1), current_hidden

    def encode_with_history(
        self,
        observations: Tensor,
        reset_mask: Optional[Tensor] = None,
        hidden: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a sequence and retain the post-observation hidden state at each step."""
        if observations.ndim != 3:
            raise ValueError("observations must have shape [batch, time, features]")
        if reset_mask is None:
            reset_mask = torch.zeros(
                observations.shape[:2], dtype=torch.bool, device=observations.device
            )
        if reset_mask.shape != observations.shape[:2]:
            raise ValueError("reset_mask must match the batch and time dimensions")
        outputs: list[Tensor] = []
        history: list[Tensor] = []
        current_hidden = hidden
        for index in range(observations.shape[1]):
            if current_hidden is not None:
                reset = reset_mask[:, index].to(dtype=observations.dtype).view(1, -1, 1)
                current_hidden = current_hidden * (1.0 - reset)
            output, current_hidden = self.gru(
                observations[:, index : index + 1], current_hidden
            )
            outputs.append(output)
            history.append(current_hidden[0].unsqueeze(1))
        if not outputs:
            raise ValueError("observations must contain at least one time step")
        return torch.cat(outputs, dim=1), current_hidden, torch.cat(history, dim=1)

    def forward(
        self,
        observations: Tensor,
        reset_mask: Optional[Tensor] = None,
        hidden: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        encoded, output_hidden = self.encode(observations, reset_mask, hidden)
        return self.head(encoded), output_hidden

    def forward_with_history(
        self,
        observations: Tensor,
        reset_mask: Optional[Tensor] = None,
        hidden: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded, output_hidden, history = self.encode_with_history(observations, reset_mask, hidden)
        return self.head(encoded), output_hidden, history

    def advance_from_history(
        self,
        observations: Tensor,
        history: Tensor,
        reset_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate each next observation from the hidden state after its current observation."""
        if observations.ndim != 3 or history.ndim != 3:
            raise ValueError("observations and history must be rank-three tensors")
        if observations.shape[0] != history.shape[0] or observations.shape[1] != history.shape[1]:
            raise ValueError("history must match the batch and time dimensions")
        if reset_mask is not None and reset_mask.shape != observations.shape[:2]:
            raise ValueError("reset_mask must match the batch and time dimensions")
        outputs: list[Tensor] = []
        for index in range(observations.shape[1]):
            current_history = history[:, index : index + 1].transpose(0, 1)
            if reset_mask is not None:
                reset = reset_mask[:, index].to(dtype=observations.dtype).view(1, -1, 1)
                current_history = current_history * (1.0 - reset)
            encoded, _ = self.gru(
                observations[:, index : index + 1], current_history
            )
            outputs.append(self.head(encoded))
        if not outputs:
            raise ValueError("observations must contain at least one time step")
        return torch.cat(outputs, dim=1)


class RecurrentDiscreteSAC:
    """Twin-Q discrete SAC with GRU history and exact masked expectations."""

    action_schema = "five-discrete-forward-bias-v2"

    def __init__(
        self,
        observation_dim: int,
        *,
        action_dim: int = ACTION_COUNT,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        learning_rate: float = 3e-4,
        seed: int = 0,
        observation_schema: str = "local-observation-v2-reduced",
    ) -> None:
        if not isinstance(observation_dim, int) or observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if action_dim != ACTION_COUNT:
            raise ValueError("the first version requires exactly five actions")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not all(torch.isfinite(torch.tensor(value, dtype=torch.float32)).item() for value in (gamma, tau, alpha_init, learning_rate)):
            raise ValueError("SAC configuration must be finite")
        if not 0.0 < gamma <= 1.0 or not 0.0 < tau <= 1.0 or alpha_init <= 0.0 or learning_rate <= 0.0:
            raise ValueError("SAC configuration is outside bounds")
        torch.manual_seed(seed)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.observation_schema = observation_schema
        self.actor = _RecurrentHead(observation_dim, hidden_dim, action_dim)
        self.critic1 = _RecurrentHead(observation_dim, hidden_dim, action_dim)
        self.critic2 = _RecurrentHead(observation_dim, hidden_dim, action_dim)
        self.target_critic1 = _RecurrentHead(observation_dim, hidden_dim, action_dim)
        self.target_critic2 = _RecurrentHead(observation_dim, hidden_dim, action_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for parameter in (*self.target_critic1.parameters(), *self.target_critic2.parameters()):
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)
        self.log_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32).log())
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=learning_rate)
        self.training_step = 0

    def parameters(self) -> Iterable[Tensor]:
        yield from self.actor.parameters()
        yield from self.critic1.parameters()
        yield from self.critic2.parameters()
        yield from self.target_critic1.parameters()
        yield from self.target_critic2.parameters()
        yield self.log_alpha

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp().clamp(1e-6, 100.0)

    def _validate_batch(self, batch: ReplaySequenceBatch) -> None:
        if not isinstance(batch, ReplaySequenceBatch):
            raise ValueError("batch type is invalid")
        if batch.observation_dim != self.observation_dim or batch.action_dim != self.action_dim:
            raise ValueError("batch schema dimensions do not match SAC")
        for name in (
            "observations",
            "next_observations",
            "rewards",
            "learning_mask",
        ):
            _finite_tensor(getattr(batch, name), name)
        if batch.actions.dtype != torch.long:
            raise ValueError("actions must be integer encoded")
        if torch.any((batch.actions < -1) | (batch.actions >= self.action_dim)).item():
            raise ValueError("actions contain an invalid value")
        for name in (
            "terminated",
            "timeout",
            "safety_truncation",
            "safe_action_mask",
            "next_safe_action_mask",
            "padding_mask",
            "hidden_reset",
            "next_hidden_reset",
        ):
            if getattr(batch, name).dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")

    @staticmethod
    def _masked_distribution(logits: Tensor, action_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if logits.shape != action_mask.shape:
            raise ValueError("logits and action mask must have the same shape")
        mask = action_mask.to(dtype=torch.bool)
        has_safe = mask.any(dim=-1)
        safe_logits = logits.masked_fill(~mask, -1e9)
        probabilities = torch.softmax(safe_logits, dim=-1)
        probabilities = probabilities * mask.to(dtype=probabilities.dtype)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probabilities = torch.where(has_safe.unsqueeze(-1), probabilities, torch.zeros_like(probabilities))
        log_probabilities = torch.where(
            probabilities > 0.0,
            torch.log(probabilities.clamp_min(1e-12)),
            torch.zeros_like(probabilities),
        )
        return probabilities, log_probabilities, has_safe

    def act(
        self,
        observation: LocalObservationV2,
        safe_action_mask: Sequence[bool],
        *,
        hidden: Optional[Tensor | RecurrentHiddenState] = None,
        deterministic: bool = False,
    ) -> tuple[PolicyProposal, RecurrentHiddenState]:
        started = perf_counter()
        if not isinstance(observation, LocalObservationV2) or observation.schema_version != self.observation_schema:
            raise ValueError("observation schema is incompatible with SAC")
        vector = observation.tensor()
        if vector.numel() != self.observation_dim:
            raise ValueError("observation dimension is incompatible with SAC")
        try:
            mask_values = tuple(safe_action_mask)
        except TypeError as exc:
            raise ValueError("safe_action_mask must contain booleans") from exc
        if len(mask_values) != self.action_dim:
            raise ValueError("safe_action_mask must contain five actions")
        if any(type(value) is not bool for value in mask_values):
            raise ValueError("safe_action_mask must contain booleans")
        mask = torch.tensor(mask_values, dtype=torch.bool).view(1, 1, -1)
        input_id = f"{observation.session_id}:{observation.stamp_sim:.6f}"
        reset = torch.tensor([[observation.hidden_reset]], dtype=torch.bool)
        if hidden is None:
            recurrent_hidden = RecurrentHiddenState(None, None, None)
        elif isinstance(hidden, RecurrentHiddenState):
            recurrent_hidden = hidden
        elif isinstance(hidden, Tensor):
            # Accept the pre-v1 tensor form for callers upgrading in place;
            # returned state is always split into three explicit streams.
            recurrent_hidden = RecurrentHiddenState(hidden, hidden, hidden)
        else:
            raise ValueError("hidden must be a tensor or RecurrentHiddenState")
        logits, actor_hidden = self.actor(vector.view(1, 1, -1), reset, recurrent_hidden.actor)
        q1, critic1_hidden = self.critic1(
            vector.view(1, 1, -1), reset, recurrent_hidden.critic1
        )
        q2, critic2_hidden = self.critic2(
            vector.view(1, 1, -1), reset, recurrent_hidden.critic2
        )
        probabilities = torch.softmax(logits[:, -1], dim=-1)
        masked, log_probs, has_safe = self._masked_distribution(logits[:, -1:], mask)
        masked = masked[:, 0]
        log_probs = log_probs[:, 0]
        action: Optional[int]
        if not has_safe[0].item():
            action = None
            entropy = 0.0
        elif deterministic:
            action = int(torch.argmax(masked[0]).item())
            entropy = float(-(masked[0] * log_probs[0]).sum().item())
        else:
            action = int(torch.multinomial(masked[0], 1).item())
            entropy = float(-(masked[0] * log_probs[0]).sum().item())
        _finite_tensor(probabilities, "policy probabilities")
        _finite_tensor(q1, "q1")
        _finite_tensor(q2, "q2")
        return (
            PolicyProposal(
                probabilities=tuple(float(value) for value in probabilities[0].detach()),
                masked_probabilities=tuple(float(value) for value in masked[0].detach()),
                action=action,
                q1_values=tuple(float(value) for value in q1[0, -1].detach()),
                q2_values=tuple(float(value) for value in q2[0, -1].detach()),
                entropy=entropy,
                hidden_state_input_id=input_id,
                hidden_state_output_id=f"{input_id}:next",
                observation_schema=observation.schema_version,
                action_schema=self.action_schema,
                inference_ms=(perf_counter() - started) * 1000.0,
            ),
            RecurrentHiddenState(
                actor=actor_hidden.detach(),
                critic1=critic1_hidden.detach(),
                critic2=critic2_hidden.detach(),
            ),
        )

    def update(self, batch: ReplaySequenceBatch) -> dict[str, float | bool | int]:
        self._validate_batch(batch)
        _finite_modules((self.actor, self.critic1, self.critic2, self.target_critic1, self.target_critic2), "parameters")
        learning = batch.learning_mask > 0.0
        usable_action = batch.actions >= 0
        current_has_safe = batch.safe_action_mask.any(dim=-1)
        action_is_safe = batch.safe_action_mask.gather(
            -1, batch.actions.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)
        critic_mask = (
            learning
            & ~batch.padding_mask
            & usable_action
            & action_is_safe
            & current_has_safe
            & ~batch.safety_truncation
        )
        if not critic_mask.any().item():
            return {
                "updated": False,
                "reason": "NO_EXECUTED_ACTIONS",
                "critic_samples": 0,
                "bootstrap_count": 0,
            }
        reset = batch.hidden_reset
        logits, _, actor_history = self.actor.forward_with_history(batch.observations, reset)
        q1, _ = self.critic1(batch.observations, reset)
        q2, _ = self.critic2(batch.observations, reset)
        with torch.no_grad():
            next_logits = self.actor.advance_from_history(
                batch.next_observations,
                actor_history,
                batch.next_hidden_reset,
            )
            _, _, target1_history = self.target_critic1.forward_with_history(batch.observations, reset)
            _, _, target2_history = self.target_critic2.forward_with_history(batch.observations, reset)
            next_q1 = self.target_critic1.advance_from_history(
                batch.next_observations,
                target1_history,
                batch.next_hidden_reset,
            )
            next_q2 = self.target_critic2.advance_from_history(
                batch.next_observations,
                target2_history,
                batch.next_hidden_reset,
            )
        current_prob, current_log_prob, current_has_safe = self._masked_distribution(
            logits, batch.safe_action_mask
        )
        next_prob, next_log_prob, next_has_safe = self._masked_distribution(
            next_logits, batch.next_safe_action_mask
        )
        q1_taken = q1.gather(-1, batch.actions.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        q2_taken = q2.gather(-1, batch.actions.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_min_q = torch.minimum(next_q1, next_q2)
            next_value = (next_prob * (next_min_q - self.alpha.detach() * next_log_prob)).sum(dim=-1)
            bootstrap = (
                ~batch.terminated
                & ~batch.safety_truncation
                & ~batch.padding_mask
                & next_has_safe
            )
            target = batch.rewards + self.gamma * bootstrap.to(torch.float32) * next_value
        target = target.detach()
        critic_loss = F.mse_loss(q1_taken[critic_mask], target[critic_mask]) + F.mse_loss(
            q2_taken[critic_mask], target[critic_mask]
        )
        actor_mask = learning & ~batch.padding_mask & current_has_safe & ~batch.safety_truncation
        if actor_mask.any().item():
            current_min_q = torch.minimum(q1, q2).detach()
            actor_terms = (current_prob * (self.alpha.detach() * current_log_prob - current_min_q)).sum(dim=-1)
            actor_loss = actor_terms[actor_mask].mean()
            entropy = -(current_prob * current_log_prob).sum(dim=-1)[actor_mask].mean()
            safe_count = batch.safe_action_mask.to(torch.float32).sum(dim=-1).clamp_min(1.0)
            desired_entropy = (0.5 * torch.log(safe_count))[actor_mask].mean().detach()
            alpha_loss = -(self.log_alpha * (entropy.detach() - desired_entropy))
        else:
            actor_loss = torch.zeros((), dtype=torch.float32)
            entropy = torch.zeros((), dtype=torch.float32)
            alpha_loss = torch.zeros((), dtype=torch.float32)
        for loss, name in ((critic_loss, "critic_loss"), (actor_loss, "actor_loss"), (alpha_loss, "alpha_loss")):
            _finite_tensor(loss.detach(), name)

        before_update = self._snapshot_training_state()
        try:
            self.critic1_optimizer.zero_grad(set_to_none=True)
            self.critic2_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self._finite_gradients((self.critic1, self.critic2), "critic gradients")
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()

            if actor_mask.any().item():
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self._finite_gradients((self.actor,), "actor gradients")
                self.actor_optimizer.step()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self._finite_gradients((self,), "temperature gradients")
                self.alpha_optimizer.step()
            self._soft_update(self.target_critic1, self.critic1, self.tau)
            self._soft_update(self.target_critic2, self.critic2, self.tau)
            _finite_modules((self.actor, self.critic1, self.critic2, self.target_critic1, self.target_critic2), "parameters")
            _finite_tensor(self.log_alpha.detach(), "log_alpha")
            self._finite_optimizer_states()
        except Exception:
            self._restore_training_state(before_update)
            raise
        self.training_step += 1
        return {
            "updated": True,
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "entropy": float(entropy.detach().item()),
            "critic_samples": int(critic_mask.sum().item()),
            "bootstrap_count": int(bootstrap[critic_mask].sum().item()),
        }

    @staticmethod
    def _finite_gradients(modules: Iterable[nn.Module] | tuple["RecurrentDiscreteSAC", ...], name: str) -> None:
        if isinstance(modules, tuple) and modules and isinstance(modules[0], RecurrentDiscreteSAC):
            tensors = (modules[0].log_alpha,)
        else:
            tensors = tuple(parameter for module in modules for parameter in module.parameters())
        for parameter in tensors:
            if parameter.grad is not None:
                _finite_tensor(parameter.grad.detach(), name)

    def _snapshot_training_state(self) -> dict[str, object]:
        return {
            "actor": deepcopy(self.actor.state_dict()),
            "critic1": deepcopy(self.critic1.state_dict()),
            "critic2": deepcopy(self.critic2.state_dict()),
            "target_critic1": deepcopy(self.target_critic1.state_dict()),
            "target_critic2": deepcopy(self.target_critic2.state_dict()),
            "actor_optimizer": deepcopy(self.actor_optimizer.state_dict()),
            "critic1_optimizer": deepcopy(self.critic1_optimizer.state_dict()),
            "critic2_optimizer": deepcopy(self.critic2_optimizer.state_dict()),
            "alpha_optimizer": deepcopy(self.alpha_optimizer.state_dict()),
            "log_alpha": self.log_alpha.detach().clone(),
            "training_step": self.training_step,
        }

    def _restore_training_state(self, state: dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target_critic1.load_state_dict(state["target_critic1"])
        self.target_critic2.load_state_dict(state["target_critic2"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(state["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(state["critic2_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.training_step = state["training_step"]

    def _finite_optimizer_states(self) -> None:
        for name, optimizer in (
            ("actor optimizer", self.actor_optimizer),
            ("critic1 optimizer", self.critic1_optimizer),
            ("critic2 optimizer", self.critic2_optimizer),
            ("alpha optimizer", self.alpha_optimizer),
        ):
            _finite_value_tree(optimizer.state_dict(), name)

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005) -> None:
        with torch.no_grad():
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.mul_(1.0 - tau).add_(tau * source_parameter)

    def save_checkpoint(self, path: str | Path) -> Path:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
        _finite_modules((self.actor, self.critic1, self.critic2, self.target_critic1, self.target_critic2), "parameters")
        _finite_tensor(self.log_alpha.detach(), "log_alpha")
        self._finite_optimizer_states()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "observation_schema": self.observation_schema,
            "action_schema": self.action_schema,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "training_step": self.training_step,
        }
        with target.open("xb") as handle:
            torch.save(payload, handle)
        return target

    def load_checkpoint(self, path: str | Path) -> Path:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("checkpoint cannot be safely loaded") from exc
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("checkpoint format is incompatible")
        if payload.get("observation_schema") != self.observation_schema or payload.get("action_schema") != self.action_schema:
            raise ValueError("checkpoint schema is incompatible")
        if payload.get("observation_dim") != self.observation_dim or payload.get("action_dim") != self.action_dim or payload.get("hidden_dim") != self.hidden_dim:
            raise ValueError("checkpoint dimensions are incompatible")
        for name in ("actor", "critic1", "critic2", "target_critic1", "target_critic2"):
            state = payload.get(name)
            if not isinstance(state, dict):
                raise ValueError(f"checkpoint {name} is missing")
            for value in state.values():
                if isinstance(value, Tensor):
                    _finite_tensor(value, f"checkpoint {name}")
        log_alpha = payload.get("log_alpha")
        if not isinstance(log_alpha, Tensor):
            raise ValueError("checkpoint temperature is missing")
        _finite_tensor(log_alpha, "checkpoint log_alpha")
        before_load = self._snapshot_training_state()
        try:
            self.actor.load_state_dict(payload["actor"])
            self.critic1.load_state_dict(payload["critic1"])
            self.critic2.load_state_dict(payload["critic2"])
            self.target_critic1.load_state_dict(payload["target_critic1"])
            self.target_critic2.load_state_dict(payload["target_critic2"])
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic1_optimizer.load_state_dict(payload["critic1_optimizer"])
            self.critic2_optimizer.load_state_dict(payload["critic2_optimizer"])
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
            with torch.no_grad():
                self.log_alpha.copy_(log_alpha)
            self.training_step = int(payload.get("training_step", 0))
            _finite_modules((self.actor, self.critic1, self.critic2, self.target_critic1, self.target_critic2), "parameters")
            _finite_tensor(self.log_alpha.detach(), "log_alpha")
            self._finite_optimizer_states()
        except Exception:
            self._restore_training_state(before_load)
            raise
        return source


__all__ = [
    "LocalObservationV2",
    "PolicyProposal",
    "RecurrentHiddenState",
    "RecurrentDiscreteSAC",
    "ReplaySequenceBatch",
    "SequenceReplay",
    "SequenceTransition",
]
