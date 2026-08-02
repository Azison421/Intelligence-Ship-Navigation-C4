"""Offline SAC training on the one fixed National_Test map.

The map and obstacle set never change.  Exploration is limited to the five
complete rudder candidates that pass the independent predictive safety mask.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from usvlib4ros.mapping import CompiledSidecarMap
from usvlib4ros.planning import (
    GoalRegion,
    PrototypeReducedDynamics,
    Trajectory,
    VesselState,
)
from usvlib4ros.planning.fixed_route import (
    ROUTE_GUIDANCE_VERSION,
    compile_offline_national_map,
    fixed_route_gate_region,
    plan_fixed_leg,
)

from .recurrent_sac import (
    LocalObservationV2,
    RecurrentDiscreteSAC,
    RecurrentHiddenState,
    SequenceReplay,
    SequenceTransition,
)
from .fixed_map_features import (
    TrajectoryPreview,
    build_fixed_map_observation,
    feedback_tracking_control,
    preview_trajectory,
)
from .safety_supervisor import (
    CandidateControl,
    CandidateControlGenerator,
    FIXED_MAP_PREDICTION_HORIZON_S,
    PredictiveSafetySupervisor,
)


LASER_COUNT = 72
OFFLINE_LASER_RANGE_M = 20.0


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    completed: bool
    safety_stop: bool
    timeout: bool
    steps: int
    mission_index: int
    total_reward: float
    minimum_clearance_m: float
    replans: int


@dataclass(frozen=True)
class TrainingSummary:
    episodes: int
    completed_episodes: int
    safety_stops: int
    total_steps: int
    updates: int
    final_actor_loss: float
    final_critic_loss: float


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    completed_episodes: int
    safety_stops: int
    timeouts: int
    total_steps: int
    minimum_clearance_m: float
    policy_mode: str = "deterministic-sac-only"

    @property
    def live_ready(self) -> bool:
        return (
            self.episodes > 0
            and self.completed_episodes == self.episodes
            and self.safety_stops == 0
            and self.timeouts == 0
        )


class FixedMapSACTrainer:
    """Train and evaluate masked recurrent discrete SAC on the fixed route."""

    def __init__(
        self,
        *,
        compiled_map: Optional[CompiledSidecarMap] = None,
        dynamics: Optional[PrototypeReducedDynamics] = None,
        seed: int = 31,
        hidden_dim: int = 32,
    ) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.compiled_map = compiled_map or compile_offline_national_map(
            session_id=f"fixed-map-training-{self.seed}",
        )
        self.dynamics = dynamics or PrototypeReducedDynamics()
        self.generator = CandidateControlGenerator()
        self.supervisor = PredictiveSafetySupervisor(
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
            max_state_age_s=1.0,
        )
        initial = self._initial_state()
        trajectory = plan_fixed_leg(
            self.compiled_map,
            start_state=initial,
            mission_index=1,
            dynamics=self.dynamics,
            seed=self.seed,
        )
        preview = self._preview(initial, trajectory, 0)
        candidates, safe_mask, _, _ = self._safe_candidates(
            initial,
            trajectory.controls[preview.nominal_control_index],
            preview,
        )
        del candidates
        observation = self._observation(
            initial,
            trajectory,
            preview,
            safe_mask,
            hidden_reset=True,
        )
        self.sac = RecurrentDiscreteSAC(
            observation_dim=observation.feature_dim,
            hidden_dim=hidden_dim,
            seed=self.seed,
            observation_schema=observation.schema_version,
        )
        self.replay = SequenceReplay(capacity=256, seed=self.seed)

    @property
    def observation_dim(self) -> int:
        return self.sac.observation_dim

    def _initial_state(self) -> VesselState:
        manifest = self.compiled_map.manifest
        first = manifest.route_points_enu[0]
        second = manifest.route_points_enu[1]
        x = first[0] - manifest.origin_enu[0]
        y = first[1] - manifest.origin_enu[1]
        next_x = second[0] - manifest.origin_enu[0]
        next_y = second[1] - manifest.origin_enu[1]
        return VesselState(
            x=x,
            y=y,
            yaw=math.atan2(next_y - y, next_x - x),
            speed=0.0,
            yaw_rate=0.0,
            stamp_sim=self.compiled_map.snapshot.stamp_sim,
        )

    def _goal(self, mission_index: int) -> GoalRegion:
        goal_x, goal_y, tolerance = fixed_route_gate_region(
            self.compiled_map,
            mission_index,
        )
        return GoalRegion(
            x=goal_x,
            y=goal_y,
            position_tolerance=tolerance,
            speed_limit=1.2,
            yaw_rate_limit=1.2,
        )

    @staticmethod
    def _preview(
        state: VesselState,
        trajectory: Trajectory,
        previous_index: int,
    ) -> TrajectoryPreview:
        return preview_trajectory(state, trajectory, previous_index)

    def _safe_candidates(
        self,
        state: VesselState,
        nominal_control,
        preview: TrajectoryPreview,
    ) -> tuple[
        tuple[CandidateControl, ...],
        tuple[bool, ...],
        tuple[str, ...],
        tuple[float, ...],
    ]:
        feedback_control = feedback_tracking_control(
            preview,
            nominal_control,
            self.dynamics,
        )
        candidates = self.generator.generate(
            feedback_control.throttle,
            feedback_control.rudder,
        )
        mask, reasons, clearances = self.supervisor.precheck(
            state,
            candidates,
            self.compiled_map.snapshot,
            self.dynamics,
            now_sim=state.stamp_sim,
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
        )
        return candidates, mask, reasons, clearances

    def _observation(
        self,
        state: VesselState,
        trajectory: Trajectory,
        preview: TrajectoryPreview,
        safe_mask: tuple[bool, ...],
        *,
        hidden_reset: bool,
    ) -> LocalObservationV2:
        return build_fixed_map_observation(
            state=state,
            preview=preview,
            safe_mask=safe_mask,
            session_id=self.compiled_map.snapshot.session_id,
            laser_ranges=(OFFLINE_LASER_RANGE_M,) * LASER_COUNT,
            laser_valid_mask=(False,) * LASER_COUNT,
            scan_age_s=0.0,
            pose_age_s=0.0,
            hidden_reset=hidden_reset,
        )

    def run_episode(
        self,
        *,
        episode: int,
        nominal_action_probability: float,
        deterministic_policy: bool = False,
        max_steps: int = 5_000,
    ) -> tuple[tuple[SequenceTransition, ...], EpisodeSummary]:
        if not 0.0 <= nominal_action_probability <= 1.0:
            raise ValueError("nominal_action_probability must be in [0, 1]")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        state = self._initial_state()
        mission_index = 1
        trajectory = plan_fixed_leg(
            self.compiled_map,
            start_state=state,
            mission_index=mission_index,
            dynamics=self.dynamics,
            seed=self.seed + episode,
        )
        trajectory_index = 0
        hidden: Optional[RecurrentHiddenState] = None
        hidden_reset = True
        transitions = []
        total_reward = 0.0
        minimum_clearance = float("inf")
        replans = 0
        completed = False
        safety_stop = False
        timed_out = False

        for step in range(max_steps):
            goal = self._goal(mission_index)
            preview = self._preview(
                state,
                trajectory,
                trajectory_index,
            )
            if preview.cross_track_error_m > 0.8:
                trajectory = plan_fixed_leg(
                    self.compiled_map,
                    start_state=state,
                    mission_index=mission_index,
                    dynamics=self.dynamics,
                    seed=self.seed + episode + step,
                )
                preview = self._preview(state, trajectory, 0)
                trajectory_index = 0
                hidden = None
                hidden_reset = True
                replans += 1
            trajectory_index = preview.state_index
            nominal = trajectory.controls[
                preview.nominal_control_index
            ]
            candidates, safe_mask, reasons, clearances = (
                self._safe_candidates(state, nominal, preview)
            )
            observation = self._observation(
                state,
                trajectory,
                preview,
                safe_mask,
                hidden_reset=hidden_reset,
            )
            if not any(safe_mask):
                next_observation = self._observation(
                    state,
                    trajectory,
                    preview,
                    (False,) * 5,
                    hidden_reset=True,
                )
                transitions.append(
                    SequenceTransition(
                        observation=observation,
                        next_observation=next_observation,
                        executed_action=None,
                        reward=-25.0,
                        terminated=False,
                        timeout=False,
                        safety_truncation=True,
                        safe_action_mask=safe_mask,
                        hidden_reset=hidden_reset,
                        next_safe_action_mask=(False,) * 5,
                    )
                )
                total_reward -= 25.0
                safety_stop = True
                break

            proposal, hidden = self.sac.act(
                observation,
                safe_mask,
                hidden=hidden,
                deterministic=deterministic_policy,
            )
            hidden_reset = False
            policy_action = proposal.action
            if (
                safe_mask[2]
                and self.rng.random() < nominal_action_probability
            ):
                policy_action = 2
            decision = self.supervisor.finalize(
                policy_action=policy_action,
                nominal_action=2,
                candidate_mask=safe_mask,
                candidates=candidates,
                snapshot_id=self.compiled_map.snapshot.snapshot_id,
                current_snapshot_id=(
                    self.compiled_map.snapshot.snapshot_id
                ),
                reasons=reasons,
                clearances=clearances,
                current_state=state,
                current_map_snapshot=self.compiled_map.snapshot,
                dynamics=self.dynamics,
                now_sim=state.stamp_sim,
                prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
            )
            if decision.stop or decision.final_action is None:
                raise RuntimeError(
                    "safety finalize stopped after a non-empty safe mask"
                )

            old_goal_distance = math.hypot(
                state.x - goal.x,
                state.y - goal.y,
            )
            next_state = self.dynamics.propagate(
                state,
                decision.control,
                0.1,
            )[-1]
            minimum_clearance = min(
                minimum_clearance,
                self.compiled_map.snapshot.clearance_at(next_state),
            )
            advanced = goal.contains(next_state)
            terminated = False
            next_hidden_reset = False
            if advanced:
                mission_index += 1
                if mission_index >= len(
                    self.compiled_map.manifest.route_points_enu
                ):
                    terminated = True
                    completed = True
                else:
                    trajectory = plan_fixed_leg(
                        self.compiled_map,
                        start_state=next_state,
                        mission_index=mission_index,
                        dynamics=self.dynamics,
                        seed=self.seed + episode + step + mission_index,
                    )
                    trajectory_index = 0
                    hidden = None
                    next_hidden_reset = True

            if terminated:
                next_preview = preview
                next_safe_mask = (False,) * 5
            else:
                next_preview = self._preview(
                    next_state,
                    trajectory,
                    trajectory_index,
                )
                next_nominal = trajectory.controls[
                    next_preview.nominal_control_index
                ]
                _, next_safe_mask, _, _ = self._safe_candidates(
                    next_state,
                    next_nominal,
                    next_preview,
                )
            next_observation = self._observation(
                next_state,
                trajectory,
                next_preview,
                next_safe_mask,
                hidden_reset=next_hidden_reset,
            )
            new_goal_distance = (
                0.0
                if terminated
                else math.hypot(
                    next_state.x - self._goal(mission_index).x,
                    next_state.y - self._goal(mission_index).y,
                )
            )
            progress_reward = (
                old_goal_distance - new_goal_distance
                if not advanced
                else 0.5
            )
            reward = (
                2.0 * progress_reward
                - 0.1 * next_preview.cross_track_error_m
                - 0.03 * abs(decision.final_action - 2)
                + 0.02 * min(minimum_clearance, 2.0)
                + (5.0 if advanced else 0.0)
                + (20.0 if terminated else 0.0)
            )
            transitions.append(
                SequenceTransition(
                    observation=observation,
                    next_observation=next_observation,
                    executed_action=decision.final_action,
                    reward=reward,
                    terminated=terminated,
                    timeout=False,
                    safety_truncation=False,
                    safe_action_mask=safe_mask,
                    hidden_reset=observation.hidden_reset,
                    next_safe_action_mask=next_safe_mask,
                )
            )
            total_reward += reward
            state = next_state
            hidden_reset = next_hidden_reset
            if terminated:
                break
        else:
            timed_out = True
            if transitions:
                last = transitions[-1]
                transitions[-1] = SequenceTransition(
                    observation=last.observation,
                    next_observation=last.next_observation,
                    executed_action=last.executed_action,
                    reward=last.reward - 10.0,
                    terminated=False,
                    timeout=True,
                    safety_truncation=False,
                    safe_action_mask=last.safe_action_mask,
                    hidden_reset=last.hidden_reset,
                    next_safe_action_mask=last.next_safe_action_mask,
                )
                total_reward -= 10.0

        summary = EpisodeSummary(
            episode=episode,
            completed=completed,
            safety_stop=safety_stop,
            timeout=timed_out,
            steps=len(transitions),
            mission_index=mission_index,
            total_reward=total_reward,
            minimum_clearance_m=(
                0.0
                if not math.isfinite(minimum_clearance)
                else minimum_clearance
            ),
            replans=replans,
        )
        return tuple(transitions), summary

    def train(
        self,
        *,
        episodes: int,
        updates_per_episode: int = 16,
        batch_size: int = 8,
        burn_in: int = 2,
        unroll: int = 8,
    ) -> tuple[TrainingSummary, tuple[EpisodeSummary, ...]]:
        if episodes <= 0 or updates_per_episode < 0:
            raise ValueError("training episode and update counts are invalid")
        episode_summaries = []
        updates = 0
        final_metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
        }
        for episode in range(episodes):
            curriculum_fraction = episode / max(1, episodes - 1)
            nominal_probability = (
                1.0
                if episode == 0
                else max(0.65, 0.95 - 0.3 * curriculum_fraction)
            )
            transitions, summary = self.run_episode(
                episode=episode,
                nominal_action_probability=nominal_probability,
                deterministic_policy=(episode == 0),
            )
            self.replay.add_episode(transitions)
            episode_summaries.append(summary)
            for _ in range(updates_per_episode):
                batch = self.replay.sample(
                    batch_size=batch_size,
                    burn_in=burn_in,
                    unroll=unroll,
                )
                final_metrics = self.sac.update(batch)
                updates += 1

        training_summary = TrainingSummary(
            episodes=episodes,
            completed_episodes=sum(
                summary.completed for summary in episode_summaries
            ),
            safety_stops=sum(
                summary.safety_stop for summary in episode_summaries
            ),
            total_steps=sum(
                summary.steps for summary in episode_summaries
            ),
            updates=updates,
            final_actor_loss=float(final_metrics["actor_loss"]),
            final_critic_loss=float(final_metrics["critic_loss"]),
        )
        return training_summary, tuple(episode_summaries)

    def evaluate(
        self,
        *,
        episodes: int = 1,
        max_steps: int = 5_000,
    ) -> tuple[EvaluationSummary, tuple[EpisodeSummary, ...]]:
        if episodes <= 0:
            raise ValueError("evaluation episodes must be positive")
        episode_summaries = []
        for index in range(episodes):
            _, summary = self.run_episode(
                episode=100_000 + index,
                nominal_action_probability=0.0,
                deterministic_policy=True,
                max_steps=max_steps,
            )
            episode_summaries.append(summary)
        evaluation = EvaluationSummary(
            episodes=episodes,
            completed_episodes=sum(
                summary.completed for summary in episode_summaries
            ),
            safety_stops=sum(
                summary.safety_stop for summary in episode_summaries
            ),
            timeouts=sum(
                summary.timeout for summary in episode_summaries
            ),
            total_steps=sum(
                summary.steps for summary in episode_summaries
            ),
            minimum_clearance_m=min(
                summary.minimum_clearance_m
                for summary in episode_summaries
            ),
        )
        return evaluation, tuple(episode_summaries)

    def save_checkpoint(
        self,
        path: Path,
        training_summary: TrainingSummary,
        evaluation_summary: Optional[EvaluationSummary] = None,
    ) -> tuple[Path, Path]:
        target = self.sac.save_checkpoint(path)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = {
            "schema_version": "national-test-sac-checkpoint-v3",
            "algorithm": "discrete-recurrent-sac",
            "dynamics_version": self.dynamics.version,
            "route_guidance_version": ROUTE_GUIDANCE_VERSION,
            "map_profile": "北湖/National_Test",
            "map_snapshot_id": self.compiled_map.snapshot.snapshot_id,
            "map_payload_hash": (
                self.compiled_map.snapshot.payload_content_hash
            ),
            "map_source_artifact_hash": (
                self.compiled_map.snapshot.source_artifact_hash
            ),
            "route_id": self.compiled_map.manifest.route_id,
            "route_version": self.compiled_map.manifest.route_version,
            "observation_schema": self.sac.observation_schema,
            "observation_dim": self.sac.observation_dim,
            "hidden_dim": self.sac.hidden_dim,
            "action_schema": self.sac.action_schema,
            "action_dim": self.sac.action_dim,
            "checkpoint_sha256": digest,
            "training_summary": asdict(training_summary),
            "evaluation_summary": (
                None
                if evaluation_summary is None
                else asdict(evaluation_summary)
            ),
            "live_ready": (
                evaluation_summary is not None
                and evaluation_summary.live_ready
            ),
        }
        manifest_path = target.with_suffix(target.suffix + ".json")
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return target, manifest_path


__all__ = [
    "EpisodeSummary",
    "EvaluationSummary",
    "FixedMapSACTrainer",
    "TrainingSummary",
]
