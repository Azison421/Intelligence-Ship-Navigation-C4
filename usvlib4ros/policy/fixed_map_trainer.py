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
from usvlib4ros.navigation.reverse_control_calibration import (
    ReverseControlProfile,
    enable_reverse_dynamics,
    reverse_control_profile_to_dict,
)
from usvlib4ros.planning import (
    Control,
    GoalRegion,
    PrototypeReducedDynamics,
    Trajectory,
    VesselState,
)
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    action_protocol_hash,
    diagnostic_forward_control_profile,
    reduced_dynamics_from_profile,
)
from usvlib4ros.planning.fixed_route import (
    NARROW_ESCAPE_TOLERANCE_M,
    NARROW_ESCAPE_XY,
    NARROW_ROUTE_INDEX,
    ROUTE_GUIDANCE_VERSION,
    compile_offline_national_map,
    fixed_route_goal_xy,
    fixed_route_gate_region,
    fixed_route_guidance_hash,
    fixed_route_tolerance,
    narrow_escape_released,
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
    braking_future_controls,
    build_fixed_map_observation,
    feedback_tracking_control,
    narrow_ingress_control,
    narrow_ingress_future_controls,
    preview_trajectory,
    reverse_tracking_control,
    tracking_future_controls,
)
from .safety_supervisor import (
    CandidateControl,
    CandidateControlGenerator,
    FIXED_MAP_PREDICTION_HORIZON_S,
    MINIMUM_INTERVENTION_GATE_VERSION,
    PREDICTION_HORIZON_POLICY_VERSION,
    PredictiveSafetySupervisor,
    minimum_intervention_action,
)


LASER_COUNT = 72
OFFLINE_LASER_RANGE_M = 20.0
LIVE_RESET_SPAWN_X_M = 40.418016575586535
LIVE_RESET_SPAWN_Y_M = 63.725363742991874
LIVE_RESET_SPAWN_YAW_RAD = 1.778354580040184


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
    waypoint_min_distances_m: tuple[float, ...] = ()


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
    waypoint_min_distances_m: tuple[float, ...] = ()
    policy_mode: str = "deterministic-sac-minimum-intervention"

    @property
    def offline_ready(self) -> bool:
        return (
            self.episodes >= 20
            and self.completed_episodes == self.episodes
            and self.safety_stops == 0
            and self.timeouts == 0
            and len(self.waypoint_min_distances_m) == 13
            and all(
                distance <= 0.5
                for distance in self.waypoint_min_distances_m
            )
        )


@dataclass(frozen=True)
class TrainingManeuverTransition:
    mission_index: int
    maneuver_phase: str
    task_point_advanced: bool
    needs_new_plan: bool
    completed: bool


def advance_training_maneuver(
    *,
    mission_index: int,
    maneuver_phase: str,
    reached: bool,
    route_point_count: int,
) -> TrainingManeuverTransition:
    """Apply the same narrow-point state transition as the live controller."""

    if maneuver_phase not in ("NORMAL", "ESCAPE_PENDING"):
        raise ValueError("training maneuver phase is invalid")
    if not reached:
        return TrainingManeuverTransition(
            mission_index,
            maneuver_phase,
            False,
            False,
            False,
        )
    if maneuver_phase == "ESCAPE_PENDING":
        return TrainingManeuverTransition(
            mission_index,
            "NORMAL",
            False,
            True,
            False,
        )
    reached_index = mission_index
    next_index = mission_index + 1
    if reached_index == NARROW_ROUTE_INDEX:
        return TrainingManeuverTransition(
            next_index,
            "ESCAPE_PENDING",
            True,
            False,
            False,
        )
    completed = next_index >= route_point_count
    return TrainingManeuverTransition(
        next_index,
        "NORMAL",
        True,
        not completed,
        completed,
    )


class FixedMapSACTrainer:
    """Train and evaluate masked recurrent discrete SAC on the fixed route."""

    def __init__(
        self,
        *,
        compiled_map: Optional[CompiledSidecarMap] = None,
        dynamics: Optional[PrototypeReducedDynamics] = None,
        forward_profile: Optional[ForwardControlProfile] = None,
        reverse_profile: Optional[ReverseControlProfile] = None,
        calibration_status: str = "diagnostic_only",
        reverse_calibration_status: str = "diagnostic_only",
        seed: int = 31,
        hidden_dim: int = 32,
    ) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.compiled_map = compiled_map or compile_offline_national_map(
            session_id=f"fixed-map-training-{self.seed}",
        )
        self.forward_profile = (
            forward_profile or diagnostic_forward_control_profile()
        )
        self.reverse_profile = reverse_profile
        base_dynamics = dynamics or reduced_dynamics_from_profile(
            self.forward_profile
        )
        if reverse_profile is None:
            self.dynamics = base_dynamics
            self.planning_controls = self.forward_profile.action_controls
        else:
            if base_dynamics.allow_reverse:
                if (
                    base_dynamics.max_reverse_speed
                    != reverse_profile.max_reverse_speed_mps
                    or base_dynamics.reverse_throttle_speed_gain
                    != reverse_profile.reverse_throttle_speed_gain
                ):
                    raise ValueError(
                        "reverse profile and dynamics are incompatible"
                    )
                self.dynamics = base_dynamics
            else:
                self.dynamics = enable_reverse_dynamics(
                    base_dynamics,
                    reverse_profile,
                )
            self.planning_controls = (
                *self.forward_profile.action_controls,
                reverse_profile.control,
            )
        if calibration_status not in ("diagnostic_only", "calibrated"):
            raise ValueError("forward calibration status is invalid")
        if reverse_calibration_status not in (
            "diagnostic_only",
            "calibrated",
        ):
            raise ValueError("reverse calibration status is invalid")
        self.calibration_status = calibration_status
        self.reverse_calibration_status = reverse_calibration_status
        self.generator = CandidateControlGenerator(
            max_throttle=max(
                control.throttle
                for control in self.forward_profile.action_controls
            ),
            max_abs_rudder=max(
                abs(control.rudder)
                for control in self.forward_profile.action_controls
            ),
            action_controls=self.forward_profile.action_controls,
        )
        self.supervisor = PredictiveSafetySupervisor(
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
            max_state_age_s=1.0,
        )
        initial = self._initial_state()
        trajectory = plan_fixed_leg(
            self.compiled_map,
            start_state=initial,
            mission_index=0,
            dynamics=self.dynamics,
            seed=self.seed,
            forward_action_controls=self.planning_controls,
        )
        preview = self._preview(initial, trajectory, 0)
        candidates, safe_mask, _, _, _, _ = self._safe_candidates(
            initial,
            trajectory.controls[preview.nominal_control_index],
            trajectory,
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

    def _initial_state(
        self,
        episode: Optional[int] = None,
    ) -> VesselState:
        if episode is None or episode == 0:
            return VesselState(
                x=LIVE_RESET_SPAWN_X_M,
                y=LIVE_RESET_SPAWN_Y_M,
                yaw=LIVE_RESET_SPAWN_YAW_RAD,
                speed=0.0,
                yaw_rate=0.0,
                stamp_sim=self.compiled_map.snapshot.stamp_sim,
            )
        rng = random.Random(self.seed + 104_729 * int(episode))
        for _ in range(20):
            state = VesselState(
                x=LIVE_RESET_SPAWN_X_M + rng.uniform(-1.0, 1.0),
                y=LIVE_RESET_SPAWN_Y_M + rng.uniform(-1.0, 1.0),
                yaw=(
                    LIVE_RESET_SPAWN_YAW_RAD
                    + rng.uniform(-math.pi / 6.0, math.pi / 6.0)
                )
                % (2.0 * math.pi),
                speed=0.0,
                yaw_rate=0.0,
                stamp_sim=self.compiled_map.snapshot.stamp_sim,
            )
            if (
                self.compiled_map.snapshot.is_state_valid(state)
                and self.compiled_map.snapshot.clearance_at(state) >= 3.0
            ):
                return state
        raise RuntimeError("no safe perturbed Unity reset pose was found")

    def _goal(self, mission_index: int) -> GoalRegion:
        goal_x, goal_y = fixed_route_goal_xy(
            self.compiled_map.manifest,
            mission_index,
        )
        return GoalRegion(
            x=goal_x,
            y=goal_y,
            position_tolerance=fixed_route_tolerance(
                self.compiled_map,
                mission_index,
            ),
            speed_limit=1.2,
            yaw_rate_limit=1.2,
        )

    @staticmethod
    def _preview(
        state: VesselState,
        trajectory: Trajectory,
        previous_index: int,
        *,
        allow_reverse_branch_progress: bool = False,
    ) -> TrajectoryPreview:
        return preview_trajectory(
            state,
            trajectory,
            previous_index,
            allow_reverse_branch_progress=(
                allow_reverse_branch_progress
            ),
        )

    def _safe_candidates(
        self,
        state: VesselState,
        nominal_control,
        trajectory: Trajectory,
        preview: TrajectoryPreview,
        *,
        force_nominal: bool = False,
    ) -> tuple[
        tuple[CandidateControl, ...],
        tuple[bool, ...],
        tuple[str, ...],
        tuple[float, ...],
        float,
        bool,
    ]:
        braking_override = False
        reverse_feedback = None
        if nominal_control.throttle < 0.0:
            reverse_feedback = reverse_tracking_control(
                preview,
                nominal_control,
                self.dynamics,
                yaw_rate=state.yaw_rate,
            )
            candidates = tuple(
                CandidateControl(action=index, control=reverse_feedback)
                for index in range(5)
            )
        elif force_nominal:
            candidates = tuple(
                CandidateControl(action=index, control=nominal_control)
                for index in range(5)
            )
        else:
            feedback_control = feedback_tracking_control(
                preview,
                nominal_control,
                self.dynamics,
                yaw_rate=state.yaw_rate,
                speed=state.speed,
                clearance_m=self.compiled_map.snapshot.clearance_at(
                    state
                ),
            )
            braking_override = feedback_control.throttle < 0.0
            candidates = (
                tuple(
                    CandidateControl(
                        action=index,
                        control=feedback_control,
                    )
                    for index in range(5)
                )
                if braking_override
                else self.generator.generate(
                    feedback_control.throttle,
                    feedback_control.rudder,
                )
            )
        planned_future = self._nominal_future_controls(
            trajectory,
            preview,
        )
        if force_nominal:
            nominal_future_controls = narrow_ingress_future_controls(
                nominal_control,
                planned_future,
            )
        elif braking_override:
            nominal_future_controls = braking_future_controls(
                feedback_control
            )
        elif reverse_feedback is not None:
            nominal_future_controls = tracking_future_controls(
                reverse_feedback,
                planned_future,
            )
        else:
            nominal_future_controls = tracking_future_controls(
                feedback_control,
                planned_future,
            )
        if (
            nominal_control.throttle < 0.0
            or force_nominal
            or braking_override
        ):
            mask, reasons, clearances = self.supervisor.precheck(
                state,
                candidates,
                self.compiled_map.snapshot,
                self.dynamics,
                now_sim=state.stamp_sim,
                prediction_horizon_s=(
                    FIXED_MAP_PREDICTION_HORIZON_S
                ),
                candidate_prefix_s=0.3,
                nominal_future_controls=nominal_future_controls,
            )
            horizon = FIXED_MAP_PREDICTION_HORIZON_S
        else:
            (
                mask,
                reasons,
                clearances,
                horizon,
            ) = self.supervisor.precheck_with_horizon_fallback(
                state,
                candidates,
                self.compiled_map.snapshot,
                self.dynamics,
                now_sim=state.stamp_sim,
                candidate_prefix_s=0.3,
                nominal_future_controls=nominal_future_controls,
            )
        return (
            candidates,
            mask,
            reasons,
            clearances,
            horizon,
            braking_override,
        )

    @staticmethod
    def _nominal_future_controls(
        trajectory: Trajectory,
        preview: TrajectoryPreview,
    ) -> tuple[tuple[Control, float], ...]:
        remaining = FIXED_MAP_PREDICTION_HORIZON_S - 0.3
        skip = 0.3
        controls: list[tuple[Control, float]] = []
        for control, duration in zip(
            trajectory.controls[preview.nominal_control_index :],
            trajectory.durations[preview.nominal_control_index :],
        ):
            if remaining <= 1e-12:
                break
            available = float(duration)
            if skip > 1e-12:
                removed = min(skip, available)
                available -= removed
                skip -= removed
            if available <= 1e-12:
                continue
            applied = min(available, remaining)
            controls.append((control, applied))
            remaining -= applied
        if remaining > 1e-12:
            controls.append((trajectory.controls[-1], remaining))
        return tuple(controls)

    def _active_goal(
        self,
        mission_index: int,
        maneuver_phase: str,
    ) -> GoalRegion:
        if maneuver_phase == "ESCAPE_PENDING":
            return GoalRegion(
                x=NARROW_ESCAPE_XY[0],
                y=NARROW_ESCAPE_XY[1],
                position_tolerance=NARROW_ESCAPE_TOLERANCE_M,
                speed_limit=1.2,
                yaw_rate_limit=1.2,
            )
        return self._goal(mission_index)

    def _narrow_ingress_control(
        self,
        state: VesselState,
    ) -> Control:
        gate_x, gate_y, _ = fixed_route_gate_region(
            self.compiled_map,
            NARROW_ROUTE_INDEX,
        )
        desired_yaw = math.atan2(
            gate_y - state.y,
            gate_x - state.x,
        )
        heading_error = (
            desired_yaw - state.yaw + math.pi
        ) % (2.0 * math.pi) - math.pi
        return narrow_ingress_control(
            throttle=self.forward_profile.minimum_steerage_throttle,
            heading_error=heading_error,
            rudder_yaw_sign=self.dynamics.rudder_yaw_sign,
        )

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
        max_steps: int = 3_000,
    ) -> tuple[tuple[SequenceTransition, ...], EpisodeSummary]:
        if not 0.0 <= nominal_action_probability <= 1.0:
            raise ValueError("nominal_action_probability must be in [0, 1]")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        state = self._initial_state(episode)
        mission_index = 0
        trajectory = plan_fixed_leg(
            self.compiled_map,
            start_state=state,
            mission_index=mission_index,
            dynamics=self.dynamics,
            seed=self.seed + episode,
            forward_action_controls=self.planning_controls,
        )
        trajectory_index = 0
        maneuver_phase = "NORMAL"
        hidden: Optional[RecurrentHiddenState] = None
        hidden_reset = True
        transitions = []
        total_reward = 0.0
        minimum_clearance = float("inf")
        replans = 0
        completed = False
        safety_stop = False
        timed_out = False
        route_points = tuple(
            fixed_route_goal_xy(self.compiled_map.manifest, index)
            for index in range(
                len(self.compiled_map.manifest.route_points_enu)
            )
        )
        waypoint_min_distances = [
            math.hypot(state.x - x, state.y - y)
            for x, y in route_points
        ]

        for step in range(max_steps):
            goal = self._active_goal(mission_index, maneuver_phase)
            preview = self._preview(
                state,
                trajectory,
                trajectory_index,
                allow_reverse_branch_progress=(
                    maneuver_phase == "ESCAPE_PENDING"
                ),
            )
            if (
                preview.cross_track_error_m > 0.8
                and maneuver_phase != "ESCAPE_PENDING"
            ):
                planning_index = (
                    NARROW_ROUTE_INDEX
                    if maneuver_phase == "ESCAPE_PENDING"
                    else mission_index
                )
                trajectory = plan_fixed_leg(
                    self.compiled_map,
                    start_state=state,
                    mission_index=planning_index,
                    dynamics=self.dynamics,
                    seed=self.seed + episode + step,
                    narrow_visit_completed=(
                        maneuver_phase == "ESCAPE_PENDING"
                    ),
                    forward_action_controls=self.planning_controls,
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
            ingress_recovery = (
                nominal.throttle < 0.0
                and maneuver_phase != "ESCAPE_PENDING"
                and mission_index == NARROW_ROUTE_INDEX
            )
            if ingress_recovery:
                nominal = self._narrow_ingress_control(state)
            (
                candidates,
                safe_mask,
                reasons,
                clearances,
                safety_horizon,
                braking_override,
            ) = (
                self._safe_candidates(
                    state,
                    nominal,
                    trajectory,
                    preview,
                    force_nominal=ingress_recovery,
                )
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

            if (
                nominal.throttle < 0.0
                or ingress_recovery
                or braking_override
            ):
                policy_action = 2
            else:
                proposal, hidden = self.sac.act(
                    observation,
                    safe_mask,
                    hidden=hidden,
                    deterministic=deterministic_policy,
                )
                policy_action = proposal.action
                if deterministic_policy:
                    policy_action = minimum_intervention_action(
                        policy_action=policy_action,
                        safe_action_mask=safe_mask,
                        candidates=candidates,
                        nominal_control=candidates[2].control,
                    )
            hidden_reset = False
            if (
                not ingress_recovery
                and nominal.throttle >= 0.0
                and
                safe_mask[2]
                and self.rng.random() < nominal_action_probability
            ):
                policy_action = 2
            finalize_future_controls = self._nominal_future_controls(
                trajectory,
                preview,
            )
            if ingress_recovery:
                finalize_future_controls = narrow_ingress_future_controls(
                    nominal,
                    finalize_future_controls,
                )
            elif braking_override:
                finalize_future_controls = braking_future_controls(
                    candidates[2].control,
                )
            elif nominal.throttle < 0.0:
                finalize_future_controls = tracking_future_controls(
                    candidates[2].control,
                    finalize_future_controls,
                )
            elif nominal.throttle >= 0.0:
                finalize_future_controls = tracking_future_controls(
                    candidates[2].control,
                    finalize_future_controls,
                )
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
                prediction_horizon_s=safety_horizon,
                candidate_prefix_s=0.3,
                nominal_future_controls=finalize_future_controls,
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
            for index, (x, y) in enumerate(route_points):
                waypoint_min_distances[index] = min(
                    waypoint_min_distances[index],
                    math.hypot(next_state.x - x, next_state.y - y),
                )
            minimum_clearance = min(
                minimum_clearance,
                self.compiled_map.snapshot.clearance_at(next_state),
            )
            if maneuver_phase == "ESCAPE_PENDING":
                advanced = narrow_escape_released(
                    self.compiled_map,
                    next_state,
                )
            elif mission_index == NARROW_ROUTE_INDEX:
                advanced = goal.contains(next_state)
            else:
                gate_x, gate_y, gate_tolerance = (
                    fixed_route_gate_region(
                        self.compiled_map,
                        mission_index,
                    )
                )
                advanced = (
                    goal.contains(next_state)
                    and math.hypot(
                        next_state.x - gate_x,
                        next_state.y - gate_y,
                    )
                    <= gate_tolerance + 1e-9
                )
            terminated = False
            next_hidden_reset = False
            transition = advance_training_maneuver(
                mission_index=mission_index,
                maneuver_phase=maneuver_phase,
                reached=advanced,
                route_point_count=len(
                    self.compiled_map.manifest.route_points_enu
                ),
            )
            mission_index = transition.mission_index
            maneuver_phase = transition.maneuver_phase
            terminated = transition.completed
            completed = transition.completed
            if transition.needs_new_plan:
                if not terminated:
                    trajectory = plan_fixed_leg(
                        self.compiled_map,
                        start_state=next_state,
                        mission_index=mission_index,
                        dynamics=self.dynamics,
                        seed=self.seed + episode + step + mission_index,
                        forward_action_controls=self.planning_controls,
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
                    allow_reverse_branch_progress=(
                        maneuver_phase == "ESCAPE_PENDING"
                    ),
                )
                next_nominal = trajectory.controls[
                    next_preview.nominal_control_index
                ]
                next_ingress_recovery = (
                    next_nominal.throttle < 0.0
                    and maneuver_phase != "ESCAPE_PENDING"
                    and mission_index == NARROW_ROUTE_INDEX
                )
                if next_ingress_recovery:
                    next_nominal = self._narrow_ingress_control(
                        next_state
                    )
                _, next_safe_mask, _, _, _, _ = self._safe_candidates(
                    next_state,
                    next_nominal,
                    trajectory,
                    next_preview,
                    force_nominal=next_ingress_recovery,
                )
            next_observation = self._observation(
                next_state,
                trajectory,
                next_preview,
                next_safe_mask,
                hidden_reset=next_hidden_reset,
            )
            next_goal = (
                None
                if terminated
                else self._active_goal(
                    mission_index,
                    maneuver_phase,
                )
            )
            new_goal_distance = (
                0.0
                if terminated
                else math.hypot(
                    next_state.x - next_goal.x,
                    next_state.y - next_goal.y,
                )
            )
            progress_reward = (
                old_goal_distance - new_goal_distance
                if not transition.task_point_advanced
                and not transition.needs_new_plan
                else 0.5
            )
            reward = (
                2.0 * progress_reward
                - 0.1 * next_preview.cross_track_error_m
                - 0.03 * abs(decision.final_action - 2)
                + 0.02 * min(minimum_clearance, 2.0)
                + (5.0 if transition.task_point_advanced else 0.0)
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
            waypoint_min_distances_m=tuple(
                waypoint_min_distances
            ),
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
        max_steps: int = 3_000,
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
            waypoint_min_distances_m=tuple(
                max(
                    summary.waypoint_min_distances_m[index]
                    for summary in episode_summaries
                )
                for index in range(13)
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
        offline_ready = (
            evaluation_summary is not None
            and evaluation_summary.offline_ready
            and self.calibration_status == "calibrated"
            and self.reverse_profile is not None
            and self.reverse_calibration_status == "calibrated"
        )
        manifest = {
            "schema_version": "national-test-sac-checkpoint-v4",
            "algorithm": "discrete-recurrent-sac",
            "dynamics_version": self.dynamics.version,
            "route_guidance_version": ROUTE_GUIDANCE_VERSION,
            "route_guidance_hash": fixed_route_guidance_hash(
                self.compiled_map
            ),
            "map_profile": "北湖/National_Test",
            "map_snapshot_id": self.compiled_map.snapshot.snapshot_id,
            "map_payload_hash": (
                self.compiled_map.snapshot.payload_content_hash
            ),
            "map_source_artifact_hash": (
                self.compiled_map.snapshot.source_artifact_hash
            ),
            "geometry_version": (
                self.compiled_map.snapshot.geometry_version
            ),
            "route_id": self.compiled_map.manifest.route_id,
            "route_version": self.compiled_map.manifest.route_version,
            "observation_schema": self.sac.observation_schema,
            "observation_dim": self.sac.observation_dim,
            "hidden_dim": self.sac.hidden_dim,
            "action_schema": self.sac.action_schema,
            "action_dim": self.sac.action_dim,
            "action_controls": [
                {
                    "throttle": control.throttle,
                    "rudder": control.rudder,
                }
                for control in self.forward_profile.action_controls
            ],
            "action_protocol_hash": action_protocol_hash(
                self.forward_profile
            ),
            "policy_gate_version": (
                MINIMUM_INTERVENTION_GATE_VERSION
            ),
            "prediction_horizon_policy_version": (
                PREDICTION_HORIZON_POLICY_VERSION
            ),
            "forward_control_profile": asdict(self.forward_profile),
            "reverse_control_profile": (
                None
                if self.reverse_profile is None
                else reverse_control_profile_to_dict(
                    self.reverse_profile
                )
            ),
            "calibration_hash": (
                self.forward_profile.calibration_hash
            ),
            "calibration_status": self.calibration_status,
            "reverse_calibration_status": (
                self.reverse_calibration_status
            ),
            "checkpoint_sha256": digest,
            "training_summary": asdict(training_summary),
            "evaluation_summary": (
                None
                if evaluation_summary is None
                else asdict(evaluation_summary)
            ),
            "waypoint_min_distances_m": (
                None
                if evaluation_summary is None
                else list(
                    evaluation_summary.waypoint_min_distances_m
                )
            ),
            "offline_ready": offline_ready,
            "live_ready": False,
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
    "TrainingManeuverTransition",
    "TrainingSummary",
    "advance_training_maneuver",
]
