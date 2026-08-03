"""Predictive five-action safety supervisor for the isolated first version."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence

from usvlib4ros.planning import Control, PlanningMapSnapshot, PrototypeReducedDynamics, VesselState


ACTION_RUDDER_OFFSETS = (1.0, 0.5, 0.0, -0.5, -1.0)
FIXED_MAP_PREDICTION_HORIZON_S = 2.0
FIXED_MAP_FALLBACK_PREDICTION_HORIZON_S = 1.1
MINIMUM_INTERVENTION_GATE_VERSION = "sac-minimum-intervention-v2"
PREDICTION_HORIZON_POLICY_VERSION = "primary-2.0-fallback-1.1-v7"


@dataclass(frozen=True)
class CandidateControl:
    action: int
    control: Control


class CandidateControlGenerator:
    """Complete every discrete action with throttle and rudder before masking."""

    def __init__(
        self,
        rudder_step: float = 0.05,
        max_throttle: float = 0.1,
        max_abs_rudder: float = 0.1,
        action_controls: Sequence[Control] = (),
    ) -> None:
        try:
            controls = tuple(action_controls)
        except TypeError as exc:
            raise ValueError("profile action controls are invalid") from exc
        if (
            not all(
                isfinite(value)
                for value in (
                    rudder_step,
                    max_throttle,
                    max_abs_rudder,
                )
            )
            or rudder_step <= 0.0
            or not 0.0 < max_throttle <= 1.0
            or not 0.0 < max_abs_rudder <= 1.0
            or (
                controls
                and (
                    len(controls) != 5
                    or any(
                        not isinstance(control, Control)
                        or not control.is_valid()
                        or control.throttle < 0.0
                        for control in controls
                    )
                )
            )
        ):
            raise ValueError("live action bounds are invalid")
        self.rudder_step = float(rudder_step)
        self.max_throttle = float(max_throttle)
        self.max_abs_rudder = float(max_abs_rudder)
        self.action_controls = controls

    def generate(self, nominal_throttle: float, nominal_rudder: float) -> tuple[CandidateControl, ...]:
        if not isfinite(nominal_throttle) or not isfinite(nominal_rudder):
            raise ValueError("nominal control must be finite")
        rudder = max(
            -self.max_abs_rudder,
            min(self.max_abs_rudder, float(nominal_rudder)),
        )
        throttle = max(
            0.0,
            min(self.max_throttle, float(nominal_throttle)),
        )
        if self.action_controls:
            return tuple(
                CandidateControl(
                    action=action,
                    control=Control(
                        throttle=max(
                            0.0,
                            min(
                                self.max_throttle,
                                (
                                    throttle
                                    if action == 2
                                    else min(
                                        throttle,
                                        template.throttle,
                                    )
                                ),
                            ),
                        ),
                        rudder=max(
                            -self.max_abs_rudder,
                            min(
                                self.max_abs_rudder,
                                (
                                    template.rudder
                                    if action in (0, 4)
                                    else rudder + template.rudder
                                ),
                            ),
                        ),
                    ),
                )
                for action, template in enumerate(self.action_controls)
            )
        return tuple(
            CandidateControl(
                action=action,
                control=Control(
                    throttle=throttle,
                    rudder=max(
                        -self.max_abs_rudder,
                        min(
                            self.max_abs_rudder,
                            rudder + offset * self.rudder_step,
                        ),
                    ),
                ),
            )
            for action, offset in enumerate(ACTION_RUDDER_OFFSETS)
        )


def minimum_intervention_action(
    *,
    policy_action: Optional[int],
    safe_action_mask: Sequence[bool],
    candidates: Sequence[CandidateControl],
    nominal_control: Control,
) -> Optional[int]:
    """Keep SAC inside the least-change set allowed by the safety mask.

    SAC breaks ties between equally small safe interventions.  A larger
    throttle or rudder deviation cannot displace a smaller safe correction.
    """

    try:
        mask = tuple(safe_action_mask)
        candidate_values = tuple(candidates)
    except TypeError as exc:
        raise ValueError("minimum-intervention inputs are invalid") from exc
    if (
        len(mask) != 5
        or any(type(value) is not bool for value in mask)
        or len(candidate_values) != 5
        or any(
            not isinstance(candidate, CandidateControl)
            or candidate.action != index
            or not candidate.control.is_valid()
            for index, candidate in enumerate(candidate_values)
        )
        or not isinstance(nominal_control, Control)
        or not nominal_control.is_valid()
        or (
            policy_action is not None
            and (
                not isinstance(policy_action, int)
                or isinstance(policy_action, bool)
                or not 0 <= policy_action < 5
            )
        )
    ):
        raise ValueError("minimum-intervention inputs are invalid")
    safe_actions = tuple(
        index for index, safe in enumerate(mask) if safe
    )
    if not safe_actions:
        return None
    deviations = {
        index: (
            abs(
                candidate_values[index].control.throttle
                - nominal_control.throttle
            )
            + abs(
                candidate_values[index].control.rudder
                - nominal_control.rudder
            )
        )
        for index in safe_actions
    }
    minimum = min(deviations.values())
    least_change = tuple(
        index
        for index in safe_actions
        if deviations[index] <= minimum + 1e-12
    )
    if policy_action in least_change:
        return policy_action
    return min(least_change, key=lambda index: (abs(index - 2), index))


@dataclass(frozen=True)
class SafetyDecision:
    policy_action: Optional[int]
    final_action: Optional[int]
    control: Control
    candidate_mask: tuple[bool, ...]
    reasons: tuple[str, ...]
    reason: str
    stop: bool
    overridden: bool
    minimum_clearance: float


class PredictiveSafetySupervisor:
    """Predictive pre-mask plus final version check; it never publishes ROS output."""

    def __init__(self, prediction_horizon_s: float = 5.0, max_state_age_s: float = 2.0) -> None:
        if not isfinite(prediction_horizon_s) or prediction_horizon_s <= 0.0:
            raise ValueError("prediction_horizon_s must be positive and finite")
        if not isfinite(max_state_age_s) or max_state_age_s < 0.0:
            raise ValueError("max_state_age_s must be finite and non-negative")
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.max_state_age_s = float(max_state_age_s)

    def precheck(
        self,
        state: VesselState,
        candidates: Sequence[CandidateControl],
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        *,
        now_sim: float,
        prediction_horizon_s: Optional[float] = None,
        candidate_prefix_s: float = 0.3,
        nominal_future_controls: Optional[
            Sequence[tuple[Control, float]]
        ] = None,
    ) -> tuple[tuple[bool, ...], tuple[str, ...], tuple[float, ...]]:
        try:
            horizon_s = (
                self.prediction_horizon_s
                if prediction_horizon_s is None
                else float(prediction_horizon_s)
            )
        except (TypeError, ValueError, OverflowError):
            return (False,) * 5, ("INVALID_INPUT",) * 5, (0.0,) * 5
        try:
            candidate_values = tuple(candidates)
        except TypeError:
            return (False,) * 5, ("INVALID_INPUT",) * 5, (0.0,) * 5
        try:
            prefix_s = float(candidate_prefix_s)
            future_values = (
                None
                if nominal_future_controls is None
                else tuple(nominal_future_controls)
            )
        except (TypeError, ValueError, OverflowError):
            return (False,) * 5, ("INVALID_INPUT",) * 5, (0.0,) * 5
        if len(candidate_values) != 5 or any(
            not isinstance(candidate, CandidateControl)
            or candidate.action != index
            for index, candidate in enumerate(candidate_values)
        ):
            return (False,) * 5, ("INVALID_INPUT",) * 5, (0.0,) * 5
        if (
            not isinstance(state, VesselState)
            or not state.is_finite()
            or not isinstance(map_snapshot, PlanningMapSnapshot)
            or map_snapshot.coverage_status != "complete_prior"
            or not isinstance(dynamics, PrototypeReducedDynamics)
            or not isfinite(now_sim)
            or now_sim < state.stamp_sim
            or now_sim - state.stamp_sim > self.max_state_age_s
            or not isfinite(horizon_s)
            or horizon_s <= 0.0
            or not isfinite(prefix_s)
            or prefix_s <= 0.0
            or (
                future_values is not None
                and any(
                    not isinstance(item, tuple)
                    or len(item) != 2
                    or not isinstance(item[0], Control)
                    or not item[0].is_valid()
                    or not isfinite(item[1])
                    or item[1] <= 0.0
                    for item in future_values
                )
            )
        ):
            return (False,) * 5, ("INVALID_INPUT",) * 5, (0.0,) * 5
        mask: list[bool] = []
        reasons: list[str] = []
        clearances: list[float] = []
        for candidate in candidate_values:
            if not candidate.control.is_valid():
                mask.append(False)
                reasons.append("INVALID_CONTROL")
                clearances.append(0.0)
                continue
            try:
                if future_values is None:
                    rollout = dynamics.propagate(
                        state,
                        candidate.control,
                        float(horizon_s),
                    )
                else:
                    prefix_duration = min(prefix_s, horizon_s)
                    first = dynamics.propagate(
                        state,
                        candidate.control,
                        prefix_duration,
                    )
                    rollout_values = list(first)
                    current = first[-1]
                    remaining = horizon_s - prefix_duration
                    for future_control, duration in future_values:
                        if remaining <= 1e-12:
                            break
                        applied = min(float(duration), remaining)
                        segment = dynamics.propagate(
                            current,
                            future_control,
                            applied,
                        )
                        rollout_values.extend(segment[1:])
                        current = segment[-1]
                        remaining -= applied
                    if remaining > 1e-12:
                        fill_control = (
                            future_values[-1][0]
                            if future_values
                            else candidate.control
                        )
                        segment = dynamics.propagate(
                            current,
                            fill_control,
                            remaining,
                        )
                        rollout_values.extend(segment[1:])
                    rollout = tuple(rollout_values)
                motion = map_snapshot.check_motion(rollout)
            except (ArithmeticError, TypeError, ValueError):
                mask.append(False)
                reasons.append("PREDICTION_ERROR")
                clearances.append(0.0)
                continue
            mask.append(motion.valid)
            reasons.append("SAFE" if motion.valid else motion.reason)
            clearances.append(motion.min_clearance if motion.valid else 0.0)
        return tuple(mask), tuple(reasons), tuple(clearances)

    def clearance_recovery_is_safe(
        self,
        state: VesselState,
        control: Control,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        *,
        now_sim: float,
        minimum_clearance_m: float,
    ) -> bool:
        """Approve only a bounded motion that returns to the strict map margin."""

        if (
            not isinstance(state, VesselState)
            or not state.is_finite()
            or not isinstance(control, Control)
            or not control.is_valid()
            or not isinstance(map_snapshot, PlanningMapSnapshot)
            or not isinstance(dynamics, PrototypeReducedDynamics)
            or not isfinite(now_sim)
            or now_sim < state.stamp_sim
            or now_sim - state.stamp_sim > self.max_state_age_s
            or not isfinite(minimum_clearance_m)
            or minimum_clearance_m < 0.0
        ):
            return False
        initial = map_snapshot.clearance_at(state)
        if (
            initial <= minimum_clearance_m + 1e-9
            or initial > map_snapshot.required_clearance + 1e-9
        ):
            return False
        try:
            rollout = dynamics.propagate(
                state,
                control,
                self.prediction_horizon_s,
            )
        except (ArithmeticError, TypeError, ValueError):
            return False
        clearances = tuple(
            map_snapshot.clearance_at(item) for item in rollout
        )
        return (
            all(isfinite(value) for value in clearances)
            and min(clearances) >= initial - 0.01
            and min(clearances) > minimum_clearance_m + 1e-9
            and clearances[-1]
            > map_snapshot.required_clearance + 1e-9
            and clearances[-1] >= initial + 0.03
        )

    def precheck_with_horizon_fallback(
        self,
        state: VesselState,
        candidates: Sequence[CandidateControl],
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        *,
        now_sim: float,
        primary_horizon_s: float = FIXED_MAP_PREDICTION_HORIZON_S,
        fallback_horizon_s: float = (
            FIXED_MAP_FALLBACK_PREDICTION_HORIZON_S
        ),
        candidate_prefix_s: float = 0.3,
        nominal_future_controls: Optional[
            Sequence[tuple[Control, float]]
        ] = None,
    ) -> tuple[
        tuple[bool, ...],
        tuple[str, ...],
        tuple[float, ...],
        float,
    ]:
        """Retry a shorter receding horizon only after total deadlock."""

        try:
            primary = float(primary_horizon_s)
            fallback = float(fallback_horizon_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("prediction horizons are invalid") from exc
        if (
            not isfinite(primary)
            or not isfinite(fallback)
            or fallback <= 0.0
            or fallback >= primary
        ):
            raise ValueError("prediction horizons are invalid")
        result = self.precheck(
            state,
            candidates,
            map_snapshot,
            dynamics,
            now_sim=now_sim,
            prediction_horizon_s=primary,
            candidate_prefix_s=candidate_prefix_s,
            nominal_future_controls=nominal_future_controls,
        )
        if any(result[0]):
            return *result, primary
        fallback_result = self.precheck(
            state,
            candidates,
            map_snapshot,
            dynamics,
            now_sim=now_sim,
            prediction_horizon_s=fallback,
            candidate_prefix_s=candidate_prefix_s,
            nominal_future_controls=nominal_future_controls,
        )
        return *fallback_result, fallback

    def finalize(
        self,
        *,
        policy_action: Optional[int],
        nominal_action: Optional[int] = None,
        candidate_mask: Sequence[bool],
        candidates: Sequence[CandidateControl],
        snapshot_id: str,
        current_snapshot_id: str,
        reasons: Optional[Sequence[str]] = None,
        clearances: Optional[Sequence[float]] = None,
        current_state: Optional[VesselState] = None,
        current_map_snapshot: Optional[PlanningMapSnapshot] = None,
        dynamics: Optional[PrototypeReducedDynamics] = None,
        now_sim: Optional[float] = None,
        prediction_horizon_s: Optional[float] = None,
        candidate_prefix_s: float = 0.3,
        nominal_future_controls: Optional[
            Sequence[tuple[Control, float]]
        ] = None,
    ) -> SafetyDecision:
        try:
            mask_values = tuple(candidate_mask)
            candidate_values = tuple(candidates)
        except TypeError:
            return self._stop(policy_action, (False,) * 5, "INVALID_INPUT")
        if any(not isinstance(value, bool) for value in mask_values):
            return self._stop(policy_action, (False,) * 5, "INVALID_INPUT")
        mask = mask_values
        if len(mask) != 5 or len(candidate_values) != 5:
            return self._stop(policy_action, (False,) * 5, "INVALID_INPUT")
        if any(
            not isinstance(candidate, CandidateControl)
            or candidate.action != index
            or not candidate.control.is_valid()
            for index, candidate in enumerate(candidate_values)
        ):
            return self._stop(policy_action, mask, "INVALID_INPUT")
        if snapshot_id != current_snapshot_id:
            return self._stop(policy_action, mask, "SNAPSHOT_VERSION_CHANGED")
        if not any(mask):
            return self._stop(policy_action, mask, "NO_SAFE_ACTION")
        if policy_action is not None and (
            not isinstance(policy_action, int)
            or isinstance(policy_action, bool)
            or not 0 <= policy_action < 5
        ):
            return self._stop(policy_action, mask, "INVALID_POLICY_ACTION")
        if nominal_action is not None and (
            not isinstance(nominal_action, int)
            or isinstance(nominal_action, bool)
            or not 0 <= nominal_action < 5
        ):
            return self._stop(policy_action, mask, "INVALID_NOMINAL_ACTION")
        if reasons is None:
            reason_values = ("SAFE",) * 5
        else:
            try:
                reason_values = tuple(reasons)
            except TypeError:
                return self._stop(policy_action, mask, "INVALID_INPUT")
            if (
                isinstance(reasons, (str, bytes))
                or len(reason_values) != 5
                or any(not isinstance(reason, str) for reason in reason_values)
            ):
                return self._stop(policy_action, mask, "INVALID_INPUT")
        if clearances is None:
            clearance_values = (0.0,) * 5
        else:
            try:
                clearance_values = tuple(float(value) for value in clearances)
            except (TypeError, ValueError, OverflowError):
                return self._stop(policy_action, mask, "INVALID_INPUT")
            if len(clearance_values) != 5 or any(
                not isfinite(value) or value < 0.0 for value in clearance_values
            ):
                return self._stop(policy_action, mask, "INVALID_INPUT")
        latest_context = (current_state, current_map_snapshot, dynamics, now_sim)
        if not all(value is not None for value in latest_context):
            return self._stop(policy_action, mask, "LATEST_CONTEXT_REQUIRED")
        assert current_state is not None
        assert current_map_snapshot is not None
        assert dynamics is not None
        assert now_sim is not None
        if (
            not isinstance(current_map_snapshot, PlanningMapSnapshot)
            or current_map_snapshot.snapshot_id != current_snapshot_id
        ):
            return self._stop(policy_action, mask, "CURRENT_MAP_SNAPSHOT_MISMATCH")
        fresh_mask, fresh_reasons, fresh_clearances = self.precheck(
            current_state,
            candidate_values,
            current_map_snapshot,
            dynamics,
            now_sim=now_sim,
            prediction_horizon_s=prediction_horizon_s,
            candidate_prefix_s=candidate_prefix_s,
            nominal_future_controls=nominal_future_controls,
        )
        mask = tuple(first and second for first, second in zip(mask, fresh_mask))
        reason_values = tuple(
            fresh_reason if not fresh_safe else reason_values[index]
            for index, (fresh_safe, fresh_reason) in enumerate(zip(fresh_mask, fresh_reasons))
        )
        clearance_values = tuple(
            min(previous, fresh)
            if mask[index]
            else 0.0
            for index, (previous, fresh) in enumerate(zip(clearance_values, fresh_clearances))
        )
        if not any(mask):
            return self._stop(policy_action, mask, "LATEST_INPUT_UNSAFE")
        if policy_action is not None and mask[policy_action]:
            final_action = policy_action
            overridden = False
            reason = "POLICY_ACTION_SAFE"
        elif nominal_action is not None and mask[nominal_action]:
            final_action = nominal_action
            overridden = policy_action is not None
            reason = (
                "POLICY_ACTION_UNSAFE"
                if policy_action is not None
                else "NOMINAL_PROGRESS_FALLBACK"
            )
        else:
            final_action = next(index for index, safe in enumerate(mask) if safe)
            overridden = policy_action is not None
            reason = "POLICY_ACTION_UNSAFE" if policy_action is not None else "SAFE_ACTION_FALLBACK"
        control = candidate_values[final_action].control
        return SafetyDecision(
            policy_action=policy_action,
            final_action=final_action,
            control=control,
            candidate_mask=mask,
            reasons=reason_values,
            reason=reason,
            stop=False,
            overridden=overridden,
            minimum_clearance=clearance_values[final_action],
        )

    @staticmethod
    def _stop(policy_action: Optional[int], mask: tuple[bool, ...], reason: str) -> SafetyDecision:
        return SafetyDecision(
            policy_action=policy_action,
            final_action=None,
            control=Control(0.0, 0.0),
            candidate_mask=mask,
            reasons=(reason,) * 5,
            reason=reason,
            stop=True,
            overridden=policy_action is not None,
            minimum_clearance=0.0,
        )


__all__ = [
    "CandidateControl",
    "CandidateControlGenerator",
    "FIXED_MAP_FALLBACK_PREDICTION_HORIZON_S",
    "FIXED_MAP_PREDICTION_HORIZON_S",
    "MINIMUM_INTERVENTION_GATE_VERSION",
    "PREDICTION_HORIZON_POLICY_VERSION",
    "PredictiveSafetySupervisor",
    "SafetyDecision",
    "minimum_intervention_action",
]
