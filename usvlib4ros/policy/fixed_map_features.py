"""Shared fixed-map trajectory and laser features for training and runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from usvlib4ros.planning import (
    Control,
    PrototypeReducedDynamics,
    Trajectory,
    VesselState,
)

from .recurrent_sac import LASER_COUNT, LocalObservationV2


TRACKING_CRUISE_THROTTLE_CAP = 0.22
TRACKING_TARGET_SPEED_MPS = 0.3
TRACKING_REVERSE_BRAKE_SPEED_MPS = 0.4
TRACKING_REVERSE_BRAKE_THROTTLE = -0.4
TRACKING_PROXIMITY_CLEARANCE_M = 0.6
TRACKING_PROXIMITY_BRAKE_SPEED_MPS = 0.2
TRACKING_HEADING_KP = 0.8
TRACKING_YAW_RATE_KD = 0.35
TRACKING_RUDDER_LIMIT = 0.5


@dataclass(frozen=True)
class TrajectoryPreview:
    state_index: int
    nominal_control_index: int
    cross_track_error_m: float
    remaining_arc_length_m: float
    progress: float
    lookahead_x: float
    lookahead_y: float
    heading_error: float


def _angle_difference(first: float, second: float) -> float:
    return (first - second + math.pi) % (2.0 * math.pi) - math.pi


def preview_trajectory(
    state: VesselState,
    trajectory: Trajectory,
    previous_index: int,
    *,
    allow_reverse_branch_progress: bool = False,
) -> TrajectoryPreview:
    if not trajectory.controls or len(trajectory.states) < 2:
        raise ValueError("trajectory preview requires at least one control")
    if not 0 <= previous_index < len(trajectory.states):
        raise ValueError("previous trajectory index is out of range")
    contains_reverse = any(
        control.throttle < 0.0 for control in trajectory.controls
    )
    candidate_stop = min(
        len(trajectory.states),
        previous_index
        + (
            12
            if not contains_reverse or allow_reverse_branch_progress
            else 3
        ),
    )
    index = min(
        range(previous_index, candidate_stop),
        key=lambda candidate: (
            math.hypot(
                state.x - trajectory.states[candidate].x,
                state.y - trajectory.states[candidate].y,
            )
            + 0.35
            * abs(
                _angle_difference(
                    trajectory.states[candidate].yaw,
                    state.yaw,
                )
            )
            + 0.2
            * abs(trajectory.states[candidate].speed - state.speed)
        ),
    )
    lookahead = index
    lookahead_distance = 0.0
    while (
        lookahead + 1 < len(trajectory.states)
        and lookahead_distance < 1.0
    ):
        first = trajectory.states[lookahead]
        second = trajectory.states[lookahead + 1]
        lookahead_distance += math.hypot(
            second.x - first.x,
            second.y - first.y,
        )
        lookahead += 1
    lookahead_state = trajectory.states[lookahead]
    desired_yaw = math.atan2(
        lookahead_state.y - state.y,
        lookahead_state.x - state.x,
    )
    remaining = sum(
        math.hypot(second.x - first.x, second.y - first.y)
        for first, second in zip(
            trajectory.states[index:],
            trajectory.states[index + 1 :],
        )
    )
    return TrajectoryPreview(
        state_index=index,
        nominal_control_index=min(
            index,
            len(trajectory.controls) - 1,
        ),
        cross_track_error_m=math.hypot(
            state.x - trajectory.states[index].x,
            state.y - trajectory.states[index].y,
        ),
        remaining_arc_length_m=remaining,
        progress=index / max(1, len(trajectory.states) - 1),
        lookahead_x=lookahead_state.x,
        lookahead_y=lookahead_state.y,
        heading_error=_angle_difference(desired_yaw, state.yaw),
    )


def feedback_tracking_control(
    preview: TrajectoryPreview,
    nominal_control: Control,
    dynamics: PrototypeReducedDynamics,
    *,
    yaw_rate: float = 0.0,
    speed: float = 0.0,
    clearance_m: float = float("inf"),
) -> Control:
    """Track the live line of sight without retaining stale open-loop rudder."""

    if (
        speed > TRACKING_REVERSE_BRAKE_SPEED_MPS
        or (
            clearance_m < TRACKING_PROXIMITY_CLEARANCE_M
            and speed > TRACKING_PROXIMITY_BRAKE_SPEED_MPS
        )
    ):
        return Control(
            throttle=TRACKING_REVERSE_BRAKE_THROTTLE,
            rudder=0.0,
        )
    rudder = dynamics.rudder_yaw_sign * (
        TRACKING_HEADING_KP * preview.heading_error
        - TRACKING_YAW_RATE_KD * yaw_rate
    )
    rudder = max(
        -TRACKING_RUDDER_LIMIT,
        min(TRACKING_RUDDER_LIMIT, rudder),
    )
    return Control(
        throttle=min(
            nominal_control.throttle,
            TRACKING_CRUISE_THROTTLE_CAP,
        )
        if speed < TRACKING_TARGET_SPEED_MPS
        else 0.0,
        rudder=rudder,
    )


def reverse_tracking_control(
    preview: TrajectoryPreview,
    nominal_control: Control,
    dynamics: PrototypeReducedDynamics,
    *,
    yaw_rate: float = 0.0,
) -> Control:
    """Track the planned escape while the stern is the leading end."""

    if nominal_control.throttle >= 0.0:
        raise ValueError("reverse tracking requires negative throttle")
    reverse_heading_error = _angle_difference(
        preview.heading_error + math.pi,
        0.0,
    )
    rudder = dynamics.rudder_yaw_sign * (
        TRACKING_HEADING_KP * reverse_heading_error
        - TRACKING_YAW_RATE_KD * yaw_rate
    )
    return Control(
        throttle=nominal_control.throttle,
        rudder=max(
            -TRACKING_RUDDER_LIMIT,
            min(TRACKING_RUDDER_LIMIT, rudder),
        ),
    )


def narrow_ingress_control(
    *,
    throttle: float,
    heading_error: float = 0.0,
    rudder_yaw_sign: float = -1.0,
) -> Control:
    """Continue into the published narrow target before arming reverse."""

    return Control(
        throttle=throttle,
        rudder=max(
            -0.5,
            min(0.5, heading_error * rudder_yaw_sign),
        ),
    )


def narrow_ingress_future_controls(
    ingress_control: Control,
    nominal_future_controls: Sequence[tuple[Control, float]],
    *,
    candidate_prefix_s: float = 0.3,
    ingress_total_s: float = 0.8,
) -> tuple[tuple[Control, float], ...]:
    """Predict forward crossing followed by the reverse escape phase."""

    return (
        (ingress_control, ingress_total_s - candidate_prefix_s),
        *tuple(nominal_future_controls),
    )


def tracking_future_controls(
    tracking_control: Control,
    nominal_future_controls: Sequence[tuple[Control, float]],
    *,
    closed_loop_hold_s: float = 0.5,
) -> tuple[tuple[Control, float], ...]:
    """Keep the live feedback correction active before the open-loop preview."""

    return (
        (tracking_control, closed_loop_hold_s),
        *tuple(nominal_future_controls),
    )


def braking_future_controls(
    braking_control: Control,
    *,
    brake_after_prefix_s: float = 0.7,
    coast_after_brake_s: float = 1.0,
) -> tuple[tuple[Control, float], ...]:
    """Predict one-second active braking before a zero-thrust coast."""

    return (
        (braking_control, brake_after_prefix_s),
        (Control(0.0, 0.0), coast_after_brake_s),
    )


def front_arc_laser_features(
    ranges: Sequence[object] | Iterable[object],
    *,
    max_range_m: float = 20.0,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Return the sample-compatible first/last 36 front-arc beams."""

    if not math.isfinite(max_range_m) or max_range_m <= 0.0:
        raise ValueError("maximum laser range must be positive and finite")
    values = tuple(ranges)
    if not values:
        return (max_range_m,) * LASER_COUNT, (False,) * LASER_COUNT
    if len(values) >= LASER_COUNT:
        indices = (
            *range(LASER_COUNT // 2),
            *range(len(values) - LASER_COUNT // 2, len(values)),
        )
    else:
        indices = tuple(
            round(index * (len(values) - 1) / (LASER_COUNT - 1))
            for index in range(LASER_COUNT)
        )
    normalized = []
    valid_mask = []
    for index in indices:
        try:
            value = float(values[index])
        except (TypeError, ValueError, OverflowError):
            normalized.append(max_range_m)
            valid_mask.append(False)
            continue
        if math.isinf(value) and value > 0.0:
            normalized.append(max_range_m)
            valid_mask.append(True)
        elif math.isfinite(value) and value > 0.0:
            normalized.append(min(value, max_range_m))
            valid_mask.append(True)
        else:
            normalized.append(max_range_m)
            valid_mask.append(False)
    return tuple(normalized), tuple(valid_mask)


def build_fixed_map_observation(
    *,
    state: VesselState,
    preview: TrajectoryPreview,
    safe_mask: Sequence[bool],
    session_id: str,
    laser_ranges: Sequence[float],
    laser_valid_mask: Sequence[bool],
    scan_age_s: float,
    pose_age_s: float,
    hidden_reset: bool,
) -> LocalObservationV2:
    mask = tuple(safe_mask)
    if len(mask) != 5 or any(type(value) is not bool for value in mask):
        raise ValueError("fixed-map safety mask must contain five booleans")
    return LocalObservationV2(
        laser_ranges=tuple(laser_ranges),
        laser_valid_mask=tuple(laser_valid_mask),
        scan_age_s=scan_age_s,
        ego_features=(
            state.speed,
            state.yaw_rate,
            state.throttle_state,
            state.rudder_state,
        ),
        path_features=(
            preview.lookahead_x - state.x,
            preview.lookahead_y - state.y,
            preview.heading_error,
            preview.cross_track_error_m,
            0.0,
            preview.remaining_arc_length_m,
            preview.progress,
        ),
        safety_features=tuple(
            1.0 if safe else 0.0 for safe in mask
        ),
        pose_age_s=pose_age_s,
        session_id=session_id,
        stamp_sim=state.stamp_sim,
        hidden_reset=hidden_reset,
    )


__all__ = [
    "TrajectoryPreview",
    "braking_future_controls",
    "build_fixed_map_observation",
    "feedback_tracking_control",
    "front_arc_laser_features",
    "narrow_ingress_control",
    "narrow_ingress_future_controls",
    "preview_trajectory",
    "reverse_tracking_control",
    "tracking_future_controls",
]
