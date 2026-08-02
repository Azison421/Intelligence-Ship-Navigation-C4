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
) -> TrajectoryPreview:
    if not trajectory.controls or len(trajectory.states) < 2:
        raise ValueError("trajectory preview requires at least one control")
    if not 0 <= previous_index < len(trajectory.states):
        raise ValueError("previous trajectory index is out of range")
    index = min(
        range(previous_index, len(trajectory.states)),
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
) -> Control:
    """Center local actions on the live line-of-sight heading error."""

    rudder = max(
        -0.1,
        min(
            0.1,
            preview.heading_error * dynamics.rudder_yaw_sign,
        ),
    )
    return Control(
        throttle=min(0.02, max(0.0, nominal_control.throttle)),
        rudder=rudder,
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
    "build_fixed_map_observation",
    "feedback_tracking_control",
    "front_arc_laser_features",
    "preview_trajectory",
]
