import math
from types import SimpleNamespace

from usvlib4ros.planning import (
    Control,
    PrototypeReducedDynamics,
    VesselState,
)
from usvlib4ros.policy.fixed_map_features import (
    TrajectoryPreview,
    feedback_tracking_control,
    front_arc_laser_features,
    preview_trajectory,
)


def test_front_arc_laser_features_match_sample_first_and_last_beams():
    ranges = tuple(float(index) for index in range(1, 181))

    values, mask = front_arc_laser_features(
        ranges,
        max_range_m=200.0,
    )

    assert values[:36] == ranges[:36]
    assert values[36:] == ranges[-36:]
    assert all(mask)


def test_front_arc_laser_features_distinguish_clear_and_invalid_beams():
    ranges = [5.0] * 72
    ranges[0] = float("inf")
    ranges[1] = float("nan")
    ranges[2] = None
    ranges[3] = -1.0

    values, mask = front_arc_laser_features(ranges)

    assert values[:4] == (20.0, 20.0, 20.0, 20.0)
    assert mask[:4] == (True, False, False, False)
    assert math.isfinite(sum(values))


def test_feedback_tracking_control_reverses_stale_open_loop_rudder():
    preview = TrajectoryPreview(
        state_index=10,
        nominal_control_index=10,
        cross_track_error_m=0.2,
        remaining_arc_length_m=5.0,
        progress=0.5,
        lookahead_x=0.0,
        lookahead_y=1.0,
        heading_error=math.pi / 2.0,
    )

    control = feedback_tracking_control(
        preview,
        Control(throttle=0.05, rudder=0.1),
        PrototypeReducedDynamics(),
    )

    assert control.throttle == 0.02
    assert control.rudder == -0.1


def test_trajectory_preview_uses_metric_lookahead():
    states = tuple(
        VesselState(
            x=index * 0.1,
            y=0.0,
            yaw=0.0,
            speed=0.3,
            yaw_rate=0.0,
            stamp_sim=index * 0.1,
        )
        for index in range(21)
    )
    trajectory = SimpleNamespace(
        states=states,
        controls=(Control(0.05, 0.0),) * 20,
    )

    preview = preview_trajectory(states[0], trajectory, 0)

    assert preview.lookahead_x >= 1.0
