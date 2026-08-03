import math
from types import SimpleNamespace

from usvlib4ros.planning import (
    Control,
    PrototypeReducedDynamics,
    VesselState,
)
from usvlib4ros.policy.fixed_map_features import (
    TrajectoryPreview,
    braking_future_controls,
    feedback_tracking_control,
    front_arc_laser_features,
    narrow_ingress_control,
    narrow_ingress_future_controls,
    preview_trajectory,
    reverse_tracking_control,
    tracking_future_controls,
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


def test_feedback_tracking_control_replaces_stale_open_loop_rudder():
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
        yaw_rate=0.0,
    )

    assert control.throttle == 0.05
    assert control.rudder == -0.5

    aligned = feedback_tracking_control(
        TrajectoryPreview(
            **{
                **preview.__dict__,
                "heading_error": 0.0,
            }
        ),
        Control(throttle=0.1, rudder=-0.5),
        PrototypeReducedDynamics(),
        yaw_rate=0.0,
    )
    assert aligned == Control(throttle=0.1, rudder=0.0)

    damping = feedback_tracking_control(
        TrajectoryPreview(
            **{
                **preview.__dict__,
                "heading_error": 0.0,
            }
        ),
        Control(throttle=0.4, rudder=-0.5),
        PrototypeReducedDynamics(),
        yaw_rate=0.6,
        speed=0.0,
    )
    assert damping.throttle == 0.22
    assert damping.rudder > 0.0

    coasting = feedback_tracking_control(
        preview,
        Control(throttle=0.4, rudder=0.5),
        PrototypeReducedDynamics(),
        yaw_rate=0.0,
        speed=0.31,
    )
    assert coasting.throttle == 0.0
    assert coasting.rudder == -0.5

    braking = feedback_tracking_control(
        preview,
        Control(throttle=0.4, rudder=0.5),
        PrototypeReducedDynamics(),
        yaw_rate=0.0,
        speed=0.41,
    )
    assert braking == Control(throttle=-0.4, rudder=0.0)

    proximity_braking = feedback_tracking_control(
        preview,
        Control(throttle=0.4, rudder=0.5),
        PrototypeReducedDynamics(),
        yaw_rate=0.0,
        speed=0.21,
        clearance_m=0.5,
    )
    assert proximity_braking == Control(throttle=-0.4, rudder=0.0)

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


def test_forward_trajectory_preview_catches_up_without_revisiting_old_states():
    states = tuple(
        VesselState(
            x=index * 0.25,
            y=0.0,
            yaw=0.0,
            speed=0.3,
            yaw_rate=0.0,
            stamp_sim=index * 0.1,
        )
        for index in range(16)
    )
    trajectory = SimpleNamespace(
        states=states,
        controls=(Control(0.25, 0.0),) * 15,
    )

    preview = preview_trajectory(
        VesselState(
            x=2.25,
            y=0.0,
            yaw=0.0,
            speed=0.3,
            yaw_rate=0.0,
        ),
        trajectory,
        previous_index=0,
    )

    assert preview.state_index == 9


def test_trajectory_preview_does_not_jump_to_overlapping_escape_branch():
    states = (
        VesselState(x=0.0, y=0.0, yaw=0.0, speed=0.2, yaw_rate=0.0),
        VesselState(x=1.0, y=0.0, yaw=0.0, speed=0.2, yaw_rate=0.0),
        VesselState(x=2.0, y=0.0, yaw=0.0, speed=0.1, yaw_rate=0.0),
        VesselState(x=1.0, y=0.0, yaw=0.0, speed=-0.1, yaw_rate=0.0),
        VesselState(x=0.0, y=0.0, yaw=0.0, speed=-0.1, yaw_rate=0.0),
    )
    trajectory = SimpleNamespace(
        states=states,
        controls=(Control(0.4, 0.0),) * 2 + (Control(-0.4, 0.0),) * 2,
    )

    preview = preview_trajectory(
        VesselState(
            x=1.0,
            y=0.0,
            yaw=0.0,
            speed=-0.05,
            yaw_rate=0.0,
        ),
        trajectory,
        previous_index=0,
    )

    assert preview.state_index == 1


def test_trajectory_preview_advances_reverse_branch_after_target_visit():
    states = (
        VesselState(x=0.0, y=0.0, yaw=0.0, speed=0.2, yaw_rate=0.0),
        VesselState(x=1.0, y=0.0, yaw=0.0, speed=0.2, yaw_rate=0.0),
        VesselState(x=2.0, y=0.0, yaw=0.0, speed=0.1, yaw_rate=0.0),
        VesselState(x=1.0, y=0.0, yaw=0.0, speed=-0.1, yaw_rate=0.0),
        VesselState(x=0.0, y=0.0, yaw=0.0, speed=-0.1, yaw_rate=0.0),
    )
    trajectory = SimpleNamespace(
        states=states,
        controls=(Control(0.4, 0.0),) * 2 + (Control(-0.4, 0.0),) * 2,
    )

    preview = preview_trajectory(
        VesselState(
            x=1.0,
            y=0.0,
            yaw=0.0,
            speed=-0.05,
            yaw_rate=0.0,
        ),
        trajectory,
        previous_index=0,
        allow_reverse_branch_progress=True,
    )

    assert preview.state_index == 3


def test_narrow_ingress_control_keeps_forward_motion_until_target():
    aligned = narrow_ingress_control(
        throttle=0.1,
    )

    assert aligned == Control(0.1, 0.0)


def test_narrow_ingress_prediction_switches_to_reverse_after_crossing():
    future = narrow_ingress_future_controls(
        Control(0.1, 0.0),
        ((Control(-0.4, 0.0), 1.7),),
    )

    assert future == (
        (Control(0.1, 0.0), 0.5),
        (Control(-0.4, 0.0), 1.7),
    )


def test_tracking_prediction_holds_closed_loop_control_before_plan():
    tracking = Control(0.25, -0.3)
    planned = ((Control(0.4, 0.5), 1.2),)

    future = tracking_future_controls(tracking, planned)

    assert future == ((tracking, 0.5), *planned)


def test_overspeed_braking_prediction_does_not_resume_stale_forward_plan():
    braking = Control(-0.4, 0.0)

    future = braking_future_controls(braking)

    assert future == (
        (braking, 0.7),
        (Control(0.0, 0.0), 1.0),
    )


def test_narrow_ingress_steers_toward_the_safe_gate():
    control = narrow_ingress_control(
        throttle=0.1,
        heading_error=-0.3,
        rudder_yaw_sign=-1.0,
    )

    assert control == Control(0.1, 0.3)


def test_reverse_tracking_corrects_the_failed_narrow_escape_pose():
    preview = TrajectoryPreview(
        state_index=12,
        nominal_control_index=12,
        cross_track_error_m=0.2284287591003575,
        remaining_arc_length_m=2.4407928425610352,
        progress=0.3157894736842105,
        lookahead_x=29.964471233362907,
        lookahead_y=99.53055405474527,
        heading_error=-2.3388284631790377,
    )

    control = reverse_tracking_control(
        preview,
        Control(-0.4, 0.0),
        PrototypeReducedDynamics(),
        yaw_rate=0.0,
    )

    assert control == Control(-0.4, -0.5)
