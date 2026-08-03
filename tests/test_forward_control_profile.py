import math
from dataclasses import asdict

import pytest

from usvlib4ros.planning.forward_control_profile import (
    ForwardProbeSafetySample,
    ForwardProbe,
    build_forward_control_profile,
    forward_probe_abort_reason,
    forward_control_profile_from_dict,
    initial_turn_probe_controls,
    reduced_dynamics_from_profile,
    supplemental_turn_probe_controls,
)
from usvlib4ros.policy.safety_supervisor import CandidateControlGenerator


def test_forward_probes_build_a_hashed_five_action_profile():
    probes = (
        ForwardProbe(0.2, 0.0, 0.25, 0.0, 0.2),
        ForwardProbe(0.2, 0.3, 0.24, -math.radians(4.0), 0.2),
        ForwardProbe(0.2, -0.3, 0.24, math.radians(4.0), 0.2),
        ForwardProbe(0.3, 0.0, 0.45, 0.0, 0.3),
        ForwardProbe(0.3, 0.3, 0.42, -math.radians(12.0), 0.3),
        ForwardProbe(0.3, -0.3, 0.41, math.radians(11.0), 0.3),
        ForwardProbe(0.3, 0.5, 0.38, -math.radians(20.0), 0.3),
        ForwardProbe(0.3, -0.5, 0.37, math.radians(18.0), 0.3),
        ForwardProbe(0.4, 0.0, 0.60, 0.0, 0.4),
    )

    profile = build_forward_control_profile(probes)

    assert profile.action_schema == "five-discrete-forward-bias-v2"
    assert profile.minimum_steerage_throttle == 0.3
    assert profile.cruise_throttle == 0.3
    assert profile.action_controls[0].rudder < 0.0
    assert profile.action_controls[4].rudder > 0.0
    assert profile.action_controls[2].rudder == 0.0
    assert len(profile.calibration_hash) == 64
    assert (
        build_forward_control_profile(tuple(reversed(probes))).calibration_hash
        == profile.calibration_hash
    )
    restored = forward_control_profile_from_dict(
        {
            "schema_version": profile.schema_version,
            "action_schema": profile.action_schema,
            "calibration_hash": profile.calibration_hash,
            "minimum_steerage_throttle": profile.minimum_steerage_throttle,
            "cruise_throttle": profile.cruise_throttle,
            "throttle_speed_gain": profile.throttle_speed_gain,
            "positive_rudder_yaw_rate_gain": (
                profile.positive_rudder_yaw_rate_gain
            ),
            "negative_rudder_yaw_rate_gain": (
                profile.negative_rudder_yaw_rate_gain
            ),
            "speed_response": profile.speed_response,
            "yaw_response": profile.yaw_response,
            "action_controls": [
                asdict(control) for control in profile.action_controls
            ],
        }
    )
    assert restored == profile
    dynamics = reduced_dynamics_from_profile(profile)
    assert dynamics.version.startswith(
        "national-test-forward-calibrated-"
    )
    assert math.isclose(
        dynamics.throttle_speed_gain,
        profile.throttle_speed_gain,
    )
    assert (
        dynamics.positive_rudder_yaw_rate_gain
        != dynamics.negative_rudder_yaw_rate_gain
    )


def test_forward_probe_start_and_active_limits_fail_closed():
    safe = {
        "pose_age_s": 0.5,
        "scan_age_s": 1.0,
        "clearance_m": 3.0,
        "minimum_laser_m": 3.0,
        "displacement_m": 0.0,
        "feedback_ok": True,
    }
    assert (
        forward_probe_abort_reason(ForwardProbeSafetySample(**safe), starting=True)
        is None
    )
    assert forward_probe_abort_reason(
        ForwardProbeSafetySample(**(safe | {"clearance_m": 2.999})),
        starting=True,
    ) == "START_CLEARANCE"
    assert forward_probe_abort_reason(
        ForwardProbeSafetySample(**(safe | {"minimum_laser_m": 2.999})),
        starting=True,
    ) == "START_LASER"

    active_safe = safe | {"clearance_m": 0.5001, "minimum_laser_m": 0.8}
    cases = (
        ({"pose_age_s": 0.5001}, "POSE_STALE"),
        ({"scan_age_s": 1.0001}, "SCAN_STALE"),
        ({"minimum_laser_m": 0.7999}, "LASER_SAFETY_STOP"),
        ({"clearance_m": 0.5}, "CLEARANCE_VIOLATION"),
        ({"displacement_m": 1.5001}, "DISPLACEMENT_LIMIT"),
        ({"feedback_ok": False}, "FEEDBACK_INVALID"),
    )
    for override, expected in cases:
        sample = ForwardProbeSafetySample(**(active_safe | override))
        assert forward_probe_abort_reason(sample, starting=False) == expected


def test_candidate_generator_keeps_rudder_endpoints_without_adding_throttle():
    probes = (
        ForwardProbe(0.3, 0.0, 0.45, 0.0, 0.3),
        ForwardProbe(0.3, 0.3, 0.42, -math.radians(12.0), 0.3),
        ForwardProbe(0.3, -0.3, 0.41, math.radians(11.0), 0.3),
        ForwardProbe(0.2, 0.5, 0.35, -math.radians(18.0), 0.2),
        ForwardProbe(0.2, -0.5, 0.34, math.radians(17.0), 0.2),
    )
    profile = build_forward_control_profile(probes)

    candidates = CandidateControlGenerator(
        max_throttle=0.4,
        max_abs_rudder=0.6,
        action_controls=profile.action_controls,
    ).generate(nominal_throttle=0.01, nominal_rudder=0.1)

    assert tuple(item.control.throttle for item in candidates) == (
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
    )
    assert candidates[0].control.rudder == -0.5
    assert candidates[1].control.rudder == pytest.approx(-0.2)
    assert candidates[2].control.rudder == 0.1
    assert candidates[3].control.rudder == pytest.approx(0.4)
    assert candidates[4].control.rudder == 0.5


def test_forward_calibration_schedule_is_adaptive_and_bounded():
    assert initial_turn_probe_controls((0.1, 0.3)) == (
        (0.1, 0.3),
        (0.1, -0.3),
        (0.3, 0.3),
        (0.3, -0.3),
    )
    profile = build_forward_control_profile(
        (
            ForwardProbe(0.3, 0.0, 0.45, 0.0, 0.3),
            ForwardProbe(0.3, 0.3, 0.42, -math.radians(12.0), 0.3),
            ForwardProbe(0.3, -0.3, 0.41, math.radians(11.0), 0.3),
            ForwardProbe(0.4, 0.0, 0.55, 0.0, 0.4),
        )
    )
    assert supplemental_turn_probe_controls(profile) == (
        (0.3, 0.2),
        (0.3, -0.2),
        (0.3, 0.5),
        (0.3, -0.5),
    )
