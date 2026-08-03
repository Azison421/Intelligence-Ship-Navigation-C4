from usvlib4ros.navigation.reverse_control_calibration import (
    ReverseControlProfile,
    build_reverse_control_profile,
    enable_reverse_dynamics,
    evaluate_reverse_response,
    reverse_control_profile_from_dict,
)
from usvlib4ros.planning.forward_control_profile import (
    diagnostic_forward_control_profile,
    reduced_dynamics_from_profile,
)


def test_negative_longitudinal_response_proves_reverse_capability():
    result = evaluate_reverse_response(
        baseline_signed_speed_mps=0.01,
        command_signed_speed_mps=-0.24,
    )

    assert result.supported
    assert result.verdict == "reverse_supported"


def test_drift_or_forward_motion_does_not_prove_reverse_capability():
    drift = evaluate_reverse_response(
        baseline_signed_speed_mps=-0.02,
        command_signed_speed_mps=-0.04,
    )
    forward = evaluate_reverse_response(
        baseline_signed_speed_mps=0.0,
        command_signed_speed_mps=0.2,
    )

    assert not drift.supported
    assert not forward.supported
    assert drift.verdict == "reverse_unsupported"
    assert forward.verdict == "reverse_unsupported"


def test_supported_reverse_probe_builds_hashed_profile_and_dynamics():
    source_hash = "a" * 64
    profile = build_reverse_control_profile(
        source_log_sha256=source_hash,
        command_throttle=-0.4,
        baseline_signed_speed_mps=-0.001,
        command_signed_speed_mps=-0.12,
    )

    assert profile.control.throttle == -0.4
    assert profile.control.rudder == 0.0
    assert profile.reverse_throttle_speed_gain == 0.3
    assert len(profile.profile_hash) == 64
    assert reverse_control_profile_from_dict(
        {
            **profile.__dict__,
            "control": {
                "throttle": profile.control.throttle,
                "rudder": profile.control.rudder,
            },
        }
    ) == profile

    dynamics = enable_reverse_dynamics(
        reduced_dynamics_from_profile(
            diagnostic_forward_control_profile()
        ),
        profile,
    )
    assert dynamics.allow_reverse
    assert dynamics.max_reverse_speed == profile.max_reverse_speed_mps
    assert (
        dynamics.reverse_throttle_speed_gain
        == profile.reverse_throttle_speed_gain
    )


def test_reverse_profile_rejects_unproven_or_tampered_evidence():
    try:
        build_reverse_control_profile(
            source_log_sha256="b" * 64,
            command_throttle=-0.4,
            baseline_signed_speed_mps=-0.02,
            command_signed_speed_mps=-0.04,
        )
    except ValueError as exc:
        assert "prove reverse" in str(exc)
    else:
        raise AssertionError("unproven reverse response was accepted")

    profile = ReverseControlProfile(
        source_log_sha256="c" * 64,
        command_throttle=-0.4,
        command_signed_speed_mps=-0.12,
        reverse_throttle_speed_gain=0.3,
        max_reverse_speed_mps=0.2,
    )
    payload = {
        **profile.__dict__,
        "profile_hash": "0" * 64,
        "control": {"throttle": -0.4, "rudder": 0.0},
    }
    try:
        reverse_control_profile_from_dict(payload)
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("tampered reverse profile was accepted")
