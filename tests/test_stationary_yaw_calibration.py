import math

from usvlib4ros.navigation.stationary_yaw_calibration import (
    CalibrationPhase,
    ProbeSafetySample,
    evaluate_stationary_yaw,
    stationary_yaw_abort_reason,
)


def test_opposite_low_drift_yaw_responses_are_competition_ready():
    baseline = CalibrationPhase(
        yaw_rates_rad_s=(0.0, 0.001, -0.001),
        displacement_m=0.01,
        minimum_clearance_m=1.0,
        minimum_laser_m=2.0,
    )
    positive = CalibrationPhase(
        yaw_rates_rad_s=tuple(math.radians(value) for value in (8.2, 8.5, 8.7)),
        displacement_m=0.04,
        minimum_clearance_m=0.9,
        minimum_laser_m=1.8,
    )
    negative = CalibrationPhase(
        yaw_rates_rad_s=tuple(math.radians(value) for value in (-8.1, -8.4, -8.6)),
        displacement_m=0.05,
        minimum_clearance_m=0.9,
        minimum_laser_m=1.8,
    )

    result = evaluate_stationary_yaw(
        baseline=baseline,
        positive=positive,
        negative=negative,
    )

    assert result.verdict == "competition_ready"
    assert result.opposite_sign_response
    assert result.maximum_displacement_m == 0.05


def test_probe_aborts_on_each_locked_safety_limit():
    safe = {
        "pose_age_s": 0.5,
        "scan_age_s": 1.0,
        "clearance_m": 0.21,
        "minimum_laser_m": 0.6,
        "displacement_m": 0.15,
    }
    cases = (
        ({"pose_age_s": 0.5001}, "POSE_STALE"),
        ({"scan_age_s": 1.0001}, "SCAN_STALE"),
        ({"clearance_m": 0.2}, "CLEARANCE_VIOLATION"),
        ({"minimum_laser_m": 0.5999}, "LASER_EMERGENCY_STOP"),
        ({"displacement_m": 0.1501}, "DISPLACEMENT_LIMIT"),
    )

    for override, expected in cases:
        sample = ProbeSafetySample(**(safe | override))
        assert stationary_yaw_abort_reason(sample) == expected

    assert stationary_yaw_abort_reason(ProbeSafetySample(**safe)) is None


def test_probe_displacement_limit_can_be_relaxed_for_open_water_diagnostics():
    sample = ProbeSafetySample(
        pose_age_s=0.1,
        scan_age_s=0.1,
        clearance_m=3.0,
        minimum_laser_m=3.0,
        displacement_m=0.5,
    )

    assert stationary_yaw_abort_reason(sample) == "DISPLACEMENT_LIMIT"
    assert (
        stationary_yaw_abort_reason(
            sample,
            maximum_displacement_m=1.5,
        )
        is None
    )
