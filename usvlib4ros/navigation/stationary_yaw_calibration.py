"""Pure evaluation rules for the bounded Unity stationary-yaw probe."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, radians
from statistics import median


@dataclass(frozen=True)
class CalibrationPhase:
    yaw_rates_rad_s: tuple[float, ...]
    displacement_m: float
    minimum_clearance_m: float
    minimum_laser_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "yaw_rates_rad_s",
            tuple(float(value) for value in self.yaw_rates_rad_s),
        )
        values = (
            *self.yaw_rates_rad_s,
            self.displacement_m,
            self.minimum_clearance_m,
            self.minimum_laser_m,
        )
        if (
            not self.yaw_rates_rad_s
            or not all(isfinite(value) for value in values)
            or self.displacement_m < 0.0
        ):
            raise ValueError("calibration phase is invalid")


@dataclass(frozen=True)
class ProbeSafetySample:
    pose_age_s: float
    scan_age_s: float
    clearance_m: float
    minimum_laser_m: float
    displacement_m: float

    def __post_init__(self) -> None:
        values = (
            self.pose_age_s,
            self.scan_age_s,
            self.clearance_m,
            self.minimum_laser_m,
            self.displacement_m,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("probe safety sample is invalid")


@dataclass(frozen=True)
class StationaryYawCalibrationResult:
    verdict: str
    positive_rate_rad_s: float
    negative_rate_rad_s: float
    noise_threshold_rad_s: float
    opposite_sign_response: bool
    maximum_displacement_m: float


def evaluate_stationary_yaw(
    *,
    baseline: CalibrationPhase,
    positive: CalibrationPhase,
    negative: CalibrationPhase,
    maximum_displacement_m: float = 0.15,
    functional_rate_rad_s: float = radians(0.5),
    competition_rate_rad_s: float = radians(8.0),
) -> StationaryYawCalibrationResult:
    """Classify zero-throttle yaw response without connection metadata."""

    if (
        maximum_displacement_m <= 0.0
        or functional_rate_rad_s <= 0.0
        or competition_rate_rad_s < functional_rate_rad_s
    ):
        raise ValueError("calibration thresholds are invalid")
    positive_rate = median(positive.yaw_rates_rad_s)
    negative_rate = median(negative.yaw_rates_rad_s)
    baseline_noise = max(abs(value) for value in baseline.yaw_rates_rad_s)
    noise_threshold = max(functional_rate_rad_s, 5.0 * baseline_noise)
    opposite = positive_rate * negative_rate < 0.0
    maximum_drift = max(
        baseline.displacement_m,
        positive.displacement_m,
        negative.displacement_m,
    )
    usable = (
        opposite
        and abs(positive_rate) >= noise_threshold
        and abs(negative_rate) >= noise_threshold
        and maximum_drift <= maximum_displacement_m
    )
    if not usable:
        verdict = "unsupported"
    elif min(abs(positive_rate), abs(negative_rate)) >= competition_rate_rad_s:
        verdict = "competition_ready"
    else:
        verdict = "functional_only"
    return StationaryYawCalibrationResult(
        verdict=verdict,
        positive_rate_rad_s=positive_rate,
        negative_rate_rad_s=negative_rate,
        noise_threshold_rad_s=noise_threshold,
        opposite_sign_response=opposite,
        maximum_displacement_m=maximum_drift,
    )


def stationary_yaw_abort_reason(
    sample: ProbeSafetySample,
    *,
    maximum_displacement_m: float = 0.15,
) -> str | None:
    """Return the first locked safety violation for a live probe sample."""

    if not isfinite(maximum_displacement_m) or maximum_displacement_m <= 0.0:
        raise ValueError("maximum displacement must be positive and finite")
    if sample.pose_age_s > 0.5:
        return "POSE_STALE"
    if sample.scan_age_s > 1.0:
        return "SCAN_STALE"
    if sample.minimum_laser_m < 0.6:
        return "LASER_EMERGENCY_STOP"
    if sample.clearance_m <= 0.2:
        return "CLEARANCE_VIOLATION"
    if sample.displacement_m > maximum_displacement_m:
        return "DISPLACEMENT_LIMIT"
    return None


__all__ = [
    "CalibrationPhase",
    "ProbeSafetySample",
    "StationaryYawCalibrationResult",
    "evaluate_stationary_yaw",
    "stationary_yaw_abort_reason",
]
