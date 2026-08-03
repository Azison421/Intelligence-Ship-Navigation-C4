"""Versioned forward-control envelope derived from bounded calibration probes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite, radians
from statistics import median
from typing import Iterable, Mapping

from .kinodynamic_informed_rrtstar import (
    Control,
    PrototypeReducedDynamics,
)


ACTION_SCHEMA = "five-discrete-forward-bias-v2"
PROFILE_SCHEMA = "forward-control-profile-v1"
MINIMUM_STEERAGE_YAW_RATE = radians(8.0)
TARGET_CRUISE_SPEED_MPS = 0.4
STRAIGHT_PROBE_THROTTLES = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)


@dataclass(frozen=True)
class ForwardProbe:
    throttle: float
    rudder: float
    speed_mps: float
    yaw_rate_rad_s: float
    coast_distance_m: float

    def __post_init__(self) -> None:
        values = (
            self.throttle,
            self.rudder,
            self.speed_mps,
            self.yaw_rate_rad_s,
            self.coast_distance_m,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("forward probe values must be finite")
        if not 0.0 < self.throttle <= 1.0:
            raise ValueError("forward probe throttle must be in (0, 1]")
        if not -1.0 <= self.rudder <= 1.0:
            raise ValueError("forward probe rudder must be in [-1, 1]")
        if self.speed_mps < 0.0 or self.coast_distance_m < 0.0:
            raise ValueError("forward probe speed and coast distance must be non-negative")


@dataclass(frozen=True)
class ForwardProbeSafetySample:
    pose_age_s: float
    scan_age_s: float
    clearance_m: float
    minimum_laser_m: float
    displacement_m: float
    feedback_ok: bool

    def __post_init__(self) -> None:
        values = (
            self.pose_age_s,
            self.scan_age_s,
            self.clearance_m,
            self.minimum_laser_m,
            self.displacement_m,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("forward probe safety sample is invalid")


@dataclass(frozen=True)
class ForwardControlProfile:
    calibration_hash: str
    minimum_steerage_throttle: float
    cruise_throttle: float
    action_controls: tuple[Control, ...]
    throttle_speed_gain: float = 1.0
    positive_rudder_yaw_rate_gain: float = 1.0
    negative_rudder_yaw_rate_gain: float = 1.0
    speed_response: float = 0.8
    yaw_response: float = 4.0
    action_schema: str = ACTION_SCHEMA
    schema_version: str = PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if len(self.calibration_hash) != 64:
            raise ValueError("calibration hash must be SHA-256")
        if not 0.0 < self.minimum_steerage_throttle <= 1.0:
            raise ValueError("minimum steerage throttle must be in (0, 1]")
        if not 0.0 < self.cruise_throttle <= 1.0:
            raise ValueError("cruise throttle must be in (0, 1]")
        if len(self.action_controls) != 5 or not all(
            control.is_valid() and control.throttle >= 0.0
            for control in self.action_controls
        ):
            raise ValueError("profile must contain five valid non-negative controls")
        gains = (
            self.throttle_speed_gain,
            self.positive_rudder_yaw_rate_gain,
            self.negative_rudder_yaw_rate_gain,
            self.speed_response,
            self.yaw_response,
        )
        if not all(isfinite(value) and value > 0.0 for value in gains):
            raise ValueError("profile dynamics gains must be positive and finite")


def _canonical_hash(probes: Iterable[ForwardProbe]) -> str:
    rows = sorted(
        (
            probe.throttle,
            probe.rudder,
            probe.speed_mps,
            probe.yaw_rate_rad_s,
            probe.coast_distance_m,
        )
        for probe in probes
    )
    payload = {
        "action_schema": ACTION_SCHEMA,
        "profile_schema": PROFILE_SCHEMA,
        "probes": rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _turn_radius(probe: ForwardProbe) -> float:
    return probe.speed_mps / max(abs(probe.yaw_rate_rad_s), 1e-12)


def forward_probe_abort_reason(
    sample: ForwardProbeSafetySample,
    *,
    starting: bool,
) -> str | None:
    """Return the first locked violation for one independently reset probe."""

    if not sample.feedback_ok:
        return "FEEDBACK_INVALID"
    if sample.pose_age_s > 0.5:
        return "POSE_STALE"
    if sample.scan_age_s > 1.0:
        return "SCAN_STALE"
    if starting:
        if sample.minimum_laser_m < 3.0:
            return "START_LASER"
        if sample.clearance_m < 3.0:
            return "START_CLEARANCE"
        return None
    if sample.minimum_laser_m < 0.8:
        return "LASER_SAFETY_STOP"
    if sample.clearance_m <= 0.5:
        return "CLEARANCE_VIOLATION"
    if sample.displacement_m > 1.5:
        return "DISPLACEMENT_LIMIT"
    return None


def build_forward_control_profile(
    probes: Iterable[ForwardProbe],
) -> ForwardControlProfile:
    """Select the lowest proven steerage/cruise envelope and asymmetric turns."""

    samples = tuple(probes)
    if not samples:
        raise ValueError("at least one forward probe is required")

    by_throttle: dict[float, list[ForwardProbe]] = {}
    for probe in samples:
        by_throttle.setdefault(probe.throttle, []).append(probe)

    steerage_throttle = None
    for throttle in sorted(by_throttle):
        group = by_throttle[throttle]
        positive_yaw = any(
            probe.yaw_rate_rad_s >= MINIMUM_STEERAGE_YAW_RATE
            for probe in group
            if probe.rudder != 0.0
        )
        negative_yaw = any(
            probe.yaw_rate_rad_s <= -MINIMUM_STEERAGE_YAW_RATE
            for probe in group
            if probe.rudder != 0.0
        )
        if positive_yaw and negative_yaw:
            steerage_throttle = throttle
            break
    if steerage_throttle is None:
        raise ValueError("calibration did not prove steerage in both directions")

    straight = sorted(
        (probe for probe in samples if abs(probe.rudder) <= 1e-12),
        key=lambda probe: probe.throttle,
    )
    if not straight:
        raise ValueError("calibration requires at least one straight probe")
    cruise = next(
        (probe for probe in straight if probe.speed_mps >= TARGET_CRUISE_SPEED_MPS),
        max(straight, key=lambda probe: probe.speed_mps),
    )

    proven_turns = tuple(
        probe
        for probe in samples
        if probe.throttle >= steerage_throttle
        and abs(probe.yaw_rate_rad_s) >= MINIMUM_STEERAGE_YAW_RATE
        and probe.rudder != 0.0
    )
    left_turns = tuple(probe for probe in proven_turns if probe.yaw_rate_rad_s > 0.0)
    right_turns = tuple(probe for probe in proven_turns if probe.yaw_rate_rad_s < 0.0)
    if not left_turns or not right_turns:
        raise ValueError("calibration did not produce two-sided turn controls")

    hard_left = min(left_turns, key=lambda probe: (_turn_radius(probe), probe.throttle))
    hard_right = min(right_turns, key=lambda probe: (_turn_radius(probe), probe.throttle))
    soft_left = min(
        left_turns,
        key=lambda probe: (abs(probe.rudder), probe.throttle, _turn_radius(probe)),
    )
    soft_right = min(
        right_turns,
        key=lambda probe: (abs(probe.rudder), probe.throttle, _turn_radius(probe)),
    )

    return ForwardControlProfile(
        calibration_hash=_canonical_hash(samples),
        minimum_steerage_throttle=steerage_throttle,
        cruise_throttle=cruise.throttle,
        action_controls=(
            Control(hard_left.throttle, hard_left.rudder),
            Control(cruise.throttle, soft_left.rudder),
            Control(cruise.throttle, 0.0),
            Control(cruise.throttle, soft_right.rudder),
            Control(hard_right.throttle, hard_right.rudder),
        ),
        throttle_speed_gain=median(
            probe.speed_mps / probe.throttle for probe in straight
        ),
        positive_rudder_yaw_rate_gain=median(
            abs(probe.yaw_rate_rad_s) / abs(probe.rudder)
            for probe in proven_turns
            if probe.rudder > 0.0
        ),
        negative_rudder_yaw_rate_gain=median(
            abs(probe.yaw_rate_rad_s) / abs(probe.rudder)
            for probe in proven_turns
            if probe.rudder < 0.0
        ),
    )


def initial_turn_probe_controls(
    effective_throttles: Iterable[float],
) -> tuple[tuple[float, float], ...]:
    """Return the initial two-sided 30% rudder trials only where thrust worked."""

    throttles = tuple(sorted(set(float(value) for value in effective_throttles)))
    if any(
        not isfinite(throttle) or not 0.0 < throttle <= 1.0
        for throttle in throttles
    ):
        raise ValueError("effective throttle list is invalid")
    return tuple(
        (throttle, rudder)
        for throttle in throttles
        for rudder in (0.3, -0.3)
    )


def supplemental_turn_probe_controls(
    profile: ForwardControlProfile,
) -> tuple[tuple[float, float], ...]:
    """Add only the locked 20%/50% turn probes at selected throttle levels."""

    throttles = tuple(
        sorted(
            {
                profile.minimum_steerage_throttle,
                profile.cruise_throttle,
            }
        )
    )
    return tuple(
        (throttle, rudder)
        for throttle in throttles
        for rudder in (0.2, -0.2, 0.5, -0.5)
    )


def forward_control_profile_from_dict(
    payload: Mapping[str, object],
) -> ForwardControlProfile:
    """Load a profile without accepting a different schema or action protocol."""

    if payload.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("forward-control profile schema is incompatible")
    if payload.get("action_schema") != ACTION_SCHEMA:
        raise ValueError("forward-control action schema is incompatible")
    raw_controls = payload.get("action_controls")
    if not isinstance(raw_controls, list) or len(raw_controls) != 5:
        raise ValueError("forward-control action table is invalid")
    controls = []
    for item in raw_controls:
        if not isinstance(item, Mapping):
            raise ValueError("forward-control action row is invalid")
        controls.append(
            Control(
                throttle=float(item["throttle"]),
                rudder=float(item["rudder"]),
            )
        )
    return ForwardControlProfile(
        calibration_hash=str(payload.get("calibration_hash", "")),
        minimum_steerage_throttle=float(
            payload["minimum_steerage_throttle"]
        ),
        cruise_throttle=float(payload["cruise_throttle"]),
        action_controls=tuple(controls),
        throttle_speed_gain=float(payload["throttle_speed_gain"]),
        positive_rudder_yaw_rate_gain=float(
            payload["positive_rudder_yaw_rate_gain"]
        ),
        negative_rudder_yaw_rate_gain=float(
            payload["negative_rudder_yaw_rate_gain"]
        ),
        speed_response=float(payload.get("speed_response", 0.8)),
        yaw_response=float(payload.get("yaw_response", 4.0)),
    )


def reduced_dynamics_from_profile(
    profile: ForwardControlProfile,
) -> PrototypeReducedDynamics:
    """Bind the reduced model to the measured forward-control profile."""

    return PrototypeReducedDynamics(
        version=(
            "national-test-forward-calibrated-"
            f"{profile.calibration_hash[:12]}-v6"
        ),
        throttle_speed_gain=profile.throttle_speed_gain,
        rudder_yaw_rate_gain=min(
            profile.positive_rudder_yaw_rate_gain,
            profile.negative_rudder_yaw_rate_gain,
        ),
        positive_rudder_yaw_rate_gain=(
            profile.positive_rudder_yaw_rate_gain
        ),
        negative_rudder_yaw_rate_gain=(
            profile.negative_rudder_yaw_rate_gain
        ),
        speed_response=profile.speed_response,
        yaw_response=profile.yaw_response,
    )


def diagnostic_forward_control_profile() -> ForwardControlProfile:
    """Non-promotable hint from the earlier bounded 30% live response log."""

    return build_forward_control_profile(
        (
            ForwardProbe(0.3, 0.0, 0.3635, 0.0, 0.0),
            ForwardProbe(0.3, 0.3, 0.3635, -0.5027, 0.0),
            ForwardProbe(0.3, -0.3, 0.3635, 0.4068, 0.0),
        )
    )


def action_protocol_hash(profile: ForwardControlProfile) -> str:
    payload = {
        "action_schema": profile.action_schema,
        "controls": [
            {
                "throttle": control.throttle,
                "rudder": control.rudder,
            }
            for control in profile.action_controls
        ],
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ACTION_SCHEMA",
    "ForwardControlProfile",
    "ForwardProbe",
    "ForwardProbeSafetySample",
    "STRAIGHT_PROBE_THROTTLES",
    "build_forward_control_profile",
    "action_protocol_hash",
    "diagnostic_forward_control_profile",
    "forward_probe_abort_reason",
    "forward_control_profile_from_dict",
    "initial_turn_probe_controls",
    "reduced_dynamics_from_profile",
    "supplemental_turn_probe_controls",
]
