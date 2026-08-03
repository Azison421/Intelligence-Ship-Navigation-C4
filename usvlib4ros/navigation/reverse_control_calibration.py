"""Pure evidence rule for a bounded open-water reverse probe."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping

from usvlib4ros.planning.kinodynamic_informed_rrtstar import (
    Control,
    PrototypeReducedDynamics,
)


REVERSE_PROFILE_SCHEMA = "reverse-control-profile-v1"


@dataclass(frozen=True)
class ReverseControlEvaluation:
    verdict: str
    supported: bool
    baseline_signed_speed_mps: float
    command_signed_speed_mps: float


@dataclass(frozen=True)
class ReverseControlProfile:
    """One straight reverse primitive bound to a successful live probe."""

    source_log_sha256: str
    command_throttle: float
    command_signed_speed_mps: float
    reverse_throttle_speed_gain: float
    max_reverse_speed_mps: float
    schema_version: str = REVERSE_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        values = (
            self.command_throttle,
            self.command_signed_speed_mps,
            self.reverse_throttle_speed_gain,
            self.max_reverse_speed_mps,
        )
        if (
            len(self.source_log_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in self.source_log_sha256
            )
        ):
            raise ValueError("reverse source log hash must be SHA-256")
        if self.schema_version != REVERSE_PROFILE_SCHEMA:
            raise ValueError("reverse profile schema is incompatible")
        if not all(isfinite(value) for value in values):
            raise ValueError("reverse profile values must be finite")
        if not -1.0 <= self.command_throttle < 0.0:
            raise ValueError("reverse command throttle must be negative")
        if self.command_signed_speed_mps >= 0.0:
            raise ValueError("reverse command speed must be negative")
        if min(
            self.reverse_throttle_speed_gain,
            self.max_reverse_speed_mps,
        ) <= 0.0:
            raise ValueError("reverse dynamics values must be positive")

    @property
    def control(self) -> Control:
        return Control(throttle=self.command_throttle, rudder=0.0)

    @property
    def profile_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "source_log_sha256": self.source_log_sha256.lower(),
            "command_throttle": self.command_throttle,
            "command_signed_speed_mps": self.command_signed_speed_mps,
            "reverse_throttle_speed_gain": (
                self.reverse_throttle_speed_gain
            ),
            "max_reverse_speed_mps": self.max_reverse_speed_mps,
        }
        return sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def evaluate_reverse_response(
    *,
    baseline_signed_speed_mps: float,
    command_signed_speed_mps: float,
    minimum_reverse_speed_mps: float = 0.05,
    minimum_change_mps: float = 0.03,
) -> ReverseControlEvaluation:
    values = (
        baseline_signed_speed_mps,
        command_signed_speed_mps,
        minimum_reverse_speed_mps,
        minimum_change_mps,
    )
    if (
        not all(isfinite(value) for value in values)
        or minimum_reverse_speed_mps <= 0.0
        or minimum_change_mps <= 0.0
    ):
        raise ValueError("reverse evaluation values are invalid")
    supported = (
        command_signed_speed_mps <= -minimum_reverse_speed_mps
        and command_signed_speed_mps
        <= baseline_signed_speed_mps - minimum_change_mps
    )
    return ReverseControlEvaluation(
        verdict=(
            "reverse_supported" if supported else "reverse_unsupported"
        ),
        supported=supported,
        baseline_signed_speed_mps=baseline_signed_speed_mps,
        command_signed_speed_mps=command_signed_speed_mps,
    )


def build_reverse_control_profile(
    *,
    source_log_sha256: str,
    command_throttle: float,
    baseline_signed_speed_mps: float,
    command_signed_speed_mps: float,
    maximum_reverse_speed_mps: float = 0.2,
) -> ReverseControlProfile:
    evaluation = evaluate_reverse_response(
        baseline_signed_speed_mps=baseline_signed_speed_mps,
        command_signed_speed_mps=command_signed_speed_mps,
    )
    if not evaluation.supported:
        raise ValueError("calibration did not prove reverse capability")
    return ReverseControlProfile(
        source_log_sha256=source_log_sha256,
        command_throttle=command_throttle,
        command_signed_speed_mps=command_signed_speed_mps,
        reverse_throttle_speed_gain=(
            abs(command_signed_speed_mps) / abs(command_throttle)
        ),
        max_reverse_speed_mps=maximum_reverse_speed_mps,
    )


def reverse_control_profile_from_dict(
    payload: Mapping[str, object],
) -> ReverseControlProfile:
    if payload.get("schema_version") != REVERSE_PROFILE_SCHEMA:
        raise ValueError("reverse-control profile schema is incompatible")
    profile = ReverseControlProfile(
        source_log_sha256=str(payload.get("source_log_sha256", "")),
        command_throttle=float(payload["command_throttle"]),
        command_signed_speed_mps=float(
            payload["command_signed_speed_mps"]
        ),
        reverse_throttle_speed_gain=float(
            payload["reverse_throttle_speed_gain"]
        ),
        max_reverse_speed_mps=float(payload["max_reverse_speed_mps"]),
    )
    claimed_hash = payload.get("profile_hash")
    if claimed_hash is not None and claimed_hash != profile.profile_hash:
        raise ValueError("reverse-control profile hash is invalid")
    raw_control = payload.get("control")
    if raw_control is not None:
        if not isinstance(raw_control, Mapping):
            raise ValueError("reverse-control row is invalid")
        control = Control(
            throttle=float(raw_control["throttle"]),
            rudder=float(raw_control["rudder"]),
        )
        if control != profile.control:
            raise ValueError("reverse-control row does not match profile")
    return profile


def reverse_control_profile_to_dict(
    profile: ReverseControlProfile,
) -> dict[str, object]:
    return {
        "schema_version": profile.schema_version,
        "source_log_sha256": profile.source_log_sha256,
        "command_throttle": profile.command_throttle,
        "command_signed_speed_mps": profile.command_signed_speed_mps,
        "reverse_throttle_speed_gain": (
            profile.reverse_throttle_speed_gain
        ),
        "max_reverse_speed_mps": profile.max_reverse_speed_mps,
        "control": {
            "throttle": profile.control.throttle,
            "rudder": profile.control.rudder,
        },
        "profile_hash": profile.profile_hash,
    }


def enable_reverse_dynamics(
    dynamics: PrototypeReducedDynamics,
    profile: ReverseControlProfile,
) -> PrototypeReducedDynamics:
    return replace(
        dynamics,
        version=(
            f"{dynamics.version}-reverse-{profile.profile_hash[:12]}-v7"
        ),
        allow_reverse=True,
        max_reverse_speed=profile.max_reverse_speed_mps,
        reverse_throttle_speed_gain=(
            profile.reverse_throttle_speed_gain
        ),
    )


__all__ = [
    "REVERSE_PROFILE_SCHEMA",
    "ReverseControlProfile",
    "ReverseControlEvaluation",
    "build_reverse_control_profile",
    "enable_reverse_dynamics",
    "evaluate_reverse_response",
    "reverse_control_profile_from_dict",
    "reverse_control_profile_to_dict",
]
