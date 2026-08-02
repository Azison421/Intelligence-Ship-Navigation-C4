"""Versioned, pure-Python Kinodynamic Informed RRT* research implementation.

The module owns planning contracts, a bounded reduced-order dynamics model, a
rewiring tree, informed sampling and an independent replay validator.  It has
no ROS, Unity or MATLAB dependency.  The prototype dynamics must be replaced
by the calibrated MATLAB export before the planner can enter a production
control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from heapq import heappop, heappush
from hashlib import sha256
from math import atan2, ceil, cos, hypot, isfinite, pi, sin, sqrt
from random import Random
from time import perf_counter
from typing import Iterable, Optional, Sequence


MAX_REQUEST_TIME_BUDGET_MS = 60_000.0
MAX_PLANNER_NODES = 10_000
MAX_EDGE_DURATION_S = 5.0
MAX_PROPAGATION_STEPS = 100

# Maps larger than this many cells use a lazily built exact Euclidean distance
# transform for clearance queries.  The LUT path is conservative: the cell
# lookup can only underestimate the exact per-point clearance, never allow a
# state the exact path would reject.  Small maps keep the exact per-cell loop.
_DISTANCE_FIELD_MIN_CELLS = 4096
_DISTANCE_FIELD_CACHE: dict = {}


def _edt_1d(f):
    """Felzenszwalb-Huttenlocher exact squared Euclidean distance transform."""

    import numpy as np

    n = len(f)
    d = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.int64)
    z = np.zeros(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def _build_distance_field(rows: tuple) -> object:
    """Exact EDT (in cells) to the nearest hard ('#'/'?') cell center."""

    import numpy as np

    grid = np.array([[cell in "#?" for cell in row] for row in rows], dtype=bool)
    height, width = grid.shape
    inf = float(height + width + 1)
    squared = np.where(grid, 0.0, inf * inf)
    for y in range(height):
        squared[y] = _edt_1d(squared[y])
    for x in range(width):
        squared[:, x] = _edt_1d(squared[:, x])
    return np.sqrt(squared)


def _finite(value: object) -> bool:
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(float(value))
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_all(values: Iterable[object]) -> bool:
    try:
        return all(_finite(value) for value in values)
    except TypeError:
        return False


def _p95(values: Sequence[float]) -> float:
    if not values:
        return float("inf")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _angle_difference(first: float, second: float) -> float:
    return (first - second + pi) % (2.0 * pi) - pi


def _state_close(first: "VesselState", second: "VesselState", tolerance: float) -> bool:
    return (
        hypot(first.x - second.x, first.y - second.y) <= tolerance
        and abs(_angle_difference(first.yaw, second.yaw)) <= tolerance
        and abs(first.speed - second.speed) <= tolerance
        and abs(first.yaw_rate - second.yaw_rate) <= tolerance
        and abs(first.throttle_state - second.throttle_state) <= tolerance
        and abs(first.rudder_state - second.rudder_state) <= tolerance
        and abs(first.stamp_sim - second.stamp_sim) <= tolerance
    )


class PlanStatus(str, Enum):
    SUCCESS = "SUCCESS"
    TIME_BUDGET_WITH_VALID_SOLUTION = "TIME_BUDGET_WITH_VALID_SOLUTION"
    TIMEOUT_NO_SOLUTION = "TIMEOUT_NO_SOLUTION"
    NO_PATH = "NO_PATH"
    INVALID_START = "INVALID_START"
    INVALID_GOAL = "INVALID_GOAL"
    START_OCCUPIED = "START_OCCUPIED"
    GOAL_OCCUPIED = "GOAL_OCCUPIED"
    INVALID_MAP = "INVALID_MAP"
    INVALID_REQUEST = "INVALID_REQUEST"
    DYNAMICS_ERROR = "DYNAMICS_ERROR"
    STEERING_UNAVAILABLE = "STEERING_UNAVAILABLE"
    STALE_REQUEST = "STALE_REQUEST"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class VesselState:
    """Reduced state in the explicit map frame and SI units."""

    x: float
    y: float
    yaw: float
    speed: float
    yaw_rate: float
    throttle_state: float = 0.0
    rudder_state: float = 0.0
    frame_id: str = "map"
    stamp_sim: float = 0.0
    state_version: str = "state-v1"
    health: str = "healthy"

    def is_finite(self) -> bool:
        return _finite_all(
            (
                self.x,
                self.y,
                self.yaw,
                self.speed,
                self.yaw_rate,
                self.throttle_state,
                self.rudder_state,
                self.stamp_sim,
            )
        )


@dataclass(frozen=True)
class Control:
    """Normalized throttle/rudder command in [-1, 1]."""

    throttle: float
    rudder: float

    def is_valid(self) -> bool:
        return (
            _finite_all((self.throttle, self.rudder))
            and -1.0 <= self.throttle <= 1.0
            and -1.0 <= self.rudder <= 1.0
        )


@dataclass(frozen=True)
class GoalRegion:
    x: float
    y: float
    position_tolerance: float = 0.5
    desired_yaw: Optional[float] = None
    heading_tolerance: float = pi
    speed_limit: float = 0.5
    yaw_rate_limit: float = 0.5

    def is_valid(self) -> bool:
        return (
            _finite_all(
                (
                    self.x,
                    self.y,
                    self.position_tolerance,
                    self.heading_tolerance,
                    self.speed_limit,
                    self.yaw_rate_limit,
                )
            )
            and self.position_tolerance >= 0.0
            and 0.0 <= self.heading_tolerance <= pi
            and self.speed_limit >= 0.0
            and self.yaw_rate_limit >= 0.0
            and (self.desired_yaw is None or _finite(self.desired_yaw))
        )

    def contains(self, state: VesselState) -> bool:
        if not self.is_valid() or not state.is_finite():
            return False
        return (
            hypot(state.x - self.x, state.y - self.y)
            <= self.position_tolerance + 1e-9
            and (
                self.desired_yaw is None
                or abs(_angle_difference(state.yaw, self.desired_yaw))
                <= self.heading_tolerance + 1e-9
            )
            and state.speed <= self.speed_limit + 1e-9
            and abs(state.yaw_rate) <= self.yaw_rate_limit + 1e-9
        )


@dataclass(frozen=True)
class MotionCheck:
    valid: bool
    reason: str
    min_clearance: float


@dataclass(frozen=True)
class CircularObstacle:
    """Exact fixed circular obstacle in the planning-map frame."""

    x: float
    y: float
    radius: float

    def __post_init__(self) -> None:
        if not _finite_all((self.x, self.y, self.radius)) or self.radius <= 0.0:
            raise ValueError("circular obstacle geometry is invalid")


@dataclass(frozen=True)
class PlanningMapSnapshot:
    """Immutable research snapshot; unknown cells are conservatively blocked."""

    snapshot_id: str
    session_id: str
    source_version: int
    map_frame: str
    resolution: float
    rows: tuple[str, ...]
    footprint_radius: float = 0.0
    required_clearance: float = 0.05
    coverage_status: str = "complete_prior"
    stamp_sim: float = 0.0
    source_artifact_hash: str = "synthetic-artifact"
    payload_content_hash: str = ""
    compiler_config_hash: str = "planning-grid-v1"
    circular_obstacles: tuple[CircularObstacle, ...] = ()

    def __post_init__(self) -> None:
        try:
            normalized_rows = tuple(self.rows)
        except TypeError as exc:
            raise ValueError("map rows must be a sequence") from exc
        object.__setattr__(self, "rows", normalized_rows)
        try:
            normalized_obstacles = tuple(self.circular_obstacles)
        except TypeError as exc:
            raise ValueError("circular obstacles must be a sequence") from exc
        if any(
            not isinstance(obstacle, CircularObstacle)
            for obstacle in normalized_obstacles
        ):
            raise ValueError("circular obstacles must use CircularObstacle")
        object.__setattr__(self, "circular_obstacles", normalized_obstacles)
        if not self.snapshot_id or not self.session_id or not self.map_frame:
            raise ValueError("map identity fields must be non-empty")
        if (
            not isinstance(self.source_version, int)
            or isinstance(self.source_version, bool)
            or self.source_version < 0
        ):
            raise ValueError("source_version must be a non-negative integer")
        if not normalized_rows or any(not isinstance(row, str) or not row for row in normalized_rows):
            raise ValueError("map rows must be non-empty strings")
        width = len(normalized_rows[0])
        if any(len(row) != width for row in normalized_rows):
            raise ValueError("map rows must be rectangular")
        if any(cell not in ".#?" for row in normalized_rows for cell in row):
            raise ValueError("map cells must be '.', '#' or '?'")
        if not _finite_all(
            (self.resolution, self.footprint_radius, self.required_clearance, self.stamp_sim)
        ):
            raise ValueError("map geometry and timestamp must be finite")
        if self.resolution <= 0.0 or self.footprint_radius < 0.0 or self.required_clearance < 0.0:
            raise ValueError("map geometry bounds are invalid")
        if not self.source_artifact_hash or not self.compiler_config_hash:
            raise ValueError("map artifact and compiler hashes are required")
        canonical_hash = self.canonical_payload_hash()
        if not self.payload_content_hash:
            object.__setattr__(self, "payload_content_hash", canonical_hash)
        elif self.payload_content_hash != canonical_hash:
            raise ValueError("payload hash does not match map content")

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[str],
        *,
        snapshot_id: str,
        session_id: str,
        source_version: int,
        map_frame: str = "map",
        resolution: float = 1.0,
        footprint_radius: float = 0.0,
        required_clearance: float = 0.05,
        coverage_status: str = "complete_prior",
        stamp_sim: float = 0.0,
        source_artifact_hash: str = "synthetic-artifact",
        payload_content_hash: str = "",
        compiler_config_hash: str = "planning-grid-v1",
        circular_obstacles: Sequence[CircularObstacle] = (),
    ) -> "PlanningMapSnapshot":
        return cls(
            snapshot_id=snapshot_id,
            session_id=session_id,
            source_version=source_version,
            map_frame=map_frame,
            resolution=resolution,
            rows=tuple(rows),
            footprint_radius=footprint_radius,
            required_clearance=required_clearance,
            coverage_status=coverage_status,
            stamp_sim=stamp_sim,
            source_artifact_hash=source_artifact_hash,
            payload_content_hash=payload_content_hash,
            compiler_config_hash=compiler_config_hash,
            circular_obstacles=tuple(circular_obstacles),
        )

    def canonical_payload_hash(self) -> str:
        payload = "|".join(
            (
                self.map_frame,
                f"{self.resolution:.17g}",
                f"{self.footprint_radius:.17g}",
                f"{self.required_clearance:.17g}",
                *(
                    "circle:"
                    f"{obstacle.x:.17g},"
                    f"{obstacle.y:.17g},"
                    f"{obstacle.radius:.17g}"
                    for obstacle in self.circular_obstacles
                ),
                *self.rows,
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, self.width * self.resolution, self.height * self.resolution)

    def _cell_for(self, x: float, y: float) -> Optional[tuple[int, int]]:
        if not _finite_all((x, y)):
            return None
        x_index = int(x // self.resolution)
        y_index = int(y // self.resolution)
        if not (0 <= x_index < self.width and 0 <= y_index < self.height):
            return None
        return x_index, y_index

    def _hard_cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (x, y)
            for y, row in enumerate(self.rows)
            for x, marker in enumerate(row)
            if marker in "#?"
        )

    def _distance_field(self):
        """Lazily built EDT for large maps, memoized by payload hash."""

        key = self.payload_content_hash
        field = _DISTANCE_FIELD_CACHE.get(key)
        if field is None:
            field = _build_distance_field(self.rows)
            _DISTANCE_FIELD_CACHE[key] = field
        return field

    def precompute_clearance(self) -> None:
        """Build the immutable grid clearance lookup before planning starts."""

        if self.width * self.height >= _DISTANCE_FIELD_MIN_CELLS:
            self._distance_field()

    def clearance_at(self, state: VesselState) -> float:
        if not state.is_finite():
            return float("-inf")
        circle_clearance = min(
            (
                hypot(state.x - item.x, state.y - item.y)
                - item.radius
                - self.footprint_radius
                for item in self.circular_obstacles
            ),
            default=float("inf"),
        )
        x_min, y_min, x_max, y_max = self.bounds
        boundary = min(
            state.x - x_min,
            x_max - state.x,
            state.y - y_min,
            y_max - state.y,
        ) - self.footprint_radius
        half_diagonal = self.resolution * sqrt(2.0) / 2.0
        if self.width * self.height >= _DISTANCE_FIELD_MIN_CELLS:
            cell = self._cell_for(state.x, state.y)
            if cell is None:
                return boundary
            cell_x, cell_y = cell
            # Conservative: the cell-centre EDT lookup never overestimates the
            # exact per-point obstacle clearance used by the small-map path.
            obstacle = (
                float(self._distance_field()[cell_y, cell_x]) * self.resolution
                - half_diagonal
                - self.footprint_radius
            )
            grid_clearance = min(boundary, obstacle)
            return min(grid_clearance, circle_clearance)
        obstacle = float("inf")
        for cell_x, cell_y in self._hard_cells():
            center_x = (cell_x + 0.5) * self.resolution
            center_y = (cell_y + 0.5) * self.resolution
            obstacle = min(
                obstacle,
                hypot(state.x - center_x, state.y - center_y)
                - half_diagonal
                - self.footprint_radius,
            )
        grid_clearance = min(boundary, obstacle)
        return min(grid_clearance, circle_clearance)

    def is_state_valid(self, state: VesselState) -> bool:
        if state.frame_id != self.map_frame or state.health != "healthy":
            return False
        return self._cell_for(state.x, state.y) is not None and (
            self.clearance_at(state) > self.required_clearance + 1e-9
        )

    def check_motion(self, states: Sequence[VesselState]) -> MotionCheck:
        if not states:
            return MotionCheck(False, "EMPTY_MOTION", 0.0)
        minimum = float("inf")
        for state in states:
            if not self.is_state_valid(state):
                return MotionCheck(False, "MOTION_COLLISION", max(0.0, minimum))
            minimum = min(minimum, self.clearance_at(state))
        for first, second in zip(states, states[1:]):
            distance = hypot(second.x - first.x, second.y - first.y)
            yaw_change = abs(_angle_difference(second.yaw, first.yaw))
            samples = max(
                1,
                ceil(distance / max(self.resolution * 0.5, 1e-9)),
                ceil(yaw_change / (5.0 * pi / 180.0)),
            )
            for index in range(1, samples + 1):
                fraction = index / samples
                interpolated = replace(
                    first,
                    x=first.x + (second.x - first.x) * fraction,
                    y=first.y + (second.y - first.y) * fraction,
                    yaw=first.yaw + _angle_difference(second.yaw, first.yaw) * fraction,
                    speed=first.speed + (second.speed - first.speed) * fraction,
                    yaw_rate=first.yaw_rate + (second.yaw_rate - first.yaw_rate) * fraction,
                    throttle_state=first.throttle_state
                    + (second.throttle_state - first.throttle_state) * fraction,
                    rudder_state=first.rudder_state
                    + (second.rudder_state - first.rudder_state) * fraction,
                    stamp_sim=first.stamp_sim + (second.stamp_sim - first.stamp_sim) * fraction,
                )
                if not self.is_state_valid(interpolated):
                    return MotionCheck(False, "MOTION_COLLISION", max(0.0, minimum))
                minimum = min(minimum, self.clearance_at(interpolated))
        return MotionCheck(True, "VALID", max(0.0, minimum))


@dataclass(frozen=True)
class PlanningRequest:
    request_id: str
    session_id: str
    start_state: VesselState
    goal_region: GoalRegion
    map_snapshot_id: str
    dynamics_version: str
    cost_config_version: str
    time_budget_ms: float
    seed: int
    mission_index: int = 0
    stamp_sim: float = 0.0
    schema_version: str = "planning-v1"
    mission_version: str = "mission-v1"
    cancelled: bool = False
    route_gate: Optional[tuple[float, float, float]] = None
    continuation_targets: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class CostConfig:
    version: str = "cost-v1"
    w_time: float = 1.0
    w_length: float = 0.1
    w_control: float = 0.01

    def is_valid(self) -> bool:
        return (
            bool(self.version)
            and _finite_all((self.w_time, self.w_length, self.w_control))
            and min(self.w_time, self.w_length, self.w_control) >= 0.0
            and self.w_time + self.w_length + self.w_control > 0.0
        )


@dataclass(frozen=True)
class PlannerConfig:
    max_nodes: int = 1200
    edge_durations: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.5)
    goal_bias: float = 0.2
    global_sample_ratio: float = 0.15
    rewire_radius: float = 2.5
    max_neighbors: int = 32
    connect_tolerance: float = 0.85
    stop_on_first_solution: bool = False
    grid_seed_enabled: bool = True
    max_request_age_s: float = 5.0
    max_map_age_s: float = 5.0
    max_throttle: float = 1.0
    max_abs_rudder: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_durations", tuple(self.edge_durations))

    def is_valid(self) -> bool:
        return (
            isinstance(self.max_nodes, int)
            and not isinstance(self.max_nodes, bool)
            and 0 < self.max_nodes <= MAX_PLANNER_NODES
            and bool(self.edge_durations)
            and _finite_all(self.edge_durations)
            and all(0.0 < duration <= MAX_EDGE_DURATION_S for duration in self.edge_durations)
            and _finite_all(
                (
                    self.goal_bias,
                    self.global_sample_ratio,
                    self.rewire_radius,
                    self.connect_tolerance,
                    self.max_request_age_s,
                    self.max_map_age_s,
                    self.max_throttle,
                    self.max_abs_rudder,
                )
            )
            and 0.0 <= self.goal_bias <= 1.0
            and 0.0 <= self.global_sample_ratio <= 1.0
            and self.rewire_radius > 0.0
            and self.connect_tolerance >= 0.0
            and isinstance(self.max_neighbors, int)
            and 0 < self.max_neighbors <= 256
            and isinstance(self.grid_seed_enabled, bool)
            and self.max_request_age_s >= 0.0
            and self.max_map_age_s >= 0.0
            and 0.0 < self.max_throttle <= 1.0
            and 0.0 < self.max_abs_rudder <= 1.0
        )


@dataclass(frozen=True)
class PrototypeReducedDynamics:
    """Deterministic reduced model calibrated against the live MATLAB plant."""

    version: str = "national-test-live-reduced-dynamics-v5"
    max_speed: float = 1.8
    throttle_speed_gain: float = 6.0
    max_yaw_rate: float = 1.2
    rudder_yaw_rate_gain: float = 2.4
    speed_response: float = 0.8
    yaw_response: float = 4.0
    rudder_yaw_sign: float = -1.0
    rudder_full_authority_speed: float = 0.3
    integration_step_s: float = 0.1
    max_duration_s: float = MAX_EDGE_DURATION_S
    max_integration_steps: int = MAX_PROPAGATION_STEPS

    def __post_init__(self) -> None:
        if not self.version or not _finite_all(
            (
                self.max_speed,
                self.throttle_speed_gain,
                self.max_yaw_rate,
                self.rudder_yaw_rate_gain,
                self.speed_response,
                self.yaw_response,
                self.rudder_yaw_sign,
                self.rudder_full_authority_speed,
                self.integration_step_s,
                self.max_duration_s,
            )
        ):
            raise ValueError("dynamics configuration is invalid")
        if min(
            self.max_speed,
            self.throttle_speed_gain,
            self.max_yaw_rate,
            self.rudder_yaw_rate_gain,
            self.speed_response,
            self.yaw_response,
            self.rudder_full_authority_speed,
            self.integration_step_s,
            self.max_duration_s,
        ) <= 0.0:
            raise ValueError("dynamics configuration must be positive")
        if abs(abs(self.rudder_yaw_sign) - 1.0) > 1e-9:
            raise ValueError("rudder yaw sign must be either -1 or 1")
        if self.rudder_full_authority_speed > self.max_speed:
            raise ValueError("rudder authority speed exceeds maximum speed")
        if not isinstance(self.max_integration_steps, int) or not (
            0 < self.max_integration_steps <= MAX_PROPAGATION_STEPS
        ):
            raise ValueError("dynamics propagation limits exceed the research cap")

    def _validate(self, state: VesselState, control: Control) -> None:
        if not state.is_finite() or state.frame_id != "map":
            raise ValueError("state is invalid")
        if not control.is_valid():
            raise ValueError("control is invalid")
        if not 0.0 <= state.speed <= self.max_speed:
            raise ValueError("speed is outside dynamics bounds")
        if abs(state.yaw_rate) > self.max_yaw_rate + 1e-9:
            raise ValueError("yaw rate is outside dynamics bounds")

    def is_state_valid(self, state: VesselState) -> bool:
        try:
            self._validate(state, Control(0.0, 0.0))
        except (ArithmeticError, TypeError, ValueError):
            return False
        return True

    def propagate(
        self, state: VesselState, control: Control, duration: float
    ) -> tuple[VesselState, ...]:
        self._validate(state, control)
        if not _finite(duration) or duration <= 0.0 or duration > self.max_duration_s:
            raise ValueError("duration must be positive and finite")
        steps = max(1, ceil(duration / self.integration_step_s))
        if steps > self.max_integration_steps:
            raise ValueError("propagation step budget exceeded")
        step = duration / steps
        current = state
        rollout = [state]
        for _ in range(steps):
            target_speed = min(
                self.max_speed,
                max(0.0, control.throttle) * self.throttle_speed_gain,
            )
            speed_alpha = min(1.0, self.speed_response * step)
            yaw_alpha = min(1.0, self.yaw_response * step)
            next_speed = current.speed + (target_speed - current.speed) * speed_alpha
            rudder_authority = _clamp(
                max(current.speed, next_speed)
                / self.rudder_full_authority_speed,
                0.0,
                1.0,
            )
            target_yaw_rate = _clamp(
                self.rudder_yaw_sign
                * control.rudder
                * self.rudder_yaw_rate_gain
                * rudder_authority,
                -self.max_yaw_rate,
                self.max_yaw_rate,
            )
            next_yaw_rate = _clamp(
                current.yaw_rate + (target_yaw_rate - current.yaw_rate) * yaw_alpha,
                -self.max_yaw_rate,
                self.max_yaw_rate,
            )
            next_yaw = current.yaw + next_yaw_rate * step
            next_state = replace(
                current,
                x=current.x + next_speed * cos(next_yaw) * step,
                y=current.y + next_speed * sin(next_yaw) * step,
                yaw=next_yaw,
                speed=_clamp(next_speed, 0.0, self.max_speed),
                yaw_rate=next_yaw_rate,
                throttle_state=control.throttle,
                rudder_state=control.rudder,
                stamp_sim=current.stamp_sim + step,
            )
            if not next_state.is_finite():
                raise ValueError("dynamics produced a non-finite state")
            rollout.append(next_state)
            current = next_state
        return tuple(rollout)


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    request_id: str
    session_id: str
    map_snapshot_id: str
    map_source_version: int
    map_payload_content_hash: str
    dynamics_version: str
    validator_version: str
    frame_id: str
    mission_index: int
    mission_version: str
    map_source_artifact_hash: str
    map_compiler_config_hash: str
    state_version: str
    states: tuple[VesselState, ...]
    controls: tuple[Control, ...]
    durations: tuple[float, ...]
    times: tuple[float, ...]
    edge_rollouts: tuple[tuple[VesselState, ...], ...]
    cost: float
    min_clearance: float
    validation_status: str
    terminal_position_error: float
    terminal_heading_error: float
    terminal_speed: float
    terminal_yaw_rate: float

    def __post_init__(self) -> None:
        for field_name in (
            "states",
            "controls",
            "durations",
            "times",
            "edge_rollouts",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        if not self.states or len(self.states) != len(self.times):
            raise ValueError("trajectory states and times must align")
        if len(self.controls) != len(self.states) - 1:
            raise ValueError("one control is required per trajectory edge")
        if len(self.durations) != len(self.controls) or len(self.edge_rollouts) != len(self.controls):
            raise ValueError("edge controls, durations and rollouts must align")


@dataclass(frozen=True)
class TrajectoryValidation:
    valid: bool
    reason: str
    min_clearance: float = 0.0
    cost: float = 0.0
    position_error: float = float("inf")
    heading_error: float = float("inf")


class TrajectoryValidator:
    version = "trajectory-validator-v3"

    def validate(
        self,
        trajectory: Trajectory,
        request: PlanningRequest,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        cost_config: CostConfig,
        *,
        tolerance: float = 1e-6,
    ) -> TrajectoryValidation:
        if not isinstance(trajectory, Trajectory):
            return TrajectoryValidation(False, "INVALID_TRAJECTORY_TYPE")
        if trajectory.request_id != request.request_id:
            return TrajectoryValidation(False, "REQUEST_ID_MISMATCH")
        if trajectory.session_id != request.session_id or trajectory.session_id != map_snapshot.session_id:
            return TrajectoryValidation(False, "SESSION_MISMATCH")
        if trajectory.map_snapshot_id != map_snapshot.snapshot_id:
            return TrajectoryValidation(False, "MAP_SNAPSHOT_MISMATCH")
        if trajectory.map_source_version != map_snapshot.source_version:
            return TrajectoryValidation(False, "MAP_VERSION_MISMATCH")
        if trajectory.map_payload_content_hash != map_snapshot.payload_content_hash:
            return TrajectoryValidation(False, "MAP_HASH_MISMATCH")
        if trajectory.map_source_artifact_hash != map_snapshot.source_artifact_hash:
            return TrajectoryValidation(False, "MAP_ARTIFACT_HASH_MISMATCH")
        if trajectory.map_compiler_config_hash != map_snapshot.compiler_config_hash:
            return TrajectoryValidation(False, "MAP_COMPILER_CONFIG_MISMATCH")
        if trajectory.frame_id != map_snapshot.map_frame:
            return TrajectoryValidation(False, "FRAME_MISMATCH")
        if trajectory.mission_index != request.mission_index:
            return TrajectoryValidation(False, "MISSION_INDEX_MISMATCH")
        if trajectory.mission_version != request.mission_version:
            return TrajectoryValidation(False, "MISSION_VERSION_MISMATCH")
        if trajectory.state_version != request.start_state.state_version:
            return TrajectoryValidation(False, "STATE_VERSION_MISMATCH")
        if any(
            state.frame_id != trajectory.frame_id or state.state_version != trajectory.state_version
            for state in trajectory.states
        ) or any(
            state.frame_id != trajectory.frame_id or state.state_version != trajectory.state_version
            for rollout in trajectory.edge_rollouts
            for state in rollout
        ):
            return TrajectoryValidation(False, "STATE_METADATA_MISMATCH")
        if trajectory.dynamics_version != dynamics.version or request.dynamics_version != dynamics.version:
            return TrajectoryValidation(False, "DYNAMICS_VERSION_MISMATCH")
        if trajectory.validator_version != self.version:
            return TrajectoryValidation(False, "VALIDATOR_VERSION_MISMATCH")
        if request.cost_config_version != cost_config.version:
            return TrajectoryValidation(False, "COST_VERSION_MISMATCH")
        if not cost_config.is_valid() or not _finite_all(trajectory.times):
            return TrajectoryValidation(False, "INVALID_COST_OR_TIME")
        if abs(trajectory.times[0]) > tolerance:
            return TrajectoryValidation(False, "TIME_ORIGIN_MISMATCH")
        if not _state_close(trajectory.states[0], request.start_state, tolerance):
            return TrajectoryValidation(False, "START_STATE_MISMATCH")
        cumulative = 0.0
        total_length = 0.0
        minimum = float("inf")
        for index, (control, duration, rollout) in enumerate(
            zip(trajectory.controls, trajectory.durations, trajectory.edge_rollouts)
        ):
            if not control.is_valid() or not _finite(duration) or duration <= 0.0:
                return TrajectoryValidation(False, "INVALID_EDGE")
            cumulative += duration
            if abs(trajectory.times[index + 1] - cumulative) > tolerance:
                return TrajectoryValidation(False, "TIME_DURATION_MISMATCH")
            try:
                expected = dynamics.propagate(trajectory.states[index], control, duration)
            except (ArithmeticError, TypeError, ValueError):
                return TrajectoryValidation(False, "DYNAMICS_REPLAY_ERROR")
            if len(expected) != len(rollout) or any(
                not _state_close(first, second, tolerance * 10.0)
                for first, second in zip(expected, rollout)
            ):
                return TrajectoryValidation(False, "EDGE_ROLLOUT_MISMATCH")
            if not _state_close(rollout[0], trajectory.states[index], tolerance * 10.0):
                return TrajectoryValidation(False, "EDGE_START_MISMATCH")
            if not _state_close(rollout[-1], trajectory.states[index + 1], tolerance * 10.0):
                return TrajectoryValidation(False, "EDGE_ENDPOINT_MISMATCH")
            motion = map_snapshot.check_motion(rollout)
            if not motion.valid:
                return TrajectoryValidation(False, motion.reason, motion.min_clearance)
            minimum = min(minimum, motion.min_clearance)
            total_length += sum(
                hypot(second.x - first.x, second.y - first.y)
                for first, second in zip(rollout, rollout[1:])
            )
        recomputed_cost = (
            cost_config.w_time * cumulative
            + cost_config.w_length * total_length
            + cost_config.w_control
            * sum(control.throttle * control.throttle + control.rudder * control.rudder for control in trajectory.controls)
        )
        if not _finite(trajectory.cost) or abs(trajectory.cost - recomputed_cost) > 1e-5:
            return TrajectoryValidation(False, "COST_MISMATCH", minimum, recomputed_cost)
        terminal = trajectory.states[-1]
        if not map_snapshot.is_state_valid(terminal):
            return TrajectoryValidation(False, "TERMINAL_STATE_INVALID", minimum, recomputed_cost)
        if not request.goal_region.contains(terminal):
            return TrajectoryValidation(False, "GOAL_TOLERANCE_NOT_MET", minimum, recomputed_cost)
        position_error = hypot(terminal.x - request.goal_region.x, terminal.y - request.goal_region.y)
        heading_error = (
            0.0
            if request.goal_region.desired_yaw is None
            else abs(_angle_difference(terminal.yaw, request.goal_region.desired_yaw))
        )
        return TrajectoryValidation(
            True,
            "VALID",
            max(0.0, minimum),
            recomputed_cost,
            position_error,
            heading_error,
        )


@dataclass(frozen=True)
class PlanResult:
    status: PlanStatus
    reason: str
    trajectory: Optional[Trajectory]
    elapsed_ms: float
    node_count: int
    propagation_count: int
    rewire_attempts: int
    rewire_successes: int
    sample_counts: dict[str, int]
    seed: int
    best_cost: float


@dataclass
class _Node:
    state: VesselState
    parent: Optional[int]
    control: Optional[Control]
    duration: Optional[float]
    rollout: Optional[tuple[VesselState, ...]]
    cost: float


@dataclass(frozen=True)
class _Connection:
    end_state: VesselState
    control: Control
    duration: float
    rollout: tuple[VesselState, ...]
    edge_cost: float
    min_clearance: float
    score: float


@dataclass(frozen=True)
class SteeringResult:
    """Observable result of one bounded local steering attempt."""

    success: bool
    reason: str
    end_state: Optional[VesselState]
    control: Optional[Control]
    duration: float
    rollout: tuple[VesselState, ...]
    position_error: float
    heading_error: float
    speed_error: float
    min_clearance: float
    attempts: int
    elapsed_ms: float
    constraint_violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class SteeringFeasibilityConfig:
    """PRD 9.6 thresholds for an offline steering feasibility report."""

    minimum_pairs: int = 1000
    success_rate_threshold: float = 0.95
    p95_time_limit_ms: float = 10.0
    position_error_limit: float = 0.25
    heading_error_limit: float = 2.0 * pi / 180.0
    speed_error_limit: float = 0.1
    require_reverse: bool = True

    def is_valid(self) -> bool:
        return (
            isinstance(self.minimum_pairs, int)
            and not isinstance(self.minimum_pairs, bool)
            and self.minimum_pairs > 0
            and _finite_all(
                (
                    self.success_rate_threshold,
                    self.p95_time_limit_ms,
                    self.position_error_limit,
                    self.heading_error_limit,
                    self.speed_error_limit,
                )
            )
            and 0.0 <= self.success_rate_threshold <= 1.0
            and self.p95_time_limit_ms > 0.0
            and self.position_error_limit >= 0.0
            and self.heading_error_limit >= 0.0
            and self.speed_error_limit >= 0.0
            and type(self.require_reverse) is bool
        )


@dataclass(frozen=True)
class SteeringDirectionMetrics:
    """Aggregated metrics for one connector direction."""

    attempted: int
    successes: int
    success_rate: float
    p95_elapsed_ms: float
    max_position_error: float
    max_heading_error: float
    max_speed_error: float
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SteeringFeasibilityReport:
    """Auditable forward/reverse steering gate result."""

    forward: SteeringDirectionMetrics
    reverse: SteeringDirectionMetrics
    minimum_pairs: int
    passed: bool
    reasons: tuple[str, ...]


class KinodynamicInformedRRTStarPlanner:
    """Bounded Kinodynamic Informed RRT* with versioned replay validation."""

    variant = "kinodynamic-informed-rrtstar-v0"

    def __init__(self, config: Optional[PlannerConfig] = None) -> None:
        self.config = config or PlannerConfig()

    def _grid_seed_trajectory(
        self,
        request: PlanningRequest,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        cost_config: CostConfig,
        *,
        deadline: float,
    ) -> Optional[Trajectory]:
        """Build a conservative kinodynamic warm start on the fixed grid.

        The grid search is only a feasibility seed.  Every returned edge is
        generated by the same dynamics model and replayed by the independent
        trajectory validator.  The RRT* loop can then use its finite cost for
        informed sampling and retain the seed as the fail-safe solution.
        """

        if not self.config.grid_seed_enabled:
            return None
        start_cell = map_snapshot._cell_for(
            request.start_state.x,
            request.start_state.y,
        )
        if start_cell is None:
            return None

        resolution = map_snapshot.resolution
        tracking_clearance = map_snapshot.required_clearance
        if (
            request.route_gate is None
            and not request.continuation_targets
        ):
            tracking_clearance += max(0.1, 0.2 * resolution)
        if request.route_gate is None:
            search_goal_x = request.goal_region.x
            search_goal_y = request.goal_region.y
            search_goal_tolerance = request.goal_region.position_tolerance
        else:
            (
                search_goal_x,
                search_goal_y,
                search_goal_tolerance,
            ) = request.route_gate
        if (
            request.route_gate is None
            and not request.continuation_targets
            and request.goal_region.desired_yaw is not None
        ):
            continuation_distance = 5.0
            search_goal_x += continuation_distance * cos(
                request.goal_region.desired_yaw
            )
            search_goal_y += continuation_distance * sin(
                request.goal_region.desired_yaw
            )
            search_goal_tolerance = max(
                resolution,
                1.5 * resolution,
            )
        clearance_cache: dict[tuple[int, int], float] = {}

        def cell_center(cell: tuple[int, int]) -> tuple[float, float]:
            return (
                (cell[0] + 0.5) * resolution,
                (cell[1] + 0.5) * resolution,
            )

        def cell_clearance(cell: tuple[int, int]) -> float:
            cached = clearance_cache.get(cell)
            if cached is not None:
                return cached
            x, y = cell_center(cell)
            value = map_snapshot.clearance_at(
                replace(
                    request.start_state,
                    x=x,
                    y=y,
                    yaw=0.0,
                    speed=0.0,
                    yaw_rate=0.0,
                    throttle_state=0.0,
                    rudder_state=0.0,
                )
            )
            clearance_cache[cell] = value
            return value

        def in_bounds(cell: tuple[int, int]) -> bool:
            return (
                0 <= cell[0] < map_snapshot.width
                and 0 <= cell[1] < map_snapshot.height
            )

        def traversable(cell: tuple[int, int]) -> bool:
            return in_bounds(cell) and (
                cell == start_cell or cell_clearance(cell) > tracking_clearance
            )

        neighbors = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
        def search_cells(
            search_start: tuple[int, int],
            target_x: float,
            target_y: float,
            tolerance: float,
        ) -> Optional[list[tuple[int, int]]]:
            frontier: list[tuple[float, tuple[int, int]]] = [
                (0.0, search_start)
            ]
            costs = {search_start: 0.0}
            parents: dict[tuple[int, int], tuple[int, int]] = {}
            goal_cell: Optional[tuple[int, int]] = None
            while frontier and perf_counter() < deadline:
                _, current = heappop(frontier)
                x, y = cell_center(current)
                if (
                    hypot(x - target_x, y - target_y) <= tolerance
                    and traversable(current)
                ):
                    goal_cell = current
                    break
                for dx, dy in neighbors:
                    candidate = (current[0] + dx, current[1] + dy)
                    if not traversable(candidate):
                        continue
                    if dx and dy:
                        if not traversable(
                            (current[0] + dx, current[1])
                        ):
                            continue
                        if not traversable(
                            (current[0], current[1] + dy)
                        ):
                            continue
                    clearance = cell_clearance(candidate)
                    step = hypot(dx, dy) * resolution
                    clearance_penalty = 0.03 / max(
                        clearance - tracking_clearance,
                        0.02,
                    )
                    new_cost = costs[current] + step + clearance_penalty
                    if (
                        new_cost + 1e-12
                        >= costs.get(candidate, float("inf"))
                    ):
                        continue
                    costs[candidate] = new_cost
                    parents[candidate] = current
                    x, y = cell_center(candidate)
                    heuristic = hypot(x - target_x, y - target_y)
                    heappush(
                        frontier,
                        (new_cost + heuristic, candidate),
                    )
            if goal_cell is None:
                return None
            cells = [goal_cell]
            while cells[-1] != search_start:
                cells.append(parents[cells[-1]])
            cells.reverse()
            return cells

        def search_via_goal_cells() -> Optional[list[tuple[int, int]]]:
            if not request.continuation_targets:
                return None
            directions = (
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
            )
            initial_direction = min(
                range(len(directions)),
                key=lambda index: abs(
                    _angle_difference(
                        atan2(
                            directions[index][1],
                            directions[index][0],
                        ),
                        request.start_state.yaw,
                    )
                ),
            )
            if request.route_gate is None:
                route_x = request.goal_region.x
                route_y = request.goal_region.y
                route_tolerance = max(
                    resolution,
                    request.goal_region.position_tolerance
                    - resolution * 0.5,
                )
            else:
                route_x, route_y, route_tolerance = request.route_gate
            start_progress = int(
                hypot(
                    request.start_state.x - route_x,
                    request.start_state.y - route_y,
                )
                <= route_tolerance
            )
            regions = (
                (
                    route_x,
                    route_y,
                    route_tolerance,
                ),
                *(
                    (
                        x,
                        y,
                        max(
                            resolution,
                            tolerance - resolution * 0.5,
                        ),
                    )
                    for x, y, tolerance
                    in request.continuation_targets
                ),
            )
            start = (start_cell, initial_direction, start_progress)
            frontier: list[
                tuple[
                    float,
                    tuple[tuple[int, int], int, int],
                ]
            ] = [(0.0, start)]
            costs = {start: 0.0}
            parents: dict[
                tuple[tuple[int, int], int, int],
                tuple[tuple[int, int], int, int],
            ] = {}
            terminal = None
            while frontier and perf_counter() < deadline:
                _, current = heappop(frontier)
                cell, direction_index, progress = current
                if progress >= len(regions):
                    terminal = current
                    break
                direction_deltas = (-1, 0, 1)
                if current == start and not any(
                    traversable(
                        (
                            cell[0]
                            + directions[
                                (direction_index + delta)
                                % len(directions)
                            ][0],
                            cell[1]
                            + directions[
                                (direction_index + delta)
                                % len(directions)
                            ][1],
                        )
                    )
                    for delta in direction_deltas
                ):
                    direction_deltas = (-2, -1, 0, 1, 2)
                for direction_delta in direction_deltas:
                    next_direction = (
                        direction_index + direction_delta
                    ) % len(directions)
                    dx, dy = directions[next_direction]
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if not traversable(candidate):
                        continue
                    if dx and dy:
                        if not traversable(
                            (cell[0] + dx, cell[1])
                        ):
                            continue
                        if not traversable(
                            (cell[0], cell[1] + dy)
                        ):
                            continue
                    candidate_x, candidate_y = cell_center(candidate)
                    candidate_progress = progress
                    if candidate_progress < len(regions):
                        region_x, region_y, region_tolerance = regions[
                            candidate_progress
                        ]
                        if (
                            hypot(
                                candidate_x - region_x,
                                candidate_y - region_y,
                            )
                            <= region_tolerance
                        ):
                            candidate_progress += 1
                    candidate_state = (
                        candidate,
                        next_direction,
                        candidate_progress,
                    )
                    clearance = cell_clearance(candidate)
                    step = hypot(dx, dy) * resolution
                    turn_penalty = 0.25 * abs(direction_delta)
                    clearance_penalty = 0.03 / max(
                        clearance - tracking_clearance,
                        0.02,
                    )
                    new_cost = (
                        costs[current]
                        + step
                        + turn_penalty
                        + clearance_penalty
                    )
                    if (
                        new_cost + 1e-12
                        >= costs.get(candidate_state, float("inf"))
                    ):
                        continue
                    costs[candidate_state] = new_cost
                    parents[candidate_state] = current
                    if candidate_progress >= len(regions):
                        target_x = candidate_x
                        target_y = candidate_y
                    else:
                        target_x, target_y, _ = regions[
                            candidate_progress
                        ]
                    heuristic = hypot(
                        candidate_x - target_x,
                        candidate_y - target_y,
                    )
                    heappush(
                        frontier,
                        (new_cost + heuristic, candidate_state),
                    )
            if terminal is None:
                return None
            search_states = [terminal]
            while search_states[-1] != start:
                search_states.append(parents[search_states[-1]])
            search_states.reverse()
            return [state[0] for state in search_states]

        if not request.continuation_targets:
            cells = search_cells(
                start_cell,
                search_goal_x,
                search_goal_y,
                search_goal_tolerance,
            )
            if cells is None:
                return None
            gate_path_index = len(cells) - 1
        else:
            cells = search_via_goal_cells()
            if cells is None:
                return None
            gate_x, gate_y, gate_tolerance = (
                request.route_gate
                if request.route_gate is not None
                else (
                    request.goal_region.x,
                    request.goal_region.y,
                    max(
                        resolution,
                        request.goal_region.position_tolerance
                        - resolution * 0.5,
                    ),
                )
            )
            inside_goal_indices = [
                index
                for index, cell in enumerate(cells)
                if hypot(
                    cell_center(cell)[0] - gate_x,
                    cell_center(cell)[1] - gate_y,
                )
                <= gate_tolerance
            ]
            gate_path_index = inside_goal_indices[0]
            if (
                hypot(
                    request.start_state.x - gate_x,
                    request.start_state.y - gate_y,
                )
                <= gate_tolerance
            ):
                minimum_progress = min(2, len(cells) - 1)
                gate_path_index = next(
                    (
                        index
                        for index in inside_goal_indices
                        if index >= minimum_progress
                    ),
                    inside_goal_indices[-1],
                )
        path = tuple(cell_center(cell) for cell in cells)

        states = [request.start_state]
        controls: list[Control] = []
        durations: list[float] = []
        rollouts: list[tuple[VesselState, ...]] = []
        times = [0.0]
        path_index = 0
        control_period = min(0.1, dynamics.max_duration_s)
        lookahead_distance = max(0.8, 1.6 * resolution)
        max_steps = max(1, int(120.0 / control_period))

        def reached_planning_gate(state: VesselState) -> bool:
            if not request.goal_region.contains(state):
                return False
            if request.route_gate is None:
                return True
            gate_x, gate_y, gate_tolerance = request.route_gate
            return (
                hypot(state.x - gate_x, state.y - gate_y)
                <= gate_tolerance + 1e-9
            )

        # A safe measured state can lie inside the planner's extra tracking
        # margin while facing away from the first high-clearance grid cell.
        # Turn under low propulsion before path tracking: the live vessel has
        # no stationary rudder authority.
        initial_lookahead = 0
        while (
            initial_lookahead + 1 < len(path)
            and hypot(
                request.start_state.x - path[initial_lookahead + 1][0],
                request.start_state.y - path[initial_lookahead + 1][1],
            )
            <= lookahead_distance
        ):
            initial_lookahead += 1
        initial_target = path[initial_lookahead]
        initial_desired_yaw = atan2(
            initial_target[1] - request.start_state.y,
            initial_target[0] - request.start_state.x,
        )
        initial_yaw_error = _angle_difference(
            initial_desired_yaw,
            request.start_state.yaw,
        )
        if abs(initial_yaw_error) > 0.75:
            preferred_direction = 1.0 if initial_yaw_error > 0.0 else -1.0
            recovery = None
            for yaw_direction in (
                preferred_direction,
                -preferred_direction,
            ):
                actuator_direction = (
                    yaw_direction * dynamics.rudder_yaw_sign
                )
                recovery_state = request.start_state
                recovery_states: list[VesselState] = []
                recovery_controls: list[Control] = []
                recovery_rollouts: list[tuple[VesselState, ...]] = []
                valid = True
                for _ in range(160):
                    if perf_counter() >= deadline:
                        return None
                    remaining_yaw_error = _angle_difference(
                        initial_desired_yaw,
                        recovery_state.yaw,
                    )
                    if abs(remaining_yaw_error) <= 0.25:
                        break
                    recovery_control = Control(
                        throttle=0.05,
                        rudder=0.1 * actuator_direction,
                    )
                    try:
                        recovery_rollout = dynamics.propagate(
                            recovery_state,
                            recovery_control,
                            control_period,
                        )
                    except (ArithmeticError, TypeError, ValueError):
                        valid = False
                        break
                    if not map_snapshot.check_motion(
                        recovery_rollout
                    ).valid:
                        valid = False
                        break
                    recovery_controls.append(recovery_control)
                    recovery_rollouts.append(recovery_rollout)
                    recovery_state = recovery_rollout[-1]
                    recovery_states.append(recovery_state)
                else:
                    valid = False
                if valid and (
                    abs(
                        _angle_difference(
                            initial_desired_yaw,
                            recovery_state.yaw,
                        )
                    )
                    <= 0.25
                ):
                    recovery = (
                        recovery_states,
                        recovery_controls,
                        recovery_rollouts,
                    )
                    break
            if recovery is None:
                return None
            recovery_states, recovery_controls, recovery_rollouts = recovery
            for recovery_state, recovery_control, recovery_rollout in zip(
                recovery_states,
                recovery_controls,
                recovery_rollouts,
            ):
                controls.append(recovery_control)
                durations.append(control_period)
                rollouts.append(recovery_rollout)
                states.append(recovery_state)
                times.append(times[-1] + control_period)

        for _ in range(max_steps):
            if perf_counter() >= deadline:
                return None
            current_state = states[-1]
            path_index = min(
                range(path_index, len(path)),
                key=lambda index: hypot(
                    current_state.x - path[index][0],
                    current_state.y - path[index][1],
                ),
            )
            if (
                reached_planning_gate(current_state)
                and (
                    (
                        request.route_gate is None
                        and not request.continuation_targets
                    )
                    or path_index >= gate_path_index
                )
            ):
                break
            lookahead_index = path_index
            while (
                lookahead_index + 1 < len(path)
                and hypot(
                    current_state.x - path[lookahead_index + 1][0],
                    current_state.y - path[lookahead_index + 1][1],
                )
                <= lookahead_distance
            ):
                lookahead_index += 1
            target_x, target_y = path[lookahead_index]
            desired_yaw = atan2(
                target_y - current_state.y,
                target_x - current_state.x,
            )
            goal_distance = hypot(
                current_state.x - request.goal_region.x,
                current_state.y - request.goal_region.y,
            )
            if (
                request.goal_region.desired_yaw is not None
                and goal_distance
                <= request.goal_region.position_tolerance + 2.0
            ):
                desired_yaw = request.goal_region.desired_yaw
            yaw_error = _angle_difference(desired_yaw, current_state.yaw)
            control = Control(
                throttle=(
                    0.02
                    if goal_distance
                    <= request.goal_region.position_tolerance + 1.0
                    else 0.05
                ),
                rudder=_clamp(
                    yaw_error * dynamics.rudder_yaw_sign,
                    -0.1,
                    0.1,
                ),
            )
            try:
                rollout = dynamics.propagate(
                    current_state,
                    control,
                    control_period,
                )
            except (ArithmeticError, TypeError, ValueError):
                return None
            if not map_snapshot.check_motion(rollout).valid:
                return None
            controls.append(control)
            durations.append(control_period)
            rollouts.append(rollout)
            states.append(rollout[-1])
            times.append(times[-1] + control_period)
        else:
            return None

        if not reached_planning_gate(states[-1]):
            return None

        total_length = sum(
            hypot(second.x - first.x, second.y - first.y)
            for rollout in rollouts
            for first, second in zip(rollout, rollout[1:])
        )
        total_cost = (
            cost_config.w_time * times[-1]
            + cost_config.w_length * total_length
            + cost_config.w_control
            * sum(
                control.throttle * control.throttle
                + control.rudder * control.rudder
                for control in controls
            )
        )
        trajectory = Trajectory(
            trajectory_id=f"{request.request_id}-kinodynamic-grid-seed",
            request_id=request.request_id,
            session_id=request.session_id,
            map_snapshot_id=map_snapshot.snapshot_id,
            map_source_version=map_snapshot.source_version,
            map_payload_content_hash=map_snapshot.payload_content_hash,
            dynamics_version=dynamics.version,
            validator_version=TrajectoryValidator.version,
            frame_id=map_snapshot.map_frame,
            mission_index=request.mission_index,
            mission_version=request.mission_version,
            map_source_artifact_hash=map_snapshot.source_artifact_hash,
            map_compiler_config_hash=map_snapshot.compiler_config_hash,
            state_version=request.start_state.state_version,
            states=tuple(states),
            controls=tuple(controls),
            durations=tuple(durations),
            times=tuple(times),
            edge_rollouts=tuple(rollouts),
            cost=total_cost,
            min_clearance=0.0,
            validation_status="UNVALIDATED",
            terminal_position_error=0.0,
            terminal_heading_error=0.0,
            terminal_speed=states[-1].speed,
            terminal_yaw_rate=states[-1].yaw_rate,
        )
        validation = TrajectoryValidator().validate(
            trajectory,
            request,
            map_snapshot,
            dynamics,
            cost_config,
        )
        if not validation.valid:
            return None
        return replace(
            trajectory,
            cost=validation.cost,
            min_clearance=validation.min_clearance,
            validation_status=validation.reason,
            terminal_position_error=validation.position_error,
            terminal_heading_error=validation.heading_error,
        )

    def steer(
        self,
        source: VesselState,
        target: VesselState,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        cost_config: CostConfig,
        *,
        deadline: Optional[float] = None,
    ) -> SteeringResult:
        """Attempt one bounded local dynamic connection and expose its metrics.

        This is a research-facing adapter around the same propagation and
        continuous collision checks used by planning.  It intentionally does
        not claim a two-point boundary-value solver; callers can use the
        returned endpoint errors and timing to decide whether a steering
        feasibility gate is met.
        """

        started = perf_counter()

        def failure(reason: str, attempts: int = 0) -> SteeringResult:
            return SteeringResult(
                success=False,
                reason=reason,
                end_state=None,
                control=None,
                duration=0.0,
                rollout=(),
                position_error=float("inf"),
                heading_error=float("inf"),
                speed_error=float("inf"),
                min_clearance=0.0,
                attempts=attempts,
                elapsed_ms=(perf_counter() - started) * 1000.0,
                constraint_violations=(reason,),
            )

        if (
            not isinstance(source, VesselState)
            or not isinstance(target, VesselState)
            or not isinstance(map_snapshot, PlanningMapSnapshot)
            or not isinstance(dynamics, PrototypeReducedDynamics)
            or not isinstance(cost_config, CostConfig)
        ):
            return failure("INVALID_STEERING_INPUT")
        if not self.config.is_valid() or not cost_config.is_valid():
            return failure("INVALID_STEERING_CONFIG")
        if deadline is not None and not _finite(deadline):
            return failure("INVALID_STEERING_DEADLINE")
        if (
            source.frame_id != map_snapshot.map_frame
            or target.frame_id != map_snapshot.map_frame
            or not source.is_finite()
            or not target.is_finite()
        ):
            return failure("STEERING_FRAME_OR_STATE_INVALID")
        if not dynamics.is_state_valid(source) or not dynamics.is_state_valid(target):
            return failure("STEERING_DYNAMICS_STATE_INVALID")
        if not map_snapshot.is_state_valid(source) or not map_snapshot.is_state_valid(target):
            return failure("STEERING_ENDPOINT_INVALID")
        if deadline is not None and perf_counter() >= deadline:
            return failure("STEERING_DEADLINE_EXPIRED")

        goal = GoalRegion(
            x=target.x,
            y=target.y,
            position_tolerance=self.config.connect_tolerance,
            desired_yaw=target.yaw,
            heading_tolerance=pi,
            speed_limit=dynamics.max_speed,
            yaw_rate_limit=dynamics.max_yaw_rate,
        )
        connection, attempts = self._connect(
            source,
            target,
            goal,
            map_snapshot,
            dynamics,
            cost_config,
            require_endpoint=True,
            deadline=deadline,
            early_exit_limits=(
                min(self.config.connect_tolerance, 0.25),
                2.0 * pi / 180.0,
                0.1,
            ),
        )
        if connection is None:
            reason = (
                "STEERING_DEADLINE_EXPIRED"
                if deadline is not None and perf_counter() >= deadline
                else "STEERING_UNAVAILABLE"
            )
            return failure(reason, attempts)
        end = connection.end_state
        return SteeringResult(
            success=True,
            reason="STEERING_CONNECTED",
            end_state=end,
            control=connection.control,
            duration=connection.duration,
            rollout=connection.rollout,
            position_error=hypot(end.x - target.x, end.y - target.y),
            heading_error=abs(_angle_difference(end.yaw, target.yaw)),
            speed_error=abs(end.speed - target.speed),
            min_clearance=connection.min_clearance,
            attempts=attempts,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            constraint_violations=(),
        )

    def evaluate_steering_feasibility(
        self,
        pairs: Sequence[tuple[VesselState, VesselState]],
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        cost_config: CostConfig,
        *,
        reverse_pairs: Optional[Sequence[tuple[VesselState, VesselState]]] = None,
        config: Optional[SteeringFeasibilityConfig] = None,
    ) -> SteeringFeasibilityReport:
        """Measure connector behavior without silently passing an undersized spike."""

        gate = config or SteeringFeasibilityConfig()
        empty = SteeringDirectionMetrics(0, 0, 0.0, float("inf"), float("inf"), float("inf"), float("inf"), ())
        if not gate.is_valid():
            return SteeringFeasibilityReport(empty, empty, gate.minimum_pairs, False, ("INVALID_FEASIBILITY_CONFIG",))
        try:
            forward_pairs = tuple(pairs)
        except TypeError:
            return SteeringFeasibilityReport(empty, empty, gate.minimum_pairs, False, ("INVALID_FORWARD_PAIRS",))
        if reverse_pairs is None:
            reverse_pairs = tuple((target, source) for source, target in forward_pairs)
        else:
            try:
                reverse_pairs = tuple(reverse_pairs)
            except TypeError:
                return SteeringFeasibilityReport(empty, empty, gate.minimum_pairs, False, ("INVALID_REVERSE_PAIRS",))

        def measure(
            candidates: Sequence[tuple[VesselState, VesselState]],
        ) -> SteeringDirectionMetrics:
            results: list[SteeringResult] = []
            failures: set[str] = set()
            successes = 0
            for pair in candidates:
                try:
                    source, target = pair
                except (TypeError, ValueError):
                    result = SteeringResult(
                        False,
                        "INVALID_STATE_PAIR",
                        None,
                        None,
                        0.0,
                        (),
                        float("inf"),
                        float("inf"),
                        float("inf"),
                        0.0,
                        0,
                        0.0,
                        ("INVALID_STATE_PAIR",),
                    )
                else:
                    result = self.steer(source, target, map_snapshot, dynamics, cost_config)
                results.append(result)
                errors_within_limits = (
                    result.position_error <= gate.position_error_limit
                    and result.heading_error <= gate.heading_error_limit
                    and result.speed_error <= gate.speed_error_limit
                )
                if result.success and errors_within_limits and not result.constraint_violations:
                    successes += 1
                else:
                    failures.add(
                        "TERMINAL_ERROR_LIMIT"
                        if result.success and not errors_within_limits
                        else result.reason
                    )
                    failures.update(result.constraint_violations)
            elapsed = tuple(result.elapsed_ms for result in results)
            finite_position = tuple(result.position_error for result in results if _finite(result.position_error))
            finite_heading = tuple(result.heading_error for result in results if _finite(result.heading_error))
            finite_speed = tuple(result.speed_error for result in results if _finite(result.speed_error))
            attempted = len(results)
            return SteeringDirectionMetrics(
                attempted=attempted,
                successes=successes,
                success_rate=successes / attempted if attempted else 0.0,
                p95_elapsed_ms=_p95(elapsed),
                max_position_error=max(finite_position) if finite_position else float("inf"),
                max_heading_error=max(finite_heading) if finite_heading else float("inf"),
                max_speed_error=max(finite_speed) if finite_speed else float("inf"),
                failure_reasons=tuple(sorted(failures)),
            )

        forward = measure(forward_pairs)
        reverse = measure(reverse_pairs)
        reasons: list[str] = []

        def check_direction(prefix: str, metrics: SteeringDirectionMetrics) -> None:
            if metrics.attempted < gate.minimum_pairs:
                reasons.append(f"INSUFFICIENT_{prefix}_PAIRS")
            if metrics.success_rate < gate.success_rate_threshold:
                reasons.append(f"{prefix}_SUCCESS_RATE_BELOW_GATE")
            if metrics.p95_elapsed_ms > gate.p95_time_limit_ms:
                reasons.append(f"{prefix}_P95_TIME_ABOVE_GATE")
            if metrics.max_position_error > gate.position_error_limit:
                reasons.append(f"{prefix}_POSITION_ERROR_ABOVE_GATE")
            if metrics.max_heading_error > gate.heading_error_limit:
                reasons.append(f"{prefix}_HEADING_ERROR_ABOVE_GATE")
            if metrics.max_speed_error > gate.speed_error_limit:
                reasons.append(f"{prefix}_SPEED_ERROR_ABOVE_GATE")
            if metrics.failure_reasons:
                reasons.append(f"{prefix}_CONSTRAINT_OR_CONNECTION_FAILURE")

        check_direction("FORWARD", forward)
        if gate.require_reverse:
            check_direction("REVERSE", reverse)
        return SteeringFeasibilityReport(
            forward=forward,
            reverse=reverse,
            minimum_pairs=gate.minimum_pairs,
            passed=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def plan(
        self,
        request: PlanningRequest,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        cost_config: CostConfig,
        *,
        now_sim: Optional[float],
    ) -> PlanResult:
        started = perf_counter()
        counts = {"global": 0, "goal": 0, "informed": 0}
        propagation_count = 0
        rewire_attempts = 0
        rewire_successes = 0
        nodes: list[_Node] = []
        best_node: Optional[int] = None
        best_cost = float("inf")
        best_trajectory: Optional[Trajectory] = None
        propagation_diagnostics = {"dynamics_errors": 0, "successful_rollouts": 0}
        seed = getattr(request, "seed", 0)

        def finish(status: PlanStatus, reason: str, trajectory: Optional[Trajectory] = None) -> PlanResult:
            return PlanResult(
                status=status,
                reason=reason,
                trajectory=trajectory,
                elapsed_ms=(perf_counter() - started) * 1000.0,
                node_count=len(nodes),
                propagation_count=propagation_count,
                rewire_attempts=rewire_attempts,
                rewire_successes=rewire_successes,
                sample_counts=dict(counts),
                seed=seed,
                best_cost=best_cost,
            )

        if not isinstance(request, PlanningRequest) or not isinstance(map_snapshot, PlanningMapSnapshot):
            return finish(PlanStatus.INVALID_REQUEST, "INVALID_INPUT_TYPE")
        if not isinstance(dynamics, PrototypeReducedDynamics) or not isinstance(cost_config, CostConfig):
            return finish(PlanStatus.INVALID_REQUEST, "INVALID_INPUT_TYPE")
        if not self.config.is_valid() or not cost_config.is_valid():
            return finish(PlanStatus.INVALID_REQUEST, "INVALID_CONFIG")
        if not _finite(now_sim):
            return finish(PlanStatus.STALE_REQUEST, "CURRENT_TIME_REQUIRED")
        if request.cancelled:
            return finish(PlanStatus.CANCELLED, "REQUEST_CANCELLED")
        if not request.request_id or not request.session_id or request.session_id != map_snapshot.session_id:
            return finish(PlanStatus.STALE_REQUEST, "SESSION_MISMATCH")
        if request.map_snapshot_id != map_snapshot.snapshot_id:
            return finish(PlanStatus.STALE_REQUEST, "MAP_SNAPSHOT_MISMATCH")
        if request.dynamics_version != dynamics.version:
            return finish(PlanStatus.STALE_REQUEST, "DYNAMICS_VERSION_MISMATCH")
        if request.cost_config_version != cost_config.version:
            return finish(PlanStatus.STALE_REQUEST, "COST_VERSION_MISMATCH")
        if not _finite_all((request.stamp_sim, request.time_budget_ms)) or request.time_budget_ms <= 0.0:
            return finish(PlanStatus.INVALID_REQUEST, "INVALID_REQUEST_NUMERIC")
        if (
            request.route_gate is not None
            and (
                not isinstance(request.route_gate, tuple)
                or len(request.route_gate) != 3
                or not _finite_all(request.route_gate)
                or request.route_gate[2] < 0.0
            )
        ):
            return finish(
                PlanStatus.INVALID_REQUEST,
                "INVALID_ROUTE_GATE",
            )
        if (
            not isinstance(request.continuation_targets, tuple)
            or any(
                not isinstance(target, tuple)
                or len(target) != 3
                or not _finite_all(target)
                or target[2] < 0.0
                for target in request.continuation_targets
            )
        ):
            return finish(
                PlanStatus.INVALID_REQUEST,
                "INVALID_CONTINUATION_TARGET",
            )
        if request.start_state.stamp_sim != request.stamp_sim:
            return finish(PlanStatus.STALE_REQUEST, "STATE_TIME_MISMATCH")
        request_age = float(now_sim) - request.stamp_sim
        map_age = float(now_sim) - map_snapshot.stamp_sim
        if request_age < 0.0 or map_age < 0.0:
            return finish(PlanStatus.INVALID_MAP, "CLOCK_INVALID")
        if request_age > self.config.max_request_age_s or request_age * 1000.0 > request.time_budget_ms + 5_000.0:
            return finish(PlanStatus.STALE_REQUEST, "REQUEST_EXPIRED")
        if map_age > self.config.max_map_age_s:
            return finish(PlanStatus.INVALID_MAP, "MAP_EXPIRED")
        if map_snapshot.coverage_status != "complete_prior":
            return finish(PlanStatus.INVALID_MAP, "MAP_COVERAGE_INSUFFICIENT")
        if not request.start_state.is_finite() or not dynamics.is_state_valid(request.start_state):
            return finish(PlanStatus.INVALID_START, "INVALID_START_NUMERIC")
        if not request.goal_region.is_valid():
            return finish(PlanStatus.INVALID_GOAL, "INVALID_GOAL_NUMERIC")
        if not map_snapshot.is_state_valid(request.start_state):
            return finish(PlanStatus.START_OCCUPIED, "START_NOT_VALID")
        goal_yaw = request.goal_region.desired_yaw
        if goal_yaw is None:
            goal_yaw = atan2(request.goal_region.y - request.start_state.y, request.goal_region.x - request.start_state.x)
        goal_state = VesselState(
            x=request.goal_region.x,
            y=request.goal_region.y,
            yaw=goal_yaw,
            speed=min(request.goal_region.speed_limit, dynamics.max_speed),
            yaw_rate=0.0,
            frame_id=map_snapshot.map_frame,
            stamp_sim=request.start_state.stamp_sim,
            state_version=request.start_state.state_version,
        )
        if not map_snapshot.is_state_valid(goal_state):
            # Mission waypoints may sit closer to hard geometry than the
            # configuration-space margin (e.g. a buoy-lined gate).  The goal
            # region is a disk: the centre being invalid does not make the
            # region unreachable.  Relocate the sampler/connector focus to the
            # nearest valid state inside the same tolerance disk; acceptance
            # still requires goal_region.contains(terminal), so the task
            # semantics are unchanged.  Only when the whole disk is blocked is
            # the goal reported occupied.
            goal_state = self._relocate_goal_state(goal_state, request.goal_region, map_snapshot)
            if goal_state is None:
                return finish(PlanStatus.GOAL_OCCUPIED, "GOAL_NOT_VALID")
        if (
            request.goal_region.contains(request.start_state)
            and request.route_gate is None
            and not request.continuation_targets
        ):
            trajectory = self._zero_trajectory(request, map_snapshot, dynamics, cost_config)
            return finish(PlanStatus.SUCCESS, "START_ALREADY_IN_GOAL", trajectory)

        rng = Random(request.seed)
        nodes.append(_Node(request.start_state, None, None, None, None, 0.0))
        deadline = started + min(request.time_budget_ms, MAX_REQUEST_TIME_BUDGET_MS) / 1000.0
        validator = TrajectoryValidator()

        if request.route_gate is not None or request.continuation_targets:
            seed_trajectory = self._grid_seed_trajectory(
                request,
                map_snapshot,
                dynamics,
                cost_config,
                deadline=deadline,
            )
            if seed_trajectory is None:
                return finish(
                    PlanStatus.NO_PATH,
                    "ROUTE_CONTINUATION_NO_VALIDATED_SEED",
                )
            return finish(
                PlanStatus.SUCCESS,
                "VALIDATED_KINODYNAMIC_GRID_SEED",
                seed_trajectory,
            )

        # A goal connection is a standard RRT* completeness aid: try the
        # current tree root deterministically before spending the budget on
        # random samples.  The connector still uses the reduced dynamics,
        # continuous motion check and independent trajectory validator, so an
        # obstructed direct path simply falls through to tree growth.
        direct_connection, used = self._connect(
            request.start_state,
            goal_state,
            request.goal_region,
            map_snapshot,
            dynamics,
            cost_config,
            require_endpoint=True,
            deadline=deadline,
            diagnostics=propagation_diagnostics,
        )
        propagation_count += used
        if direct_connection is not None and request.goal_region.contains(direct_connection.end_state):
            direct_index = len(nodes)
            nodes.append(
                _Node(
                    direct_connection.end_state,
                    0,
                    direct_connection.control,
                    direct_connection.duration,
                    direct_connection.rollout,
                    direct_connection.edge_cost,
                )
            )
            direct_trajectory = self._trajectory_from_node(
                nodes,
                direct_index,
                request,
                map_snapshot,
                dynamics,
                cost_config,
            )
            if direct_trajectory is not None:
                best_cost = direct_trajectory.cost
                best_node = direct_index
                best_trajectory = direct_trajectory
                if self.config.stop_on_first_solution:
                    return finish(PlanStatus.SUCCESS, "VALIDATED_GOAL_CONNECTOR", direct_trajectory)

        if best_trajectory is None:
            seed_trajectory = self._grid_seed_trajectory(
                request,
                map_snapshot,
                dynamics,
                cost_config,
                deadline=deadline,
            )
            if seed_trajectory is not None:
                best_cost = seed_trajectory.cost
                best_node = 0
                best_trajectory = seed_trajectory
                if self.config.stop_on_first_solution:
                    return finish(
                        PlanStatus.SUCCESS,
                        "VALIDATED_KINODYNAMIC_GRID_SEED",
                        seed_trajectory,
                    )

        while len(nodes) < self.config.max_nodes and perf_counter() <= deadline:
            sample, sample_kind = self._sample(
                rng,
                request.start_state,
                goal_state,
                request.goal_region,
                map_snapshot,
                dynamics,
                best_cost,
                cost_config,
            )
            counts[sample_kind] += 1
            neighbor_indices = self._neighbor_indices(nodes, sample)
            if not neighbor_indices:
                continue
            candidate: Optional[tuple[int, _Connection]] = None
            for parent_index in neighbor_indices:
                connection = self._connect(
                    nodes[parent_index].state,
                    sample,
                    request.goal_region,
                    map_snapshot,
                    dynamics,
                    cost_config,
                    deadline=deadline,
                    diagnostics=propagation_diagnostics,
                )
                propagation_count += connection[1]
                if connection[0] is None:
                    continue
                edge = connection[0]
                total_cost = nodes[parent_index].cost + edge.edge_cost
                if candidate is None or total_cost < nodes[candidate[0]].cost + candidate[1].edge_cost:
                    candidate = (parent_index, edge)
            if candidate is None:
                continue
            parent_index, edge = candidate
            if any(self._states_equivalent(node.state, edge.end_state) for node in nodes):
                continue
            new_index = len(nodes)
            nodes.append(
                _Node(
                    edge.end_state,
                    parent_index,
                    edge.control,
                    edge.duration,
                    edge.rollout,
                    nodes[parent_index].cost + edge.edge_cost,
                )
            )
            # RRT* rewiring: every attempted edge is re-propagated and only an
            # endpoint-consistent connector can replace a prior parent.
            for other_index in self._neighbor_indices(nodes[:-1], edge.end_state):
                if other_index == 0 or other_index == parent_index or self._is_ancestor(nodes, other_index, new_index):
                    continue
                rewire_attempts += 1
                connection, used = self._connect(
                    edge.end_state,
                    nodes[other_index].state,
                    request.goal_region,
                    map_snapshot,
                    dynamics,
                    cost_config,
                    require_endpoint=True,
                    require_exact_endpoint=True,
                    deadline=deadline,
                    diagnostics=propagation_diagnostics,
                )
                propagation_count += used
                if connection is None:
                    continue
                proposed_cost = nodes[new_index].cost + connection.edge_cost
                if proposed_cost + 1e-9 >= nodes[other_index].cost:
                    continue
                nodes[other_index].parent = new_index
                nodes[other_index].control = connection.control
                nodes[other_index].duration = connection.duration
                nodes[other_index].rollout = connection.rollout
                old_cost = nodes[other_index].cost
                nodes[other_index].cost = proposed_cost
                self._propagate_descendant_costs(nodes, other_index, proposed_cost - old_cost)
                rewire_successes += 1

            if request.goal_region.contains(edge.end_state):
                trajectory = self._trajectory_from_node(nodes, new_index, request, map_snapshot, dynamics, cost_config)
                if trajectory is not None:
                    if trajectory.cost < best_cost:
                        best_cost = trajectory.cost
                        best_node = new_index
                        best_trajectory = trajectory
                    if self.config.stop_on_first_solution:
                        return finish(PlanStatus.SUCCESS, "VALIDATED_SOLUTION", trajectory)

        if best_node is not None and best_trajectory is not None:
            return finish(
                PlanStatus.TIME_BUDGET_WITH_VALID_SOLUTION,
                "TIME_BUDGET_WITH_VALID_SOLUTION",
                best_trajectory,
            )
        if (
            propagation_diagnostics["dynamics_errors"] > 0
            and propagation_diagnostics["successful_rollouts"] == 0
        ):
            return finish(PlanStatus.DYNAMICS_ERROR, "DYNAMICS_PROPAGATION_FAILED")
        return finish(PlanStatus.TIMEOUT_NO_SOLUTION, "BUDGET_EXHAUSTED_NO_VALIDATED_SOLUTION")

    def _sample(
        self,
        rng: Random,
        start: VesselState,
        goal_state: VesselState,
        goal: GoalRegion,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        best_cost: float,
        cost_config: CostConfig,
    ) -> tuple[VesselState, str]:
        if rng.random() < self.config.goal_bias:
            return goal_state, "goal"
        if isfinite(best_cost) and rng.random() > self.config.global_sample_ratio:
            return self._informed_sample(
                rng,
                start,
                goal_state,
                goal,
                map_snapshot,
                dynamics,
                best_cost,
                cost_config,
            ), "informed"
        x_min, y_min, x_max, y_max = map_snapshot.bounds
        return (
            VesselState(
                x=rng.uniform(x_min, x_max),
                y=rng.uniform(y_min, y_max),
                yaw=rng.uniform(-pi, pi),
                speed=rng.uniform(0.0, dynamics.max_speed),
                yaw_rate=rng.uniform(-dynamics.max_yaw_rate, dynamics.max_yaw_rate),
                frame_id=map_snapshot.map_frame,
                stamp_sim=start.stamp_sim,
                state_version=start.state_version,
            ),
            "global",
        )

    def _informed_sample(
        self,
        rng: Random,
        start: VesselState,
        goal: VesselState,
        goal_region: GoalRegion,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        best_cost: float,
        cost_config: CostConfig,
    ) -> VesselState:
        dx = goal.x - start.x
        dy = goal.y - start.y
        c_min = hypot(dx, dy)
        denominator = max(
            cost_config.w_time / dynamics.max_speed + cost_config.w_length,
            1e-9,
        )
        c_best = max(c_min + 1e-3, best_cost / denominator)
        # The focal point is the goal-region center, not a point goal.  Any
        # improving path ending anywhere in the allowed disk is contained in
        # this center-focal ellipse expanded by the position tolerance.
        c_bound = c_best + goal_region.position_tolerance
        major = c_bound / 2.0
        minor = sqrt(max(c_bound * c_bound - c_min * c_min, 1e-6)) / 2.0
        angle = atan2(dy, dx)
        for _ in range(32):
            radius = sqrt(rng.random())
            theta = rng.uniform(-pi, pi)
            local_x = radius * cos(theta) * major
            local_y = radius * sin(theta) * minor
            x = (start.x + goal.x) / 2.0 + cos(angle) * local_x - sin(angle) * local_y
            y = (start.y + goal.y) / 2.0 + sin(angle) * local_x + cos(angle) * local_y
            if map_snapshot._cell_for(x, y) is not None:
                return VesselState(
                    x=x,
                    y=y,
                    yaw=rng.uniform(-pi, pi),
                    speed=rng.uniform(0.0, dynamics.max_speed),
                    yaw_rate=rng.uniform(-dynamics.max_yaw_rate, dynamics.max_yaw_rate),
                    frame_id=map_snapshot.map_frame,
                    stamp_sim=start.stamp_sim,
                    state_version=start.state_version,
                )
        return goal

    def _neighbor_indices(self, nodes: Sequence[_Node], sample: VesselState) -> list[int]:
        ranked = sorted(
            (
                (hypot(node.state.x - sample.x, node.state.y - sample.y), index)
                for index, node in enumerate(nodes)
            ),
            key=lambda item: item[0],
        )
        selected = [index for distance, index in ranked if distance <= self.config.rewire_radius]
        if not selected and ranked:
            selected = [ranked[0][1]]
        return selected[: self.config.max_neighbors]

    @staticmethod
    def _relocate_goal_state(
        goal_state: VesselState,
        goal_region: GoalRegion,
        map_snapshot: PlanningMapSnapshot,
    ) -> Optional[VesselState]:
        """Nearest valid state inside the goal tolerance disk, or None."""

        step = max(map_snapshot.resolution * 0.5, 1e-3)
        rings = max(1, ceil(goal_region.position_tolerance / step))
        candidates: list[tuple[float, VesselState]] = []
        for ring_x in range(-rings, rings + 1):
            for ring_y in range(-rings, rings + 1):
                x = goal_region.x + ring_x * step
                y = goal_region.y + ring_y * step
                candidate = replace(goal_state, x=x, y=y)
                if not goal_region.contains(candidate):
                    continue
                candidates.append((hypot(x - goal_region.x, y - goal_region.y), candidate))
        candidates.sort(key=lambda item: item[0])
        for _, candidate in candidates:
            if map_snapshot.is_state_valid(candidate):
                return candidate
        return None

    @staticmethod
    def _states_equivalent(first: VesselState, second: VesselState, tolerance: float = 1e-5) -> bool:
        """Return whether two tree nodes represent the same dynamic state.

        Position alone is not enough for a kinodynamic tree: yaw, velocity,
        yaw rate and actuator states change which successors are reachable.
        Absolute simulation time is intentionally not part of this static-map
        duplicate test; a higher-cost revisit to the same physical state cannot
        improve the tree under this P0-16 prototype model.
        """

        return (
            first.frame_id == second.frame_id
            and hypot(first.x - second.x, first.y - second.y) <= tolerance
            and abs(_angle_difference(first.yaw, second.yaw)) <= tolerance
            and abs(first.speed - second.speed) <= tolerance
            and abs(first.yaw_rate - second.yaw_rate) <= tolerance
            and abs(first.throttle_state - second.throttle_state) <= tolerance
            and abs(first.rudder_state - second.rudder_state) <= tolerance
        )

    def _connect(
        self,
        source: VesselState,
        target: VesselState,
        goal: GoalRegion,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        cost_config: CostConfig,
        *,
        require_endpoint: bool = False,
        require_exact_endpoint: bool = False,
        deadline: Optional[float] = None,
        diagnostics: Optional[dict[str, int]] = None,
        early_exit_limits: Optional[tuple[float, float, float]] = None,
    ) -> tuple[Optional[_Connection], int]:
        target_yaw = atan2(target.y - source.y, target.x - source.x) if hypot(target.x - source.x, target.y - source.y) > 1e-9 else source.yaw
        distance = hypot(target.x - source.x, target.y - source.y)
        best: Optional[_Connection] = None
        attempts = 0
        for duration in self.config.edge_durations:
            if deadline is not None and perf_counter() >= deadline:
                break
            guidance_rudder = _clamp(
                dynamics.rudder_yaw_sign
                * _angle_difference(target_yaw, source.yaw)
                / max(
                    dynamics.rudder_yaw_rate_gain * duration,
                    1e-9,
                ),
                -self.config.max_abs_rudder,
                self.config.max_abs_rudder,
            )
            steps = max(
                1,
                ceil(duration / dynamics.integration_step_s),
            )
            step = duration / steps
            response_alpha = min(
                1.0,
                dynamics.speed_response * step,
            )
            decay = 1.0 - response_alpha
            inherited_distance = (
                source.speed
                * step
                * decay
                * (1.0 - decay**steps)
                / max(1.0 - decay, 1e-9)
            )
            target_distance_factor = max(
                duration
                - step
                * decay
                * (1.0 - decay**steps)
                / max(1.0 - decay, 1e-9),
                1e-9,
            )
            target_speed = _clamp(
                (distance - inherited_distance)
                / target_distance_factor
                / dynamics.throttle_speed_gain,
                0.0,
                self.config.max_throttle,
            )
            control_candidates = [
                Control(target_speed, guidance_rudder),
                Control(
                    min(0.1, self.config.max_throttle),
                    guidance_rudder,
                ),
                Control(
                    min(0.25, self.config.max_throttle),
                    guidance_rudder,
                ),
                Control(
                    min(0.5, self.config.max_throttle),
                    guidance_rudder,
                ),
                Control(
                    self.config.max_throttle,
                    guidance_rudder,
                ),
                Control(
                    target_speed,
                    -self.config.max_abs_rudder,
                ),
                Control(
                    target_speed,
                    -min(0.5, self.config.max_abs_rudder),
                ),
                Control(target_speed, 0.0),
                Control(
                    target_speed,
                    min(0.5, self.config.max_abs_rudder),
                ),
                Control(
                    target_speed,
                    self.config.max_abs_rudder,
                ),
            ]
            controls = list(dict.fromkeys(control_candidates))
            for control in controls:
                if deadline is not None and perf_counter() >= deadline:
                    return best, attempts
                attempts += 1
                try:
                    rollout = dynamics.propagate(source, control, duration)
                except (ArithmeticError, TypeError, ValueError):
                    if diagnostics is not None:
                        diagnostics["dynamics_errors"] = diagnostics.get("dynamics_errors", 0) + 1
                    continue
                if diagnostics is not None:
                    diagnostics["successful_rollouts"] = diagnostics.get("successful_rollouts", 0) + 1
                motion = map_snapshot.check_motion(rollout)
                if not motion.valid:
                    continue
                end = rollout[-1]
                endpoint_error = hypot(end.x - target.x, end.y - target.y)
                if require_exact_endpoint:
                    if not _state_close(end, target, 1e-6):
                        continue
                elif require_endpoint and endpoint_error > self.config.connect_tolerance:
                    continue
                length = sum(
                    hypot(second.x - first.x, second.y - first.y)
                    for first, second in zip(rollout, rollout[1:])
                )
                edge_cost = (
                    cost_config.w_time * duration
                    + cost_config.w_length * length
                    + cost_config.w_control * (control.throttle * control.throttle + control.rudder * control.rudder)
                )
                score = (
                    endpoint_error
                    + 0.2 * abs(_angle_difference(end.yaw, target.yaw))
                    + 0.1 * abs(end.speed - target.speed)
                    + 1e-6 * edge_cost
                )
                connection = _Connection(end, control, duration, rollout, edge_cost, motion.min_clearance, score)
                if best is None or connection.score < best.score:
                    best = connection
                if early_exit_limits is not None:
                    position_limit, heading_limit, speed_limit = early_exit_limits
                    if (
                        endpoint_error <= position_limit
                        and abs(_angle_difference(end.yaw, target.yaw)) <= heading_limit
                        and abs(end.speed - target.speed) <= speed_limit
                    ):
                        return connection, attempts
        return best, attempts

    @staticmethod
    def _is_ancestor(nodes: Sequence[_Node], possible_ancestor: int, node_index: int) -> bool:
        current: Optional[int] = node_index
        while current is not None:
            if current == possible_ancestor:
                return True
            current = nodes[current].parent
        return False

    @staticmethod
    def _propagate_descendant_costs(nodes: list[_Node], parent_index: int, delta: float) -> None:
        for index, node in enumerate(nodes):
            if index == parent_index:
                continue
            current = node.parent
            while current is not None:
                if current == parent_index:
                    node.cost += delta
                    break
                current = nodes[current].parent

    def _trajectory_from_node(
        self,
        nodes: Sequence[_Node],
        node_index: int,
        request: PlanningRequest,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        cost_config: CostConfig,
    ) -> Optional[Trajectory]:
        chain: list[_Node] = []
        current: Optional[int] = node_index
        while current is not None:
            chain.append(nodes[current])
            current = nodes[current].parent
        chain.reverse()
        states = tuple(node.state for node in chain)
        controls = tuple(node.control for node in chain[1:] if node.control is not None)
        durations = tuple(node.duration for node in chain[1:] if node.duration is not None)
        rollouts = tuple(node.rollout for node in chain[1:] if node.rollout is not None)
        times = [0.0]
        for duration in durations:
            times.append(times[-1] + duration)
        trajectory = Trajectory(
            trajectory_id=f"{request.request_id}-rrtstar-{node_index}",
            request_id=request.request_id,
            session_id=request.session_id,
            map_snapshot_id=map_snapshot.snapshot_id,
            map_source_version=map_snapshot.source_version,
            map_payload_content_hash=map_snapshot.payload_content_hash,
            dynamics_version=dynamics.version,
            validator_version=TrajectoryValidator.version,
            frame_id=map_snapshot.map_frame,
            mission_index=request.mission_index,
            mission_version=request.mission_version,
            map_source_artifact_hash=map_snapshot.source_artifact_hash,
            map_compiler_config_hash=map_snapshot.compiler_config_hash,
            state_version=request.start_state.state_version,
            states=states,
            controls=controls,
            durations=durations,
            times=tuple(times),
            edge_rollouts=rollouts,
            cost=nodes[node_index].cost,
            min_clearance=0.0,
            validation_status="UNVALIDATED",
            terminal_position_error=0.0,
            terminal_heading_error=0.0,
            terminal_speed=states[-1].speed,
            terminal_yaw_rate=states[-1].yaw_rate,
        )
        validation = TrajectoryValidator().validate(
            trajectory, request, map_snapshot, dynamics, cost_config
        )
        if not validation.valid:
            return None
        return replace(
            trajectory,
            cost=validation.cost,
            min_clearance=validation.min_clearance,
            validation_status=validation.reason,
            terminal_position_error=validation.position_error,
            terminal_heading_error=validation.heading_error,
        )

    @staticmethod
    def _zero_trajectory(
        request: PlanningRequest,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        cost_config: CostConfig,
    ) -> Trajectory:
        state = request.start_state
        return Trajectory(
            trajectory_id=f"{request.request_id}-already-at-goal",
            request_id=request.request_id,
            session_id=request.session_id,
            map_snapshot_id=map_snapshot.snapshot_id,
            map_source_version=map_snapshot.source_version,
            map_payload_content_hash=map_snapshot.payload_content_hash,
            dynamics_version=dynamics.version,
            validator_version=TrajectoryValidator.version,
            frame_id=map_snapshot.map_frame,
            mission_index=request.mission_index,
            mission_version=request.mission_version,
            map_source_artifact_hash=map_snapshot.source_artifact_hash,
            map_compiler_config_hash=map_snapshot.compiler_config_hash,
            state_version=state.state_version,
            states=(state,),
            controls=(),
            durations=(),
            times=(0.0,),
            edge_rollouts=(),
            cost=0.0,
            min_clearance=map_snapshot.clearance_at(state),
            validation_status="VALID",
            terminal_position_error=hypot(state.x - request.goal_region.x, state.y - request.goal_region.y),
            terminal_heading_error=0.0,
            terminal_speed=state.speed,
            terminal_yaw_rate=state.yaw_rate,
        )


__all__ = [
    "Control",
    "CostConfig",
    "GoalRegion",
    "KinodynamicInformedRRTStarPlanner",
    "MAX_EDGE_DURATION_S",
    "MotionCheck",
    "PlanResult",
    "PlanStatus",
    "PlanningMapSnapshot",
    "PlanningRequest",
    "PlannerConfig",
    "PrototypeReducedDynamics",
    "Trajectory",
    "TrajectoryValidation",
    "TrajectoryValidator",
    "VesselState",
]
