"""Deterministic forward-only motion-primitive seed for Kinodynamic RRT*."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import atan2, hypot, pi
from time import perf_counter
from typing import Optional, Sequence

from .kinodynamic_informed_rrtstar import (
    Control,
    GoalRegion,
    PlanningMapSnapshot,
    PlanningRequest,
    PrototypeReducedDynamics,
    VesselState,
)


@dataclass(frozen=True)
class ForwardLatticeConfig:
    position_resolution_m: float = 0.2
    heading_resolution_rad: float = pi / 12.0
    primitive_durations_s: tuple[float, ...] = (0.4, 0.8)
    max_expansions: int = 50_000


@dataclass(frozen=True)
class ForwardLatticeSeed:
    states: tuple[VesselState, ...]
    controls: tuple[Control, ...]
    durations: tuple[float, ...]
    edge_rollouts: tuple[tuple[VesselState, ...], ...]


@dataclass(frozen=True)
class _Record:
    state: VesselState
    parent: Optional[tuple[int, int, int, int]]
    control: Optional[Control]
    duration: float
    rollout: tuple[VesselState, ...]
    cost: float


class ForwardStateLatticePlanner:
    """A* over calibrated forward primitives; output is a feasibility seed only."""

    version = "forward-state-lattice-v1"

    def __init__(self, config: ForwardLatticeConfig | None = None) -> None:
        self.config = config or ForwardLatticeConfig()
        if (
            self.config.position_resolution_m <= 0.0
            or self.config.heading_resolution_rad <= 0.0
            or not self.config.primitive_durations_s
            or any(value <= 0.0 for value in self.config.primitive_durations_s)
            or self.config.max_expansions <= 0
        ):
            raise ValueError("forward lattice configuration is invalid")

    def _key(
        self,
        state: VesselState,
        visit_stage: int,
    ) -> tuple[int, int, int, int]:
        heading_bins = max(1, round(2.0 * pi / self.config.heading_resolution_rad))
        return (
            round(state.x / self.config.position_resolution_m),
            round(state.y / self.config.position_resolution_m),
            round((state.yaw % (2.0 * pi)) / self.config.heading_resolution_rad)
            % heading_bins,
            visit_stage,
        )

    @staticmethod
    def _advance_stage(
        stage: int,
        regions: Sequence[GoalRegion],
        rollout: Sequence[VesselState],
    ) -> int:
        current = stage
        for state in rollout:
            while current < len(regions) and regions[current].contains(state):
                current += 1
        return current

    @staticmethod
    def _target(
        request: PlanningRequest,
        stage: int,
    ) -> GoalRegion:
        if stage < len(request.required_visit_regions):
            return request.required_visit_regions[stage]
        return request.goal_region

    @staticmethod
    def _heuristic(
        request: PlanningRequest,
        state: VesselState,
        stage: int,
        maximum_speed: float,
        maximum_yaw_rate: float,
    ) -> float:
        remaining = (
            *request.required_visit_regions[stage:],
            request.goal_region,
        )
        distance = 0.0
        x, y = state.x, state.y
        for region in remaining:
            distance += max(
                0.0,
                hypot(x - region.x, y - region.y)
                - region.position_tolerance,
            )
            x, y = region.x, region.y
        target = ForwardStateLatticePlanner._target(request, stage)
        desired_yaw = atan2(target.y - state.y, target.x - state.x)
        heading_error = abs(
            (desired_yaw - state.yaw + pi) % (2.0 * pi) - pi
        )
        heading_cost = (
            0.0
            if maximum_yaw_rate <= 1e-12
            else heading_error / maximum_yaw_rate
        )
        return distance / maximum_speed + heading_cost

    def plan(
        self,
        request: PlanningRequest,
        map_snapshot: PlanningMapSnapshot,
        dynamics: PrototypeReducedDynamics,
        action_controls: Sequence[Control],
        *,
        deadline: float,
    ) -> ForwardLatticeSeed | None:
        controls = tuple(action_controls)
        if (
            len(controls) != 5
            or any(not control.is_valid() or control.throttle < 0.0 for control in controls)
        ):
            raise ValueError("forward lattice requires five non-negative controls")
        achievable_speed = min(
            dynamics.max_speed,
            max(control.throttle for control in controls)
            * dynamics.throttle_speed_gain,
        )
        achievable_yaw_rate = min(
            dynamics.max_yaw_rate,
            max(
                abs(control.rudder)
                * (
                    dynamics.positive_rudder_yaw_rate_gain
                    if control.rudder > 0.0
                    else dynamics.negative_rudder_yaw_rate_gain
                    if control.rudder < 0.0
                    else dynamics.rudder_yaw_rate_gain
                )
                for control in controls
            ),
        )

        start_stage = self._advance_stage(
            0,
            request.required_visit_regions,
            (request.start_state,),
        )
        start_key = self._key(request.start_state, start_stage)
        records = {
            start_key: _Record(
                state=request.start_state,
                parent=None,
                control=None,
                duration=0.0,
                rollout=(),
                cost=0.0,
            )
        }
        serial = count()
        frontier = [
            (
                2.0
                * self._heuristic(
                    request,
                    request.start_state,
                    start_stage,
                    achievable_speed,
                    achievable_yaw_rate,
                ),
                next(serial),
                start_key,
                0.0,
            )
        ]
        expansions = 0
        action_order = (2, 1, 3, 0, 4)
        goal_key: tuple[int, int, int, int] | None = None

        while (
            frontier
            and expansions < self.config.max_expansions
            and perf_counter() < deadline
        ):
            _, _, current_key, queued_cost = heappop(frontier)
            current = records[current_key]
            if queued_cost > current.cost + 1e-9:
                continue
            stage = current_key[3]
            if (
                stage == len(request.required_visit_regions)
                and request.goal_region.contains(current.state)
            ):
                goal_key = current_key
                break
            expansions += 1
            for duration in self.config.primitive_durations_s:
                for action_index in action_order:
                    control = controls[action_index]
                    try:
                        rollout = dynamics.propagate(
                            current.state,
                            control,
                            duration,
                        )
                    except (ArithmeticError, TypeError, ValueError):
                        continue
                    if not map_snapshot.check_motion(rollout).valid:
                        continue
                    next_stage = self._advance_stage(
                        stage,
                        request.required_visit_regions,
                        rollout[1:],
                    )
                    next_state = rollout[-1]
                    next_key = self._key(next_state, next_stage)
                    edge_length = sum(
                        hypot(second.x - first.x, second.y - first.y)
                        for first, second in zip(rollout, rollout[1:])
                    )
                    new_cost = (
                        current.cost
                        + duration
                        + 0.1 * edge_length
                        + 0.01 * control.rudder * control.rudder
                    )
                    previous = records.get(next_key)
                    if previous is not None and new_cost + 1e-9 >= previous.cost:
                        continue
                    records[next_key] = _Record(
                        state=next_state,
                        parent=current_key,
                        control=control,
                        duration=duration,
                        rollout=rollout,
                        cost=new_cost,
                    )
                    heuristic = self._heuristic(
                        request,
                        next_state,
                        next_stage,
                        achievable_speed,
                        achievable_yaw_rate,
                    )
                    heappush(
                        frontier,
                        (
                            new_cost + 2.0 * heuristic,
                            next(serial),
                            next_key,
                            new_cost,
                        ),
                    )

        if goal_key is None:
            return None

        chain: list[_Record] = []
        current_key: tuple[int, int, int, int] | None = goal_key
        while current_key is not None:
            record = records[current_key]
            chain.append(record)
            current_key = record.parent
        chain.reverse()
        return ForwardLatticeSeed(
            states=tuple(record.state for record in chain),
            controls=tuple(
                record.control for record in chain[1:] if record.control is not None
            ),
            durations=tuple(record.duration for record in chain[1:]),
            edge_rollouts=tuple(record.rollout for record in chain[1:]),
        )


__all__ = [
    "ForwardLatticeConfig",
    "ForwardLatticeSeed",
    "ForwardStateLatticePlanner",
]
