"""Official-sample lifecycle wrapped around the fixed-map controller core."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from usvlib4ros.navigation.fixed_map_runtime import (
    DEFAULT_CHECKPOINT,
    FixedMapControllerCore,
    LiveInputAdapter,
    RuntimeDecision,
    build_live_route_context,
    load_live_ready_policy,
    load_offline_ready_policy,
)
from usvlib4ros.usvRosUtil import LogUtil


MAX_EPOCH = 4_000
MAX_STEPS = 3_000
MAX_EPISODE_SECONDS = 300.0
CONTROL_PERIOD_S = 0.1
FAILURE_CONFIRMATION_STEPS = 50
UNSAFE_STOP_REASONS = frozenset(
    {
        "NO_SAFE_ACTION",
        "LATEST_INPUT_UNSAFE",
        "STATE_INVALID",
        "LASER_EMERGENCY_STOP",
    }
)


def advance_failure_streak(
    previous: int,
    reason: str,
    failure_reasons,
) -> int:
    """Count only consecutive failure samples; one recovery clears the streak."""

    return previous + 1 if reason in failure_reasons else 0


def is_collision_evidence(
    reason: str,
    minimum_laser_m,
) -> bool:
    """Confirm impact only when map invalidity and laser evidence agree."""

    return (
        reason == "STATE_INVALID"
        and minimum_laser_m is not None
        and minimum_laser_m <= 0.6
    )


def advance_collision_confirmation(
    previous_laser_streak: int,
    reason: str,
    minimum_laser_m,
) -> tuple[int, bool]:
    """Require direct dual evidence or five seconds of continuous laser stop."""

    laser_streak = (
        previous_laser_streak + 1
        if reason == "LASER_EMERGENCY_STOP"
        else 0
    )
    return (
        laser_streak,
        is_collision_evidence(reason, minimum_laser_m)
        or laser_streak >= FAILURE_CONFIRMATION_STEPS,
    )


class FixedMapNavigationService:
    """Keep the sample reset/auto/route UI while replacing its PPO internals."""

    def __init__(
        self,
        ros_ctrl,
        global_data,
        *,
        checkpoint_path: Optional[Path] = None,
        action_bridge=None,
        allow_offline_candidate: bool = False,
    ) -> None:
        self.ros_ctrl = ros_ctrl
        self.global_data = global_data
        self.action_bridge = action_bridge
        self.allow_offline_candidate = bool(allow_offline_candidate)
        self.checkpoint_path = Path(
            checkpoint_path or DEFAULT_CHECKPOINT
        )
        self.last_episode_metrics = None

    def _publish_zero(
        self,
        *,
        mission_index: int = 0,
        distance_m: float = 0.0,
        heading_deg: float = 0.0,
    ) -> None:
        self.global_data.updateThrottleRudderOutput(
            0,
            0,
            heading_deg,
            mission_index,
            distance_m,
        )
        if self.action_bridge is not None:
            self.action_bridge.set_command(0, 0)

    def _publish_decision(
        self,
        decision: RuntimeDecision,
        *,
        episode: int,
        step: int,
    ) -> None:
        if decision.stop or decision.control is None:
            throttle = 0
            rudder = 0
        else:
            throttle = int(
                max(
                    0,
                    min(100, round(decision.control.throttle * 100.0)),
                )
            )
            rudder = int(
                max(
                    -100,
                    min(100, round(decision.control.rudder * 100.0)),
                )
            )
        self.global_data.updateThrottleRudderOutput(
            throttle,
            rudder,
            decision.advised_heading_deg,
            decision.mission_index,
            decision.distance_to_goal_m,
        )
        if self.action_bridge is not None:
            self.action_bridge.set_command(throttle, rudder)
        score = int(
            round(
                100.0
                * decision.mission_index
                / max(1, 12)
            )
        )
        self.global_data.updateAlgorithmOutput(
            episode,
            step,
            score,
            0.0,
            MAX_EPOCH,
            2,
        )

    def _wait_for_reset(
        self,
        timeout_s: float = 30.0,
        *,
        initial_request_time: Optional[float] = None,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        if initial_request_time is None:
            initial_request_time = float(
                getattr(
                    self.global_data.device_data,
                    "reset_request_time",
                    0.0,
                )
                or 0.0
            )
        else:
            initial_request_time = float(initial_request_time)
        observed_current_request = False
        while time.monotonic() < deadline:
            if int(
                getattr(
                    self.global_data.device_data,
                    "task_status",
                    0,
                )
                or 0
            ) == 0:
                return False
            reset_status = int(
                getattr(
                    self.global_data.device_data,
                    "reset_status",
                    0,
                )
                or 0
            )
            request_time = float(
                getattr(
                    self.global_data.device_data,
                    "reset_request_time",
                    0.0,
                )
                or 0.0
            )
            if (
                reset_status == 1
                or request_time != initial_request_time
            ):
                observed_current_request = True
            if observed_current_request and reset_status == 2:
                return True
            self._publish_zero()
            time.sleep(0.1)
        return False

    def _wait_for_auto(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if int(
                getattr(
                    self.global_data.device_data,
                    "work_model",
                    0,
                )
                or 0
            ) == 2:
                return True
            self._publish_zero()
            time.sleep(0.1)
        return False

    def _wait_for_pose(self, timeout_s: float = 10.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pose = getattr(self.global_data.scada_data, "pose", None)
            lat = float(getattr(pose, "lat", 0.0) or 0.0)
            lng = float(getattr(pose, "lng", 0.0) or 0.0)
            if abs(lat) > 1e-9 and abs(lng) > 1e-9:
                return pose
            self._publish_zero()
            time.sleep(0.1)
        raise TimeoutError("no live ship pose received after Unity reset")

    def _run_episode(
        self,
        route,
        episode: int,
        *,
        max_seconds: float = MAX_EPISODE_SECONDS,
    ) -> bool:
        pose = self._wait_for_pose()
        context = build_live_route_context(
            route,
            pose,
            session_id=f"unity-episode-{episode}-{int(time.time())}",
        )
        loader = (
            load_offline_ready_policy
            if self.allow_offline_candidate
            else load_live_ready_policy
        )
        policy = loader(
            self.checkpoint_path,
            context,
        )
        adapter = LiveInputAdapter(self.global_data, context)
        core = FixedMapControllerCore(context, policy)
        started = time.monotonic()
        last_reason = ""
        points = tuple(
            (
                point[0] - context.compiled_map.manifest.origin_enu[0],
                point[1] - context.compiled_map.manifest.origin_enu[1],
            )
            for point in context.compiled_map.manifest.route_points_enu
        )
        waypoint_min_distances = [float("inf")] * len(points)
        laser_stops = 0
        unsafe_events = 0
        collision_indicators = 0
        replans = 0
        telemetry = []
        collision_streak = 0
        unsafe_streak = 0
        completed = False
        try:
            for step in range(MAX_STEPS):
                if int(
                    getattr(
                        self.global_data.device_data,
                        "task_status",
                        0,
                    )
                    or 0
                ) == 0:
                    print(f"Stop train step {step}...")
                    self._publish_zero(mission_index=core.mission_index)
                    return False
                if time.monotonic() - started > max_seconds:
                    self._publish_zero(mission_index=core.mission_index)
                    return False

                tick_started = time.monotonic()
                sample = adapter.build()
                for index, (x, y) in enumerate(points):
                    waypoint_min_distances[index] = min(
                        waypoint_min_distances[index],
                        (
                            (sample.vessel_state.x - x) ** 2
                            + (sample.vessel_state.y - y) ** 2
                        )
                        ** 0.5,
                    )
                decision = core.step(sample)
                valid_laser = tuple(
                    value
                    for value, valid in zip(
                        sample.laser_ranges,
                        sample.laser_valid_mask,
                    )
                    if valid
                )
                minimum_laser = (
                    min(valid_laser) if valid_laser else None
                )
                laser_stops += int(
                    decision.reason == "LASER_EMERGENCY_STOP"
                )
                (
                    collision_streak,
                    confirmed_collision,
                ) = advance_collision_confirmation(
                    collision_streak,
                    decision.reason,
                    minimum_laser,
                )
                unsafe_streak = advance_failure_streak(
                    unsafe_streak,
                    decision.reason,
                    UNSAFE_STOP_REASONS,
                )
                collision_indicators = max(
                    collision_indicators,
                    int(confirmed_collision),
                )
                replans += int(decision.replanned)
                self._publish_decision(
                    decision,
                    episode=episode,
                    step=step,
                )
                if step % 10 == 0 or decision.reason != last_reason:
                    telemetry.append(
                        {
                            "step": step,
                            "elapsed_s": time.monotonic() - started,
                            "mission_index": decision.mission_index,
                            "maneuver_phase": decision.maneuver_phase,
                            "x_m": sample.vessel_state.x,
                            "y_m": sample.vessel_state.y,
                            "yaw_rad": sample.vessel_state.yaw,
                            "speed_mps": sample.vessel_state.speed,
                            "yaw_rate_rad_s": (
                                sample.vessel_state.yaw_rate
                            ),
                            "distance_to_goal_m": (
                                decision.distance_to_goal_m
                            ),
                            "map_clearance_m": (
                                context.compiled_map.snapshot.clearance_at(
                                    sample.vessel_state
                                )
                            ),
                            "minimum_laser_m": (
                                minimum_laser
                            ),
                            "throttle": (
                                decision.control.throttle
                                if decision.control is not None
                                else 0.0
                            ),
                            "rudder": (
                                decision.control.rudder
                                if decision.control is not None
                                else 0.0
                            ),
                            "reason": decision.reason,
                            "replanned": decision.replanned,
                        }
                    )
                if (
                    step % 20 == 0
                    or decision.replanned
                    or decision.reason != last_reason
                ):
                    print(
                        f"Step: {step}, Action: {decision.action}, "
                        f"Reason: {decision.reason}, "
                        f"Point: {decision.mission_index}, "
                        f"Distance: {decision.distance_to_goal_m:.2f}"
                    )
                last_reason = decision.reason
                if confirmed_collision:
                    unsafe_events = 1
                    last_reason = "CONFIRMED_COLLISION"
                    self._publish_zero(
                        mission_index=decision.mission_index,
                        heading_deg=decision.advised_heading_deg,
                    )
                    return False
                if decision.completed:
                    completed = True
                    print(
                        f"Episode ended at step {step}, "
                        "National_Test route completed"
                    )
                    self._publish_zero(
                        mission_index=decision.mission_index,
                        heading_deg=decision.advised_heading_deg,
                    )
                    return True
                elapsed = time.monotonic() - tick_started
                time.sleep(
                    max(0.0, CONTROL_PERIOD_S - elapsed)
                )
            self._publish_zero(mission_index=core.mission_index)
            return False
        finally:
            self.last_episode_metrics = {
                "duration_s": time.monotonic() - started,
                "completed_waypoints": (
                    13 if completed else core.mission_index
                ),
                "waypoint_min_distances_m": waypoint_min_distances,
                "collisions": collision_indicators,
                "laser_emergency_stops": laser_stops,
                "unrecovered_unsafe_events": max(
                    unsafe_events,
                    int(unsafe_streak > 0 and not completed),
                ),
                "replans": replans,
                "final_mission_index": core.mission_index,
                "stop_reason": last_reason,
                "telemetry": telemetry,
            }

    def run(self) -> None:
        try:
            self.ros_ctrl.initParameterList()
        except Exception as exc:
            LogUtil.error(exc)
        while True:
            try:
                print("wait train button trigger ...")
                while int(
                    getattr(
                        self.global_data.device_data,
                        "task_status",
                        0,
                    )
                    or 0
                ) == 0:
                    self._publish_zero()
                    time.sleep(1.0)

                for episode in range(MAX_EPOCH):
                    if int(
                        getattr(
                            self.global_data.device_data,
                            "task_status",
                            0,
                        )
                        or 0
                    ) == 0:
                        print("Stop train ...")
                        break
                    print(f"train {episode} ...")
                    print("Reset unity ...")
                    reset_request_time = float(
                        getattr(
                            self.global_data.device_data,
                            "reset_request_time",
                            0.0,
                        )
                        or 0.0
                    )
                    if not self.ros_ctrl.reset_unity():
                        raise RuntimeError("Unity reset request failed")
                    if not self._wait_for_reset(
                        initial_request_time=reset_request_time,
                    ):
                        raise TimeoutError("Unity reset did not complete")
                    if not self.ros_ctrl.set_auto_work():
                        raise RuntimeError("automatic work-mode request failed")
                    if not self._wait_for_auto():
                        raise TimeoutError(
                            "device did not enter automatic work mode"
                        )
                    route = self.ros_ctrl.getRoute()
                    if not getattr(route, "points", None):
                        LogUtil.info("Error : len(route.points) is 0. ")
                        self._publish_zero()
                        time.sleep(1.0)
                        continue
                    print(f"Route {route}...")
                    self.global_data.route = route
                    self._run_episode(route, episode)
            except Exception as exc:
                self._publish_zero()
                LogUtil.error(exc)
            finally:
                self._publish_zero()
                time.sleep(0.02)


__all__ = ["FixedMapNavigationService"]
