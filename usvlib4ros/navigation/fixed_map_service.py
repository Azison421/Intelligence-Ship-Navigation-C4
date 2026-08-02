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
)
from usvlib4ros.usvRosUtil import LogUtil


MAX_EPOCH = 4_000
MAX_STEPS = 3_000
MAX_EPISODE_SECONDS = 300.0
CONTROL_PERIOD_S = 0.1


class FixedMapNavigationService:
    """Keep the sample reset/auto/route UI while replacing its PPO internals."""

    def __init__(
        self,
        ros_ctrl,
        global_data,
        *,
        checkpoint_path: Optional[Path] = None,
        action_bridge=None,
    ) -> None:
        self.ros_ctrl = ros_ctrl
        self.global_data = global_data
        self.action_bridge = action_bridge
        self.checkpoint_path = Path(
            checkpoint_path or DEFAULT_CHECKPOINT
        )

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
        policy = load_live_ready_policy(
            self.checkpoint_path,
            context,
        )
        adapter = LiveInputAdapter(self.global_data, context)
        core = FixedMapControllerCore(context, policy)
        started = time.monotonic()
        last_reason = ""
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
            decision = core.step(sample)
            self._publish_decision(
                decision,
                episode=episode,
                step=step,
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
            if decision.completed:
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
            time.sleep(max(0.0, CONTROL_PERIOD_S - elapsed))
        self._publish_zero(mission_index=core.mission_index)
        return False

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
