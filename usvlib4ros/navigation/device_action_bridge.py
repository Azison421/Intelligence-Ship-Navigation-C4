"""Fail-closed bridge from navigation decisions to Unity actuator input."""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from typing import Callable, Mapping, Optional


_REQUIRED_STATUS_FIELDS = {
    "time",
    "work_model",
    "pose_model",
    "task_status",
    "route_id",
    "route_version",
    "throttle_percent",
    "rudder_percent",
    "reset_status",
    "reset_request_time",
    "parameter_monitor",
    "parameter_adjuster",
}


class DeviceActionBridge:
    """Publish bounded actions using an authoritative DeviceStatus template."""

    def __init__(
        self,
        publish: Callable[[dict], None],
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        command_ttl_s: float = 0.25,
        status_ttl_s: float = 1.0,
    ) -> None:
        self._publish = publish
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._command_ttl_s = command_ttl_s
        self._status_ttl_s = status_ttl_s
        self._status: Optional[dict] = None
        self._status_time = float("-inf")
        self._command = (0, 0)
        self._command_time = float("-inf")
        self._sent_signatures = deque(maxlen=256)
        self._last_message_time = float("-inf")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def on_status(self, status: Mapping[str, object]) -> None:
        received = dict(status)
        if not _REQUIRED_STATUS_FIELDS.issubset(received):
            return
        signature = (
            float(received.get("time", 0.0) or 0.0),
            int(received.get("throttle_percent", 0) or 0),
            int(received.get("rudder_percent", 0) or 0),
        )
        with self._lock:
            try:
                self._sent_signatures.remove(signature)
            except ValueError:
                self._status = copy.deepcopy(received)
                self._status_time = self._monotonic()

    def set_command(self, throttle_percent: float, rudder_percent: float) -> None:
        values = (float(throttle_percent), float(rudder_percent))
        if not all(math.isfinite(value) for value in values):
            with self._lock:
                self._command = (0, 0)
                self._command_time = self._monotonic()
            raise ValueError("device action must be finite")
        if not all(-100.0 <= value <= 100.0 for value in values):
            with self._lock:
                self._command = (0, 0)
                self._command_time = self._monotonic()
            raise ValueError("device action exceeds percentage bounds")
        with self._lock:
            self._command = tuple(int(round(value)) for value in values)
            self._command_time = self._monotonic()

    def publish_once(self) -> dict:
        with self._lock:
            if self._status is None:
                raise RuntimeError(
                    "authoritative DeviceStatus is unavailable"
                )
            payload = copy.deepcopy(self._status)
            command = self._command
            if (
                self._monotonic() - self._command_time
                > self._command_ttl_s
                or self._monotonic() - self._status_time
                > self._status_ttl_s
                or int(payload.get("work_model", 0) or 0) != 2
                or int(payload.get("task_status", 0) or 0) == 0
                or int(payload.get("reset_status", 0) or 0) != 2
            ):
                command = (0, 0)
            message_time = max(
                self._wall_clock(),
                self._last_message_time + 1e-6,
            )
            self._last_message_time = message_time
            payload["time"] = message_time
            payload["throttle_percent"] = command[0]
            payload["rudder_percent"] = command[1]
            self._sent_signatures.append(
                (message_time, command[0], command[1])
            )
        self._publish(payload)
        return payload

    def start(self, *, publish_hz: float = 30.0) -> None:
        if not math.isfinite(publish_hz) or publish_hz <= 0.0:
            raise ValueError("publish_hz must be positive and finite")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(1.0 / publish_hz,),
                daemon=True,
                name="device-action-bridge",
            )
            self._thread.start()

    def _run(self, period_s: float) -> None:
        while not self._stop.is_set():
            try:
                self.publish_once()
            except RuntimeError:
                pass
            if self._stop.wait(period_s):
                break

    def close(self) -> None:
        self.set_command(0, 0)
        for _ in range(3):
            try:
                self.publish_once()
            except RuntimeError:
                break
            time.sleep(0.01)
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)


def create_ros_device_action_bridge(device_id: str) -> DeviceActionBridge:
    """Bind the bridge to the already connected sample ROS client."""
    if not str(device_id):
        raise ValueError("device_id is required")
    from usvlib4ros.usvRosUtil import (
        RosPublisherProxy,
        RosSubscriberProxy,
    )

    topic = f"usv/device/status/{device_id}"
    publisher = RosPublisherProxy(
        topicName=topic,
        msgType="message_pkg/DeviceStatus",
    )
    bridge = DeviceActionBridge(publisher.publish)
    bridge._ros_subscriber = RosSubscriberProxy(
        topicName=topic,
        msgType="message_pkg/DeviceStatus",
        callback=bridge.on_status,
    )
    return bridge


__all__ = [
    "DeviceActionBridge",
    "create_ros_device_action_bridge",
]
