import copy
import time

import pytest

from usvlib4ros.navigation.device_action_bridge import (
    DeviceActionBridge,
    create_ros_device_action_bridge,
)


class _Clock:
    def __init__(self, value=10.0):
        self.value = value

    def __call__(self):
        return self.value


def _active_status():
    return {
        "time": 1.0,
        "work_model": 2,
        "pose_model": 2,
        "task_status": 2,
        "route_id": "National_Test",
        "route_version": 1,
        "throttle_percent": 0,
        "rudder_percent": 0,
        "reset_status": 2,
        "reset_request_time": 0.0,
        "parameter_monitor": {
            "version": "",
            "subversion": "",
            "data": [],
        },
        "parameter_adjuster": {
            "version": "",
            "subversion": "",
            "data": [],
        },
    }


def test_active_fresh_command_is_published_without_mutating_status():
    clock = _Clock()
    published = []
    status = _active_status()
    original = copy.deepcopy(status)
    bridge = DeviceActionBridge(
        published.append,
        monotonic=clock,
        wall_clock=lambda: 123.0,
    )

    bridge.on_status(status)
    bridge.set_command(25, -40)
    payload = bridge.publish_once()

    assert payload["throttle_percent"] == 25
    assert payload["rudder_percent"] == -40
    assert payload["time"] == 123.0
    assert published == [payload]
    assert status == original


@pytest.mark.parametrize(
    ("status_change", "elapsed"),
    [
        ({"work_model": 1}, 0.0),
        ({"task_status": 0}, 0.0),
        ({"reset_status": 1}, 0.0),
        ({}, 0.3),
    ],
)
def test_nonoperational_or_stale_command_publishes_zero(
    status_change,
    elapsed,
):
    clock = _Clock()
    published = []
    status = _active_status()
    status.update(status_change)
    bridge = DeviceActionBridge(published.append, monotonic=clock)
    bridge.on_status(status)
    bridge.set_command(60, 20)
    clock.value += elapsed

    payload = bridge.publish_once()

    assert payload["throttle_percent"] == 0
    assert payload["rudder_percent"] == 0


def test_delayed_self_echo_cannot_overwrite_external_stop():
    clock = _Clock()
    wall_clock = _Clock(100.0)
    published = []
    bridge = DeviceActionBridge(
        published.append,
        monotonic=clock,
        wall_clock=wall_clock,
    )
    bridge.on_status(_active_status())
    bridge.set_command(50, 10)
    self_message = bridge.publish_once()
    stopped = _active_status()
    stopped["time"] = 2.0
    stopped["task_status"] = 0

    bridge.on_status(stopped)
    bridge.on_status(self_message)
    wall_clock.value += 1.0
    payload = bridge.publish_once()

    assert payload["task_status"] == 0
    assert payload["throttle_percent"] == 0
    assert payload["rudder_percent"] == 0


def test_incomplete_device_status_is_not_used_as_a_command_template():
    bridge = DeviceActionBridge(lambda payload: None)
    bridge.on_status({"work_model": 2, "task_status": 2})
    bridge.set_command(20, 0)

    with pytest.raises(
        RuntimeError,
        match="authoritative DeviceStatus is unavailable",
    ):
        bridge.publish_once()


def test_stale_authoritative_status_forces_zero_with_fresh_command():
    clock = _Clock()
    bridge = DeviceActionBridge(
        lambda payload: None,
        monotonic=clock,
    )
    bridge.on_status(_active_status())
    clock.value += 1.1
    bridge.set_command(40, 10)

    payload = bridge.publish_once()

    assert payload["throttle_percent"] == 0
    assert payload["rudder_percent"] == 0


def test_background_publisher_expires_command_to_zero():
    published = []
    bridge = DeviceActionBridge(
        published.append,
        command_ttl_s=0.05,
    )
    bridge.on_status(_active_status())
    bridge.set_command(30, -10)

    try:
        bridge.start(publish_hz=100.0)
        deadline = time.monotonic() + 0.5
        while (
            not any(item["throttle_percent"] == 30 for item in published)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        deadline = time.monotonic() + 0.5
        while (
            not published
            or published[-1]["throttle_percent"] != 0
        ) and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        bridge.close()

    assert any(item["throttle_percent"] == 30 for item in published)
    assert published[-1]["throttle_percent"] == 0
    assert published[-1]["rudder_percent"] == 0


def test_ros_adapter_uses_device_status_topic(monkeypatch):
    created = {}

    class _Publisher:
        def __init__(self, topicName, msgType):
            created["publisher"] = (topicName, msgType)

        def publish(self, payload):
            created["payload"] = payload

    class _Subscriber:
        def __init__(self, topicName, msgType, callback):
            created["subscriber"] = (topicName, msgType, callback)

    monkeypatch.setattr(
        "usvlib4ros.usvRosUtil.RosPublisherProxy",
        _Publisher,
    )
    monkeypatch.setattr(
        "usvlib4ros.usvRosUtil.RosSubscriberProxy",
        _Subscriber,
    )

    bridge = create_ros_device_action_bridge("test-device")

    assert created["publisher"][:2] == (
        "usv/device/status/test-device",
        "message_pkg/DeviceStatus",
    )
    assert created["subscriber"][:2] == created["publisher"][:2]
    assert created["subscriber"][2] == bridge.on_status
