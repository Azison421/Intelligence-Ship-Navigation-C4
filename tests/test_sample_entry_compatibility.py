import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace

from usvlib4ros.navigation.fixed_map_runtime import RuntimeDecision
from usvlib4ros.navigation.fixed_map_service import (
    FixedMapNavigationService,
)
from usvlib4ros.planning import Control
from usvlib4ros.user.nav import DQN_NAV


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MAIN_SHA256 = (
    "4e79044277c7baed1a00b3bc000e415a6405f2c8c3e80c00e62188d22fff0848"
)


class _OutputCapture:
    def __init__(self):
        self.throttle = None
        self.throttle_updates = 0
        self.algorithm = None
        self.device_data = SimpleNamespace()
        self.scada_data = SimpleNamespace()

    def updateThrottleRudderOutput(self, *values):
        self.throttle = values
        self.throttle_updates += 1

    def updateAlgorithmOutput(self, *values):
        self.algorithm = values


class _ActionBridgeCapture:
    def __init__(self):
        self.command = None

    def set_command(self, throttle, rudder):
        self.command = (throttle, rudder)


def test_official_main_entry_remains_byte_identical():
    digest = hashlib.sha256(
        (PROJECT_ROOT / "usvlib4ros" / "main.py").read_bytes()
    ).hexdigest()

    assert digest == OFFICIAL_MAIN_SHA256


def test_dqn_nav_remains_the_thin_threaded_sample_entry():
    navigation = DQN_NAV(object(), _OutputCapture())
    called = threading.Event()
    navigation._service.run = called.set

    navigation.startService()
    navigation.navThread.join(timeout=1.0)

    assert called.is_set()
    assert navigation.navThread.daemon


def test_dqn_nav_wires_and_starts_live_device_bridge(monkeypatch):
    action_bridge = SimpleNamespace(
        started=False,
        start=lambda **kwargs: setattr(
            action_bridge,
            "started",
            kwargs,
        ),
    )
    monkeypatch.setattr(
        "usvlib4ros.user.nav.create_ros_device_action_bridge",
        lambda device_id: action_bridge,
    )
    navigation = DQN_NAV(
        SimpleNamespace(deviceId="test-device"),
        _OutputCapture(),
    )
    called = threading.Event()
    navigation._service.run = called.set

    navigation.startService()
    navigation.navThread.join(timeout=1.0)

    assert navigation._service.action_bridge is action_bridge
    assert action_bridge.started == {"publish_hz": 30.0}
    assert called.is_set()


def test_sample_output_fields_receive_bounded_runtime_decision():
    output = _OutputCapture()
    action_bridge = _ActionBridgeCapture()
    service = FixedMapNavigationService(
        object(),
        output,
        action_bridge=action_bridge,
    )
    decision = RuntimeDecision(
        reason="POLICY_ACTION_SAFE",
        control=Control(throttle=0.25, rudder=-1.0),
        action=4,
        mission_index=3,
        distance_to_goal_m=4.5,
        advised_heading_deg=-30.0,
        safe_mask=(True,) * 5,
        completed=False,
        replanned=False,
    )

    service._publish_decision(decision, episode=2, step=7)

    assert output.throttle == (25, -100, -30.0, 3, 4.5)
    assert output.algorithm == (2, 7, 25, 0.0, 4_000, 2)
    assert action_bridge.command == (25, -100)


def test_reset_wait_requires_current_request_transition(monkeypatch):
    output = _OutputCapture()
    output.device_data = SimpleNamespace(
        task_status=2,
        reset_status=2,
    )
    transitions = iter((1, 2))

    def advance_reset(_duration):
        output.device_data.reset_status = next(transitions)

    monkeypatch.setattr(
        "usvlib4ros.navigation.fixed_map_service.time.sleep",
        advance_reset,
    )
    service = FixedMapNavigationService(object(), output)

    completed = service._wait_for_reset(timeout_s=1.0)

    assert completed
    assert output.throttle_updates == 2


def test_reset_wait_accepts_fast_completion_after_known_baseline():
    output = _OutputCapture()
    output.device_data = SimpleNamespace(
        task_status=2,
        reset_status=2,
        reset_request_time=11.0,
    )
    service = FixedMapNavigationService(object(), output)

    completed = service._wait_for_reset(
        timeout_s=0.01,
        initial_request_time=10.0,
    )

    assert completed
