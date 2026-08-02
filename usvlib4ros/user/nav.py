"""Official sample entry name backed by the fixed National_Test algorithm."""

from __future__ import annotations

import threading

from usvlib4ros.navigation.device_action_bridge import (
    create_ros_device_action_bridge,
)
from usvlib4ros.navigation.fixed_map_service import (
    FixedMapNavigationService,
)


class DQN_NAV:
    """Compatibility wrapper retained for ``usvlib4ros.main``."""

    Instance = None

    def __init__(self, ros_ctrl, global_data, xyzAxis=True):
        del xyzAxis
        self.ros_ctrl = ros_ctrl
        self.global_data = global_data
        self.navThread = None
        device_id = getattr(ros_ctrl, "deviceId", None)
        self._action_bridge = (
            create_ros_device_action_bridge(device_id)
            if device_id
            else None
        )
        self._service = FixedMapNavigationService(
            ros_ctrl,
            global_data,
            action_bridge=self._action_bridge,
        )

    def startService(self):
        if self._action_bridge is not None:
            self._action_bridge.start(publish_hz=30.0)
        self.navThread = threading.Thread(target=self.run)
        self.navThread.daemon = True
        self.navThread.start()

    def run(self):
        self._service.run()


__all__ = ["DQN_NAV"]
