# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time

import sys
import os

from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros import GlobalData
from usvlib4ros import USVAutoNavigationService


class USVNavMain:

    @classmethod
    def start(cls,host,port,deviceId, 
              enable_debug: bool = True,
              laser_debug_freq: float = 0.5,
              device_debug_freq: float = 1.0,
              control_debug_freq: float = 2.0):
        globalData = GlobalData().getInstance()
        # 创建控制器，启用调试输出并配置各话题频率
        rosCtrl = Ros2Controller(
            host=host, 
            port=port, 
            deviceId=deviceId, 
            globalData=globalData,
            enable_debug=enable_debug,
            laser_debug_freq=laser_debug_freq,
            device_debug_freq=device_debug_freq,
            control_debug_freq=control_debug_freq
        )
        navigationService = USVAutoNavigationService(rosCtrl=rosCtrl, globalData=globalData)
        navigationService.startService()
        pass
    pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 示例：启用调试输出，配置各传感器打印频率
    # laser_debug_freq: 激光雷达输出频率(Hz)，0.5表示每2秒打印一次
    # device_debug_freq: 设备状态输出频率(Hz)，1.0表示每秒打印一次
    # control_debug_freq: 控制指令输出频率(Hz)，2.0表示每0.5秒打印一次
    
    # USVNavMain.start("121.41.106.238",8236,"27279d7a29f4fa6e95f630f94c76eb12f6fbb230",
    #                  enable_debug=True, laser_debug_freq=0.5, device_debug_freq=1.0, control_debug_freq=2.0)
    
    # USVNavMain.start("192.168.3.35", 9090,"ID_69b7a7d6ddf6c61f0af03f66a93d32a9723b23dd",
    #                  enable_debug=True, laser_debug_freq=0.5, device_debug_freq=1.0, control_debug_freq=2.0)
    
    # USVNavMain.start("192.168.3.138", 9090, "ID_fe7413bbd4e8e630fc1e4cdf2437c7243304c238",
    #                  enable_debug=True, laser_debug_freq=0.5, device_debug_freq=1.0, control_debug_freq=2.0)
    USVNavMain.start("192.168.127.130", 9090, "ID_04eba8142323512eaa7de57cd2d5385b978280d4",
                     enable_debug=True,      # 启用调试打印
                     laser_debug_freq=0.5,   # 激光雷达每2秒打印一次
                     device_debug_freq=1.0,  # 设备状态每秒打印一次
                     control_debug_freq=2.0  # 控制指令每0.5秒打印一次
                     )
    
    print("\n[Main] USV Navigation Service started with debug logging enabled.")
    print("[Main] Press Ctrl+C to stop.\n")
    
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Main] Stopping USV Navigation Service...")
            from usvlib4ros.usvRosUtil.ros_debug_logger import debug_logger
            debug_logger.print_stats()
            break

