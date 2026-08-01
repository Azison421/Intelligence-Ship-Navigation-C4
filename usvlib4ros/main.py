# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time
import sys
import os
import json

# 将父目录添加到 sys.path，确保能正确导入 usvlib4ros 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros import GlobalData
from usvlib4ros import USVAutoNavigationService
from usvlib4ros.user.nav import DQN_NAV


def load_config():
    """加载配置文件，优先从同级目录 config.json，其次从项目根目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths = [
        os.path.join(script_dir, 'config.json'),                      # usvlib4ros/config.json
        os.path.join(os.path.dirname(script_dir), 'config.json'),     # NavAlg/config.json
    ]
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(f"配置文件未找到，已尝试路径: {config_paths}")


class USVNavMain:

    @classmethod
    def start(cls,host,port,deviceId):
        globalData = GlobalData().getInstance()
        rosCtrl = Ros2Controller(host=host, port=port, deviceId=deviceId, globalData=globalData)
        # navigationService = USVAutoNavigationService(rosCtrl=rosCtrl, globalData=globalData)
        # navigationService.startService()
        nav = DQN_NAV(ros_ctrl=rosCtrl,global_data=globalData)
        nav.startService()
        pass
    pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config = load_config()
    ros2_cfg = config['ros2']
    USVNavMain.start(ros2_cfg['host'], ros2_cfg['port'], ros2_cfg['deviceId'])

    while True:
        time.sleep(1)

