"""
ROS2 报文调试打印工具
支持控制船的运动指令、激光雷达等传感器数据的打印输出
可配置输出频率
"""

import time
import threading
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable


class RosDebugLogger:
    """ROS2 报文调试日志记录器"""
    
    def __init__(self, enable_print: bool = True, default_frequency: float = 1.0):
        """
        初始化调试日志记录器
        
        :param enable_print: 是否启用打印输出
        :param default_frequency: 默认输出频率(Hz)
        """
        self.enable_print = enable_print
        self.default_frequency = default_frequency
        self.last_print_time: Dict[str, float] = {}
        self.print_counters: Dict[str, int] = {}
        self.topic_frequencies: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def set_topic_frequency(self, topic_name: str, frequency: float):
        """
        设置指定话题的输出频率
        
        :param topic_name: 话题名称
        :param frequency: 输出频率(Hz)，0表示不打印
        """
        with self._lock:
            self.topic_frequencies[topic_name] = frequency
            
    def enable_topic(self, topic_name: str, frequency: Optional[float] = None):
        """启用指定话题的打印"""
        freq = frequency if frequency is not None else self.default_frequency
        self.set_topic_frequency(topic_name, freq)
        
    def disable_topic(self, topic_name: str):
        """禁用指定话题的打印"""
        self.set_topic_frequency(topic_name, 0)
        
    def _should_print(self, topic_name: str) -> bool:
        """检查是否应该打印该话题"""
        if not self.enable_print:
            return False
            
        with self._lock:
            frequency = self.topic_frequencies.get(topic_name, self.default_frequency)
            if frequency <= 0:
                return False
                
            current_time = time.time()
            last_time = self.last_print_time.get(topic_name, 0)
            interval = 1.0 / frequency
            
            if current_time - last_time >= interval:
                self.last_print_time[topic_name] = current_time
                self.print_counters[topic_name] = self.print_counters.get(topic_name, 0) + 1
                return True
            return False
            
    def _format_msg(self, msg: Any, max_length: int = 500) -> str:
        """格式化消息内容"""
        try:
            if isinstance(msg, dict):
                msg_str = json.dumps(msg, ensure_ascii=False, indent=2)
            else:
                msg_str = str(msg)
                
            if len(msg_str) > max_length:
                msg_str = msg_str[:max_length] + f"... (total {len(msg_str)} chars)"
            return msg_str
        except Exception as e:
            return f"<格式化失败: {e}>"
            
    def log(self, topic_name: str, msg: Any, msg_type: str = "", extra_info: str = ""):
        """
        记录并打印报文
        
        :param topic_name: 话题名称
        :param msg: 消息内容
        :param msg_type: 消息类型
        :param extra_info: 额外信息
        """
        if not self._should_print(topic_name):
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        counter = self.print_counters.get(topic_name, 0)
        
        header = f"\n{'='*80}"
        header += f"\n[ROS DEBUG] {timestamp} | Topic: {topic_name}"
        header += f" | Type: {msg_type}" if msg_type else ""
        header += f" | Count: {counter}"
        header += f" | {extra_info}" if extra_info else ""
        header += f"\n{'-'*80}"
        
        body = self._format_msg(msg)
        
        footer = f"\n{'='*80}"
        
        print(header + "\n" + body + footer)
        
    def log_laser_scan(self, topic_name: str, msg: Dict):
        """专门格式化激光雷达数据"""
        if not self._should_print(topic_name):
            return
            
        try:
            # 提取激光雷达关键信息
            info = {
                "header": msg.get("header", {}),
                "angle_min": msg.get("angle_min"),
                "angle_max": msg.get("angle_max"),
                "angle_increment": msg.get("angle_increment"),
                "range_min": msg.get("range_min"),
                "range_max": msg.get("range_max"),
                "ranges_count": len(msg.get("ranges", [])),
                "intensities_count": len(msg.get("intensities", [])),
                "ranges_sample": msg.get("ranges", [])[:10] if msg.get("ranges") else [],
            }
            self.log(topic_name, info, "sensor_msgs/LaserScan", "LaserScan Data")
        except Exception as e:
            self.log(topic_name, msg, "sensor_msgs/LaserScan", f"Parse Error: {e}")
            
    def log_device_status(self, topic_name: str, msg: Dict):
        """格式化设备状态数据"""
        if not self._should_print(topic_name):
            return
            
        try:
            info = {
                "header": msg.get("header", {}),
                "device_id": msg.get("device_id"),
                "status": msg.get("status"),
                "position": msg.get("position"),
                "velocity": msg.get("velocity"),
                "route_version": msg.get("route_version"),
            }
            self.log(topic_name, info, "message_pkg/DeviceStatus", "Device Status")
        except Exception as e:
            self.log(topic_name, msg, "message_pkg/DeviceStatus", f"Parse Error: {e}")
            
    def log_control_command(self, topic_name: str, msg: Dict):
        """格式化控制指令数据"""
        if not self._should_print(topic_name):
            return
            
        try:
            info = {
                "header": msg.get("header", {}),
                "linear_x": msg.get("linear", {}).get("x") if msg.get("linear") else None,
                "linear_y": msg.get("linear", {}).get("y") if msg.get("linear") else None,
                "linear_z": msg.get("linear", {}).get("z") if msg.get("linear") else None,
                "angular_x": msg.get("angular", {}).get("x") if msg.get("angular") else None,
                "angular_y": msg.get("angular", {}).get("y") if msg.get("angular") else None,
                "angular_z": msg.get("angular", {}).get("z") if msg.get("angular") else None,
            }
            self.log(topic_name, info, "geometry_msgs/Twist", "Control Command")
        except Exception as e:
            self.log(topic_name, msg, "geometry_msgs/Twist", f"Parse Error: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """获取打印统计信息"""
        with self._lock:
            return {
                "total_topics": len(self.print_counters),
                "counters": self.print_counters.copy(),
                "frequencies": self.topic_frequencies.copy(),
            }
            
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"\n{'='*80}")
        print("[ROS Debug Logger Statistics]")
        print(f"{'-'*80}")
        print(f"Total Topics: {stats['total_topics']}")
        print("\nTopic Counters:")
        for topic, count in stats['counters'].items():
            freq = stats['frequencies'].get(topic, self.default_frequency)
            print(f"  {topic}: {count} messages (freq: {freq} Hz)")
        print(f"{'='*80}\n")


# 全局调试日志实例
debug_logger = RosDebugLogger(enable_print=True, default_frequency=1.0)


def create_debug_callback(topic_name: str, msg_type: str = "", 
                         formatter: Optional[Callable] = None) -> Callable:
    """
    创建带调试打印的回调函数
    
    :param topic_name: 话题名称
    :param msg_type: 消息类型
    :param formatter: 自定义格式化函数
    :return: 回调函数
    """
    def callback(msg: Dict):
        if formatter:
            formatter(topic_name, msg)
        else:
            debug_logger.log(topic_name, msg, msg_type)
    return callback


# 便捷配置函数
def configure_debug_output(
    laser_frequency: float = 0.5,
    device_status_frequency: float = 1.0,
    control_frequency: float = 2.0,
    enable_all: bool = True
):
    """
    配置调试输出频率
    
    :param laser_frequency: 激光雷达输出频率(Hz)，0表示禁用
    :param device_status_frequency: 设备状态输出频率(Hz)
    :param control_frequency: 控制指令输出频率(Hz)
    :param enable_all: 是否启用所有输出
    """
    debug_logger.enable_print = enable_all
    
    # 配置各话题频率
    debug_logger.set_topic_frequency("laser", laser_frequency)
    debug_logger.set_topic_frequency("device_status", device_status_frequency)
    debug_logger.set_topic_frequency("control", control_frequency)
    
    print(f"[ROS Debug] Configured frequencies:")
    print(f"  Laser Scan: {laser_frequency} Hz")
    print(f"  Device Status: {device_status_frequency} Hz")
    print(f"  Control Command: {control_frequency} Hz")


if __name__ == "__main__":
    # 测试代码
    configure_debug_output(laser_frequency=1.0, device_status_frequency=2.0)
    
    # 模拟激光雷达数据
    laser_msg = {
        "header": {"seq": 1, "stamp": {"secs": 123, "nsecs": 456}, "frame_id": "laser"},
        "angle_min": -1.57,
        "angle_max": 1.57,
        "angle_increment": 0.01,
        "range_min": 0.1,
        "range_max": 30.0,
        "ranges": [1.0, 1.1, 1.2] * 100,
        "intensities": [100] * 300,
    }
    
    # 模拟控制指令
    control_msg = {
        "linear": {"x": 0.5, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.1},
    }
    
    # 测试打印
    for i in range(5):
        debug_logger.log_laser_scan("/usv/device123/laser/scan", laser_msg)
        debug_logger.log_control_command("/usv/device123/cmd_vel", control_msg)
        time.sleep(0.5)
    
    debug_logger.print_stats()
