"""
USV PPO Navigation Controller.
使用PPO算法实现无人船自主导航避障。

install pytorch: pip install torch torchvision torchaudio
"""
import math
import time
import threading
from dataclasses import dataclass

import numpy as np
import torch

from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.navigation.route_plan_service import RoutePlanService
from usvlib4ros.msg.global_data import GlobalData, DictToObject, Point, Constants
from usvlib4ros.msg.parameter import Parameter
from usvlib4ros.usvRosUtil import LogUtil
from usvlib4ros.user.PP0_1 import PPO, device

# ==================== 超参数配置 ====================
N_ACTIONS = 5          # 离散动作空间: +100°, -100°, +50°, -50°, 0°
N_STATES = 96          # 状态维度: 前方180°激光点数(约92) + heading + distance + obstacle_min_range + obstacle_angle
# 注意：若激光雷达总点数非184，需按 total//2 + 4 调整此值
MEMORY_CAPACITY = 2000
BATCH_SIZE = 128
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
GAMMA = 0.99           # 折扣因子
K_EPOCHS = 80          # PPO更新轮数
EPS_CLIP = 0.2         # PPO裁剪系数
ACTION_STD_INIT = 0.6  # 连续动作标准差初始化（当前未启用连续空间）
MAX_EPOCH = 4000       # 最大训练轮数
MAX_STEP_PER_EPISODE = 3000   # 每轮最大步数
MAX_EPISODE_TIME = 300  # 每轮最大时间(秒)
UPDATE_INTERVAL = 2000  # PPO更新间隔(步数)
CHECKPOINT_INTERVAL = 100  # 模型保存间隔(轮数)

# ==================== 导航常量 ====================
LASER_MAX_RANGE = 5.0        # 激光雷达有效最大距离(m)
LASER_FRONT_HALF_DEG = 180   # 前方扫描扇区角度(度)，仅用于碰撞检测
COLLISION_DISTANCE = 0.6     # 碰撞判定阈值(m)
ARRIVE_DISTANCE = 1.0        # 到达目标判定阈值(m)
DEFAULT_SPEED = 0.5          # 默认速度(m/s)
OBSTACLE_SLOW_RANGE = 4.0    # 进入此范围开始减速(m)
TARGET_SLOW_RANGE = 3.0      # 接近目标时减速阈值(m)
ANGULAR_VELOCITY_MAX = 100   # 最大角速度(°/s)
ACTION_TO_DEGREE_SCALE = 1   # 动作到转向角度的缩放因子

# ==================== 奖励权重 ====================
REWARD_ARRIVE_BONUS = 1000      # 到达奖励
REWARD_COLLISION_PENALTY = -500 # 碰撞惩罚
REWARD_OBSTACLE_PENALTY = -5    # 接近障碍物惩罚
REWARD_NEAR_TARGET_BONUS = 1    # 靠近目标奖励
REWARD_WEIGHT_DISTANCE = 0.6    # 距离奖励权重
REWARD_WEIGHT_OBSTACLE = 0.2    # 障碍物惩罚权重
REWARD_WEIGHT_HEADING = 0.2     # 航向奖励权重


@dataclass
class StepResult:
    """step函数返回值，替代原来的元组返回。"""
    next_state: np.ndarray
    reward: float
    current_distance: float
    advise_speed: float
    advise_rotate: float
    advised_heading: float


class PPONav:
    """基于PPO算法的USV导航控制器。"""

    Instance = None

    def startService(self):
        self.navThread = threading.Thread(target=self.run)
        self.navThread.setDaemon(True)
        self.navThread.start()

    def __init__(self, ros_ctrl: Ros2Controller, global_data: GlobalData, xyzAxis: bool = True):
        self.ros_ctrl: Ros2Controller = ros_ctrl
        self.global_data: GlobalData = global_data
        self.navThread = None

        # PPO智能体
        self.ppo_agent = PPO(
            N_STATES, N_ACTIONS, LR_ACTOR, LR_CRITIC,
            GAMMA, K_EPOCHS, EPS_CLIP, False, ACTION_STD_INIT
        )
        self.next_state = None
        self.action_size = N_ACTIONS  # 动作空间大小（用于角度映射）

        # 航线相关
        self.route = None
        self.destPoint: Point = Point()
        self.destPointIndex = -1
        self.prevPoint = Point()
        self.prevPointIndex = -1

        # 训练状态
        self.max_distance = 0.0
        self.score = 0
        self.episode_reward_sum = 0.0
        self.arrive = False
        self.done = False
        self.arrive_distance = ARRIVE_DISTANCE

        self.routePlaneService = RoutePlanService(
            wayPointRadius=self.arrive_distance, route=self.route
        )
        self.last_laser_scan = global_data.laser_data

    # ==================== 训练主循环 ====================

    def run(self):
        """训练与导航主循环。"""
        self.registerParameter()

        while True:
            try:
                LogUtil.info("等待训练触发...")
                while self.global_data.device_data.task_status == 0:
                    time.sleep(1)

                for epoch in range(MAX_EPOCH):
                    if self.global_data.device_data.task_status == 0:
                        LogUtil.info("停止训练")
                        break

                    LogUtil.info(f"第 {epoch} 轮训练开始")

                    # 复位Unity环境
                    self.ros_ctrl.reset_unity()
                    time.sleep(1)
                    while self.global_data.device_data.reset_status != 2:
                        time.sleep(1)

                    self.ros_ctrl.set_auto_work()

                    # 加载航线
                    self.route = self.ros_ctrl.getRoute()
                    if len(self.route.points) == 0:
                        LogUtil.error("航线数据为空")
                        time.sleep(1)
                        continue

                    LogUtil.info(f"航线加载完成: {self.route}")

                    # 初始化本轮状态
                    self._reset_episode_state()
                    self.global_data.route = self.route
                    self.__reloadNavigationRoute(self.route)

                    # 本轮导航循环
                    startTime = time.time()
                    for step in range(MAX_STEP_PER_EPISODE):
                        if self.global_data.device_data.task_status == 0:
                            LogUtil.info(f"步骤 {step} 停止训练")
                            break

                        if (time.time() - startTime) > MAX_EPISODE_TIME:
                            LogUtil.info("本轮超时，提前结束")
                            break

                        self.done = self.navigationHandler(self.next_state, epoch, step)

                        # 定期更新PPO
                        if step > 0 and step % UPDATE_INTERVAL == 0:
                            self.ppo_agent.update()

                        self.setMonitorParameterValue()

                        if self.done or self.arrive:
                            break

                        time.sleep(0.1)

                    # 定期保存模型
                    if epoch % CHECKPOINT_INTERVAL == 0:
                        checkpoint_path = f"./PPO_ship_obstacle_{epoch}.pth"
                        self.ppo_agent.save(checkpoint_path)
                        LogUtil.info(f"模型已保存: {checkpoint_path}")

            except Exception as e:
                LogUtil.error(e)
            finally:
                time.sleep(2)

    def _reset_episode_state(self):
        """重置单轮训练的状态变量。"""
        self.episode_reward_sum = 0.0
        self.next_state = None
        self.done = False
        self.arrive = False
        self.destPointIndex = -1
        self.destPoint = None
        self.max_distance = 0.0

    # ==================== 状态获取 ====================

    def getState(self, scan, heading: float, current_distance: float) -> list:
        """
        从激光雷达和传感器数据提取状态向量。

        Args:
            scan: 2D激光雷达数据对象(.ranges属性)
            heading: 船体航向(rad)
            current_distance: 到目标点的直线距离(m)

        Returns:
            状态向量 [laser_features..., heading, distance, obstacle_min_range, obstacle_angle]
        """
        scan_range = self._extract_laser_features(scan)
        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)

        LogUtil.debug(
            f"状态: heading={heading:.2f}, distance={current_distance:.2f}, "
            f"obstacle_min={obstacle_min_range:.2f}, obstacle_angle={obstacle_angle}"
        )

        # 碰撞检测
        if COLLISION_DISTANCE > obstacle_min_range > 0:
            LogUtil.info(f"检测到碰撞! obstacle_min_range={obstacle_min_range}")
            self.done = True

        # 到达检测
        if self._is_last_waypoint_reached(current_distance):
            self.arrive = True

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle]

    def _extract_laser_features(self, scan) -> list:
        """
        从激光雷达数据提取特征，仅取前方180°扇区用于碰撞检测。

        假设ROS LaserScan的ranges按角度顺序排列，中间索引对应船体正前方。
        取中间180°范围内的数据点。
        """
        total_points = len(scan.ranges)
        # 前方180°占总扫描范围的一半，取数组中间部分
        half_count = total_points // 2
        start_idx = half_count // 2          # 前90°起始索引
        end_idx = start_idx + half_count     # 后90°结束索引

        scan_range = []
        for i in range(start_idx, end_idx):
            value = scan.ranges[i]
            if value == float('Inf') or value is None or np.isnan(value) or value > LASER_MAX_RANGE:
                scan_range.append(LASER_MAX_RANGE)
            else:
                scan_range.append(value)
        return scan_range

    def _is_last_waypoint_reached(self, current_distance: float) -> bool:
        """判断是否到达最后一个航路点。"""
        return (
            self.destPointIndex + 1 == len(self.route.points)
            and current_distance < self.arrive_distance
        )

    # ==================== 动作执行 ====================

    def step(self, state: list, action: int, laser_scan, heading: float,
             shipToNextWPDistance: float, max_distance: float) -> StepResult:
        """
        执行动作并获取环境反馈。

        Returns:
            StepResult: 包含下一状态、奖励、控制指令等
        """
        obstacle_min_range = state[-2]
        current_distance = state[-3]

        # 动作到控制量的映射
        adviseRotate, adviseSpeed = self._action_to_control(action, obstacle_min_range, current_distance)

        # 获取新状态
        new_state = self.getState(laser_scan, heading, shipToNextWPDistance)
        reward = self._compute_reward(new_state, action, max_distance)

        return StepResult(
            next_state=np.asarray(new_state),
            reward=reward,
            current_distance=current_distance,
            advise_speed=adviseSpeed,
            advise_rotate=adviseRotate,
            advised_heading=heading,
        )

    def _action_to_control(self, action: int, obstacle_min_range: float,
                           current_distance: float) -> tuple:
        """将离散动作映射为(转向百分比, 速度百分比)。"""
        # 角度计算：将action映射到[-100, +100]度范围
        ang_vel = ((self.action_size - 1) / 2 - action) * ANGULAR_VELOCITY_MAX / ((self.action_size - 1) / 2)
        adviseRotate = round(ang_vel, 0) * ACTION_TO_DEGREE_SCALE

        # 自适应速度
        adviseSpeed = DEFAULT_SPEED
        if obstacle_min_range < OBSTACLE_SLOW_RANGE:
            adviseSpeed = 0.1
        if current_distance < TARGET_SLOW_RANGE:
            adviseSpeed = 0.1

        adviseSpeed = min(round(adviseSpeed * 100 / DEFAULT_SPEED, 0), 100)
        return adviseRotate, adviseSpeed

    # ==================== 奖励函数 ====================

    def _compute_reward(self, state: list, action: int, max_distance: float) -> float:
        """计算综合奖励值。"""
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]

        # 距离奖励
        distance_reward = self._calc_distance_reward(current_distance, max_distance)

        # 更新最大距离记录
        if current_distance > max_distance:
            self.max_distance = current_distance

        # 航向奖励
        heading_reward = self._calc_heading_reward(action, heading, current_distance, max_distance)

        # 障碍物/接近目标奖励
        obstacle_reward = 0.0
        if obstacle_min_range < 3:
            obstacle_reward = REWARD_OBSTACLE_PENALTY
        elif current_distance < TARGET_SLOW_RANGE:
            obstacle_reward = REWARD_NEAR_TARGET_BONUS

        # 加权组合
        reward = (
            distance_reward * REWARD_WEIGHT_DISTANCE
            + obstacle_reward * REWARD_WEIGHT_OBSTACLE
            + heading_reward * REWARD_WEIGHT_HEADING
        )

        # 终止奖励/惩罚
        if self.arrive:
            LogUtil.info("到达目标!")
            reward += REWARD_ARRIVE_BONUS
        elif self.done:
            LogUtil.info("发生碰撞!")
            reward += REWARD_COLLISION_PENALTY

        return reward

    @staticmethod
    def _calc_distance_reward(current_distance: float, max_distance: float) -> float:
        """计算基于目标距离的奖励。"""
        if current_distance <= 1:
            return 0.0
        reward = 1 - (current_distance / max_distance)
        return reward * 2 if reward < 0 else reward * 5

    @staticmethod
    def _calc_heading_reward(action: int, heading: float,
                             current_distance: float, max_distance: float) -> float:
        """计算航向对齐奖励。"""
        yaw_rewards = []
        pi = math.pi
        for i in range(N_ACTIONS):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * abs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * pi) / pi)[0])
            yaw_rewards.append(tr)

        distance_rate = 2 ** (current_distance / max_distance) if max_distance > 0 else 1.0
        return round(yaw_rewards[action] * 5, 2) * distance_rate

    # ==================== 导航处理 ====================

    def navigationHandler(self, state, episode: int, step: int) -> bool:
        """
        导航处理主入口。
        Returns:
            bool: 是否终止本轮导航
        """
        try:
            # 1. 获取载体姿态
            pose_info = self._load_vehicle_pose_info()

            # 2. 路径规划
            nav_context = self._update_navigation_target(pose_info)
            if nav_context is None:
                return False

            # 3. 获取激光雷达数据
            laser_scan = self._wait_for_laser_data()
            if laser_scan is None:
                return False

            # 4. 执行PPO决策循环
            done = self._ppo_decision_loop(state, episode, step, laser_scan, nav_context)

            # 5. 日志输出
            self._log_navigation_result(nav_context)

            return done

        except Exception as e:
            LogUtil.error(f"navigationHandler异常: {e}")
            return False

    def _load_vehicle_pose_info(self) -> tuple:
        """获取载体位置和姿态信息。"""
        workModel = self.global_data.device_data.work_model
        isReturn = workModel == Constants.WorkMode.AutoReturn
        isAuto = workModel == Constants.WorkMode.Auto

        pose = self.global_data.scada_data.pose
        lng, lat = pose.lng, pose.lat
        heading = pose.yaw
        if heading > 180:
            heading -= 360

        return isAuto, isReturn, lng, lat, heading, pose.speed, pose.rotate_speed

    def _update_navigation_target(self, pose_info: tuple):
        """更新路径规划目标点，返回导航上下文。"""
        isAuto, isReturn, lng, lat, _, _, _ = pose_info
        self.routePlaneService.setCurrentPos(lng, lat, isReturn)

        try:
            nextPointIndex = self.routePlaneService.curNextIndex
            prevPointIndex = self.routePlaneService.curPrevIndex
            nextPoint = self.route.points[nextPointIndex]

            if prevPointIndex < 0 or prevPointIndex >= len(self.route.points):
                prevPointIndex = nextPointIndex

            if nextPointIndex != self.destPointIndex or nextPoint != self.destPoint:
                self.max_distance = self.routePlaneService.distanceMetersShip2NextWP
                LogUtil.info(f"切换至航点[{nextPointIndex}]={nextPoint}, 距离={self.max_distance:.1f}m")

            self.destPointIndex = nextPointIndex
            self.destPoint = self.route.points[self.destPointIndex]
            self.prevPointIndex = prevPointIndex
            self.prevPoint = self.route.points[prevPointIndex]

        except Exception as e:
            LogUtil.error(f"路径规划异常: {e}")

        if self.destPointIndex == -1:
            return None

        shipToNextWPDistance = self.routePlaneService.distanceMetersShip2NextWP
        return {
            'nextPointIndex': nextPointIndex,
            'shipToNextWPDistance': shipToNextWPDistance,
        }

    def _wait_for_laser_data(self):
        """等待新的激光雷达数据，超时返回None。"""
        laser_scan = self.get_laser_scan(timeout=2)
        if laser_scan is None:
            LogUtil.info("获取激光雷达数据超时(2s)")
            return None
        self.last_laser_scan = laser_scan
        return laser_scan

    def _ppo_decision_loop(self, state, episode: int, step: int,
                           laser_scan, nav_context: dict) -> bool:
        """PPO核心决策逻辑。"""
        heading = self._get_current_heading()

        # 初始化状态
        if state is None:
            state = self.getState(laser_scan, heading, nav_context['shipToNextWPDistance'])

        state_tensor = torch.FloatTensor(state).to(device)
        action = self.ppo_agent.select_action(state_tensor)

        # 执行动作
        result = self.step(
            state_tensor.cpu().numpy().tolist(), action,
            laser_scan, heading, nav_context['shipToNextWPDistance'], self.max_distance
        )
        self.next_state = result.next_state

        # 记录经验
        self.ppo_agent.buffer.rewards.append(result.reward)
        self.ppo_agent.buffer.is_terminals.append(self.done)
        self.episode_reward_sum += result.reward

        # 输出控制量
        self._output_control_commands(result, episode, step, nav_context['nextPointIndex'])

        LogUtil.debug(
            f"Step={step} Action={action} Reward={result.reward:.2f} "
            f"Done={self.done} Arrive={self.arrive}"
        )

        if self.done:
            return True

        if self.arrive:
            LogUtil.info(f"Episode结束于step={step}, 总奖励={self.episode_reward_sum:.1f}")
            return True

        return False

    def _get_current_heading(self) -> float:
        """获取当前航向。"""
        pose = self.global_data.scada_data.pose
        heading = pose.yaw
        return heading - 360 if heading > 180 else heading

    def _output_control_commands(self, result: StepResult, episode: int, step: int, nextPointIndex: int):
        """将控制命令输出到global_data。"""
        self.global_data.updateAlgorithmOutput(
            episode, step, int(self.episode_reward_sum),
            result.reward, MAX_EPOCH, 2
        )
        self.global_data.updateThrottleRudderOutput(
            result.advise_speed, result.advise_rotate, result.advised_heading,
            nextPointIndex, result.current_distance
        )

    def _log_navigation_result(self, nav_context: dict):
        """记录导航结果日志。"""
        LogUtil.info(
            f"航点={self.destPointIndex} 距离={nav_context['shipToNextWPDistance']:.1f}m "
            f"速度=... 转向=..."
        )

    # ==================== 参数注册 ====================

    def registerParameter(self):
        """注册可调参数到ROS2参数服务器。"""
        self.ros_ctrl.initParameterList()

        params = [
            Parameter(name='/usv/auto/plan/attainRadius', dataType='float', defaultData=2),
            Parameter(name='/usv/auto/param2', dataType='int', defaultData=[20]),
            Parameter(name='/usv/auto/param3', dataType='int', defaultData=[30, 31]),
            Parameter(name='/usv/auto/param4', dataType='float', defaultData=[40.1, 40.2]),
            Parameter(name='/usv/auto/param6', dataType='str', defaultData='some msg'),
        ]

        for param in params:
            if self.ros_ctrl.registParameter(param):
                self.global_data.parameterAdjustMap.update({param.name: param})
                if param.name == '/usv/auto/plan/attainRadius':
                    self.routePlaneService.wayPointRadius = param.value

    def setMonitorParameterValue(self):
        """监控参数回调（待实现）。"""
        pass

    # ==================== 内部工具方法 ====================

    def __loadVehiclePoseInfo(self):
        """向后兼容接口。"""
        return self._load_vehicle_pose_info()

    def __reloadNavigationRoute(self, route) -> bool:
        """重新加载导航航线。"""
        self.route = route
        self.routePlaneService.reset(route=self.route)
        return True

    def get_laser_scan(self, timeout: float):
        """等待新的激光雷达数据，超时返回None。"""
        laser_start_time = time.time()
        while self.global_data.laser_data == self.last_laser_scan:
            time.sleep(0.1)
            if (time.time() - laser_start_time) > timeout:
                return None
        return self.global_data.laser_data

    # 向后兼容的旧接口
    def setReward(self, state, action, max_distance):
        """旧接口兼容，委托给_compute_reward。"""
        return self._compute_reward(state, action, max_distance)


# 向后兼容：保留旧类名供main.py等外部引用
DQN_NAV = PPONav
