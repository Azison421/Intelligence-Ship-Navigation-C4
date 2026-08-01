

"""
install pytorch in window10 cpu
python3.9 -m pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

"""

import torch
import math
import numpy as np
from geopy.distance import great_circle
from geopy import Point
from usvlib4ros.dqn.dqn_ros_service import DictToObject,DQN_ROS_Service
from usvlib4ros.usvRosUtil import USVMathUtil,LogUtil
from usvlib4ros.dqn.twist import Twist
pi = math.pi

class Env_test:

    def __init__(self, action_size=25, rosHost = "192.168.3.119"):  # 初始化值
        DQN_ROS_Service.startService(rosHost=rosHost)
        DQN_ROS_Service.registerTrainActionProxy()
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.goal_distance = 0
        self.score = 0
        self.position = DictToObject(**{"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0}})   # Pose
        self.index = -1                     # 从0开始计数用
        self._first_reset_called = False
        pass

    def isTrainActionTrigger(self):
        return DQN_ROS_Service.isTrainning()

    # 计算经纬度的距离 lat lng index应该是航线给出的
    def getTrueDistance(self):
        current_pos = DQN_ROS_Service.get_pose()
        route_list = DQN_ROS_Service.get_route()

        ship_pos = Point(current_pos.lat, current_pos.lng)
        target_pos = Point(route_list.points[self.index]['lat'], route_list.points[self.index]['lng'])
        true_distance = great_circle(ship_pos, target_pos).meters

        return true_distance

    # 用于判断是否到最后一个点
    def getMaxRoute(self):
        route_list = DQN_ROS_Service.get_route()
        max_epoch = len(route_list.points)

        return max_epoch

    # 获取起始点，包括后续的起始点
    def getIndex(self):
        route_list = DQN_ROS_Service.get_route()
        self.index = route_list.startIndex

        return self.index

    # 到达目标后更新当前导航点
    def updateIndex(self):
        self.index += 1
        return self.index

    def currentTarget(self):
        return self.index

    # 接收订阅的odom消息，并解析出机器人当前的位置和朝向//似乎未使用
    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        # orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = USVMathUtil.quaternionToEulerAngles(orientation.w, orientation.x, orientation.y, orientation.z)  # 将四元数转换成欧拉角，计算出当前机器人的位置和朝向

        goal_angle = math.atan2(self.goal_y - self.position.position.y, self.goal_x - self.position.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    # getstate函数，获取机器人当前状态
    # 在功能测试，主要是获取到目标点之间的距离，只需要获取距离即可（使用gps）
    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.5  # 设定判断碰撞的最小距离0.13m // 碰撞最小距离根据模型来确定 1 or 1.3 侧边碰撞比较敏感
        done = False

        for i in range(len(scan.ranges)):           # 雷达数据转换潜在的问题-裁剪数据 0.35-3.5 ||| 1-5 0.5碰撞 1到达
            value = scan.ranges[i]                  # 意味着智能体的视野为5m
            if scan.ranges[i] == float('Inf'):      # float('Inf')表示正无穷大
                scan_range.append(5)                # 雷达扫描到的值为无穷大说明该方向没有障碍物，赋值为3.5，该方向可通行
            elif value is None:                     # 碰撞
                scan_range.append(5)                # 雷达扫描检测到某方向的值为无效（nan），则赋值为0 (3.5) 探测最大距离
            elif np.isnan(scan.ranges[i]):
                scan_range.append(5)                # 雷达扫描检测到某方向的值为无效（nan），则赋值为0       若可以通行，则赋值max/2
            elif value > 5:
                scan_range.append(5)
            else:
                scan_range.append(scan.ranges[i])   # 其他方向保持扫描到的值不变

        obstacle_min_range = round(min(scan_range), 2)  # min函数找到最小值，round函数将结果四舍五入到小数点后两位
        obstacle_angle = np.argmin(scan_range)          # 返回最小值对应的索引，用于确定出现最小值的角度
        if min_range > obstacle_min_range > 0:      # 碰撞判定 0.5m
            done = True

        current_distance = self.getTrueDistance()    # 到达目标点判定
        self.goal_distance = current_distance
        if current_distance <= 1:
            self.get_goalbox = True
            done = True

        return scan_range + [heading, current_distance, obstacle_min_range,
                             obstacle_angle], done  # 返回状态，这里注意他的组成！比较重要

    def setReward(self, state, done, action, max_distance):       # 少一个方向reward    scanreward和obreward只能有一个好像
        yaw_reward = []
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]
        scan_reward = 0
        ob_reward = 0

        if current_distance > 1:
            scan_reward = 1 - (current_distance / max_distance)
            if scan_reward < 0:
                scan_reward = scan_reward * 100       # 2
            else:
                scan_reward = scan_reward * 80        # 5
        # if obstacle_min_range <= 2:
        #     scan_reward = min(scan_reward,obstacle_min_range)

        if current_distance < max_distance:                        #设置新的max返回
            max_distance = current_distance

        # 从五个方向上给出的奖励值，也就是对应了五个action 这里随着action要变动
        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)    # self.dis是全局的距离吗 这里的reward值都是正数
        forward_reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if obstacle_min_range < 3:
            ob_reward = -5
        if current_distance < 3:
            ob_reward = 1
        # else:
        #     ob_reward = 1                                       #考虑这里就不要这个了

        # reward = scan_reward * 0.5  + forward_reward * 0.3 + ob_reward * 0.2        #尝试使用加权奖励
        reward = scan_reward * 0.8 + ob_reward * 0.1 + forward_reward * 0.1

        #sprint(self.goal_distance, max_distance)

        if self.get_goalbox:
            LogUtil.info("Goal!!")
            reward += 1000

            twist = Twist()
            DQN_ROS_Service.updateVehicleAction(twist,0, 0, index=self.currentTarget()+1)
            print("reset by Goal")
            self.goal_x, self.goal_y = DQN_ROS_Service.wait_dest_position_refresh()     # 目标刷新 可以被禁用
            self.goal_distance = self.getTrueDistance()
            self.get_goalbox = False
        elif done:                                                                      # done可另行处理
            LogUtil.info("Collision!!")
            reward += -500
            twist = Twist()
            DQN_ROS_Service.updateVehicleAction(twist,0,0, index=self.currentTarget())
            print("reset by Collision")
            self.goal_x, self.goal_y = DQN_ROS_Service.wait_dest_position_refresh()
            # time.sleep(5)
            self.goal_distance = self.getTrueDistance()
            self.get_goalbox = False

        return reward, max_distance

    def reset(self):
        try:
            DQN_ROS_Service.wait_for_service_reset(timeout=5)
        except Exception as e:
            print(e)

        data = None
        while data is None:
            try:
                data = DQN_ROS_Service.wait_for_message_laserScan(timeout=5)
            except Exception as e:
                pass

        self.initGoal = True
        if self.initGoal:
            try:
                self.goal_x, self.goal_y = DQN_ROS_Service.wait_dest_position_refresh()
                self.goal_distance = self.getTrueDistance()
            except Exception as e:
                pass
            self.initGoal = False

        self.goal_distance = self.getTrueDistance()
        state, done = self.getState(data)

        return np.asarray(state)

    # step函数是用于执行一个动作并观察环境反馈的函数。它接收一个动作作为输入，并返回执行该动作后的新状态、奖励和完成标志。
    def step(self, state, action, max_distance):
        obstacle_min_range = state[-2]  # 最近障碍物距离
        current_distance = state[-3]    # 与目标距离
        heading = state[-4]
        max_angular_vel = 100
        ang_vel = ((self.action_size - 1) / 2 - action) * 100.0 / ((self.action_size - 1) / 2)      # 计算角度值

        vel_cmd = Twist()
        vel_cmd.angular.z = round(ang_vel,0) * 2  # [0，100]转向速度百分比

        vel_cmd.linear.x = 0.5                    # 0.5  设置速度默认值

        # 实现靠近障碍物的时候速度慢，离障碍物远的时候速度快，查看是否可以减速
        if obstacle_min_range < 4:                # 0.5 1.3是碰撞
            vel_cmd.linear.x = 0.1                # obstacle_min_range * 0.1  # m/s range/2，让速度再小一点，不然可能影响碰撞  /2*0.5
        if current_distance < 3:                  # 与目标点之间的距离
            vel_cmd.linear.x = 0.1                # current_distance * 0.1  油门给到20%

        vel_cmd.linear.x = min(round(vel_cmd.linear.x * 100 / 0.5, 0), 100)    # [0,100]油门百分比
        DQN_ROS_Service.updateVehicleAction(twist=vel_cmd, heading=heading, distance=current_distance, index=self.currentTarget())

        data = None
        while data is None:
            try:
                data = DQN_ROS_Service.wait_for_message_laserScan(timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward, max_distance = self.setReward(state, done, action, max_distance)

        return np.asarray(state), reward, done, max_distance

    pass