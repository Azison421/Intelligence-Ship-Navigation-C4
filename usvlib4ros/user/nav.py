"""
install pytorch in window10 cpu
python3.9 -m pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import math
import time
import threading
import numpy as np
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.navigation.route_plan_service import RoutePlanService

from usvlib4ros.msg.global_data import GlobalData, DictToObject, Point, Constants
from usvlib4ros.msg.parameter import Parameter
from usvlib4ros.usvRosUtil import LogUtil
from usvlib4ros.user.PP0 import PPO

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
N_ACTIONS = 5  # Actions: +100°, -100°, +50°, -50°, 0°
N_STATES = 76  # State dimension based on LiDAR data origin=184
MEMORY_CAPACITY = 2000
BATCH_SIZE = 128
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
GAMMA = 0.99
K_EPOCHS = 80
EPS_CLIP = 0.2
ACTION_STD_INIT = 0.6
MAX_EPOCH = 4000

class DQN_NAV:
    Instance = None

    """
    USV auto navigation service
    """

    def startService(self):
        self.navThread = threading.Thread(target=self.run)
        self.navThread.setDaemon(True)
        self.navThread.start()
        pass

    def __init__(self, ros_ctrl: Ros2Controller, global_data: GlobalData, xyzAxis=True):
        self.ros_ctrl: Ros2Controller = ros_ctrl
        self.global_data: GlobalData = global_data
        self.navThread = None

        #模型
        self.ppo_agent = PPO(N_STATES, N_ACTIONS, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP, False, ACTION_STD_INIT)
        self.next_state = None
        self.action_size = 5 #动作空间

        self.route = None   # 航线，
        self.destPoint = Point()  # 目标点  lng lat high speed
        self.destPointIndex = -1  # 目标点self.destPoint在导航点global_data.getInstance().route.points中的下标，-1表示无导航点。
        self.prevPoint = Point()
        self.prevPointIndex = -1
        self.max_distance = 0   # 船到目标的起始距离
        self.score = 0          # 评分
        self.episode_reward_sum = 0     # 本轮总分数
        self.arrive = False     # 到达
        self.done = False       # 碰撞障碍或越界
        self.arrive_distance = 1.0     # 最小距离，船到目标的距离低于此值时，认为到达目标。

        self.routePlaneService = RoutePlanService(wayPointRadius=self.arrive_distance, route=self.route)
        """启动时缓存下当前激光雷达数据，算法中检测激光雷达数据对象是否变化来判断是否收到新的数据"""
        self.last_laser_scan = global_data.laser_data
        pass

    def run(self):
        """"""
        """register parameter """
        """adjust parameter data update by topic thread"""
        self.registerParameter()

        while True:
            try:
                print("wait train button trigger ...")
                """self.ros_ctrl.device_data.task_status : 0 停止导航或结束训练;1 开启导航或训练 """
                while self.global_data.device_data.task_status == 0:
                    """unity端 训练模式，开始训练按钮没有按下，请按下开始训练按钮"""
                    time.sleep(1)
                    continue
                    pass

                for e in range(MAX_EPOCH):
                    if self.global_data.device_data.task_status == 0:
                        """unity端 训练模式，点击了停止训练按钮"""
                        print("Stop train ...")
                        break

                    """unity端 训练模式，第e轮训练"""
                    print(f"train {e} ...")
                    """初始化 unity端船位置和障碍物位置"""
                    print(f"Reset unity ...")
                    result = self.ros_ctrl.reset_unity()
                    time.sleep(1)  #等待1s ，等待self.global_data.device_data.reset_status 数据更新。
                    """等待unity复位完成, self.global_data.device_data.reset_status: 1 导航发出复位请求，2 unity回复复位完成"""
                    while self.global_data.device_data.reset_status != 2:
                        time.sleep(1)
                        continue

                    """设置为自动模式，启用算法输出控制船体（非自动模式，船体运动由其他模块输出控制）"""
                    self.ros_ctrl.set_auto_work()

                    """获取全局路线信息与导航算法路线信息比较"""

                    """重新加载航线"""
                    self.route = self.ros_ctrl.getRoute()
                    if len(self.route.points) == 0:
                        """没收到正确的航线数据"""
                        LogUtil.info("Error : len(route.points) is 0. ")
                        time.sleep(1)
                        continue
                    print(f"Route {self.route}...")

                    """开始新一轮的导航，清空缓存，加载航线"""
                    self.episode_reward_sum = 0
                    self.next_state = None
                    self.done = False
                    self.arrive = False
                    self.destPointIndex = -1
                    self.destPoint = None
                    self.max_distance = 0

                    self.global_data.route = self.route
                    self.__reloadNavigationRoute(self.route)

                    """第e次训练，最多3000步"""
                    startTime = time.time()
                    for t in range(3000):
                        if self.global_data.device_data.task_status == 0:
                            """unity端 训练模式，点击了停止训练按钮"""
                            print(f"Stop train step {t}...")
                            break
                        tempTime = time.time() - startTime

                        if tempTime > 300:
                            """本轮训练持续时间过长，主动结束本轮，开启下一轮"""
                            break

                        """第e轮训练的第t步"""
                        self.navigationHandler(self.next_state,e,t)

                        if t % 2000 == 0:
                            self.ppo_agent.update()

                        """monitor parameter data update by this thread"""
                        self.setMonitorParameterValue()

                        if self.done or self.arrive:
                            break

                        time.sleep(0.1)

                    if e % 100 == 0:
                        checkpoint_path = f"./PPO_ship_obstacle_{e}.pth"
                        self.ppo_agent.save(checkpoint_path)
                pass
            except Exception as e:
                LogUtil.error(e)
                pass
            finally:
                time.sleep(0.02)#100ms

    # getstate函数，获取机器人当前状态
    # 在功能测试钟，主要是获取到目标点之间的距离，只需要获取距离即可（使用gps）
    def getState(self, scan,heading,current_distance):
        """
        scan:2D激光雷达数据
        heading：船体方向，
        current_distance：到目标点距离
        """
        print(f"State before action: {scan}")
        min_range = 1  # origin=0.5 设定判断碰撞的最小距离0.13m // 碰撞最小距离根据模型来确定 1 or 1.3 侧边碰撞比较敏感
        scan_range = []

        # 只获取前36个和后36个数据点
        selected_indices = list(range(36)) + list(range(len(scan.ranges) - 36, len(scan.ranges)))

        # for i in range(len(scan.ranges)):           # 雷达数据转换潜在的问题-裁剪数据 0.35-3.5 ||| 1-5 0.5碰撞 1到达
        for i in selected_indices:

            value = scan.ranges[i]  # 意味着智能体的视野为5m
            if scan.ranges[i] == float('Inf'):  # float('Inf')表示正无穷大
                scan_range.append(5)  # 雷达扫描到的值为无穷大说明该方向没有障碍物，赋值为3.5，该方向可通行
            elif value is None:  # 碰撞
                scan_range.append(5)  # 雷达扫描检测到某方向的值为无效（nan），则赋值为0 (3.5) 探测最大距离
            elif np.isnan(scan.ranges[i]):
                scan_range.append(5)  # 雷达扫描检测到某方向的值为无效（nan），则赋值为0       若可以通行，则赋值max/2
            elif value > 5:
                scan_range.append(5)
            else:
                scan_range.append(scan.ranges[i])  # 其他方向保持扫描到的值不变

        obstacle_min_range = round(min(scan_range), 2)  # min函数找到最小值，round函数将结果四舍五入到小数点后两位
        obstacle_angle = np.argmin(scan_range)  # 返回最小值对应的索引，用于确定出现最小值的角度

        # 打印关键状态变量（调试用）
        print(
            f"[State Debug] heading: {heading:.2f} rad, "
            f"current_distance: {current_distance:.2f} m, "
            f"obstacle_min_range: {obstacle_min_range:.2f} m, "
            f"obstacle_angle: {obstacle_angle} (index)"
        )

        if min_range > obstacle_min_range > 0:  # 碰撞判定 0.5m
            print(f'Collision detected with obstacle_min_range: {obstacle_min_range}')
            self.done = True

        if self.destPointIndex + 1 == len(self.route.points) and current_distance < self.arrive_distance:
            """到达最后一个导航点"""
            self.arrive = True

        # print("min dist = %s , angle = %s " %(obstacle_min_range,obstacle_angle))
        return scan_range + [heading, current_distance, obstacle_min_range,
                             obstacle_angle]  # 返回状态，这里注意他的组成！比较重要

    # step函数是用于执行一个动作并观察环境反馈的函数。它接收一个动作作为输入，并返回执行该动作后的新状态、奖励和完成标志。
    def step(self, state, action, laser_scan, heading, shipToNextWPDistance, max_distance):
        """
        state: 状态空间
        action：随机动作
        laser_scan：2d激光数据
        heading：船体方向
        shipToNextWPDistance：船到目标点直线距离
        """
        obstacle_min_range = state[-2]  # 最近障碍物距离
        current_distance = state[-3]  # 与目标距离
        heading = state[-4]
        max_angular_vel = 100  #
        ang_vel = ((self.action_size - 1) / 2 - action) * 100.0 / ((self.action_size - 1) / 2)  # 计算角度值
        adviseRotate = round(ang_vel, 0) * 2 # [-100，100]转向速度百分比
        adviseSpeed = 0.5  # 0.5                   #设置速度默认值

        # 实现靠近障碍物的时候速度慢，离障碍物远的时候速度快，查看是否可以减速
        if obstacle_min_range < 4:  # 0.5 1.3是碰撞
            adviseSpeed = 0.1  # obstacle_min_range * 0.1  # m/s range/2，让速度再小一点，不然可能影响碰撞  /2*0.5
        if current_distance < 3:  # 与目标点之间的距离
            adviseSpeed = 0.1  # current_distance * 0.1  油门给到20%

        adviseSpeed = min(round(adviseSpeed * 100 / 0.5, 0), 100)  # [0,100]油门百分比

        state = self.getState(laser_scan, heading, shipToNextWPDistance)

        reward = self.setReward(state, action, self.max_distance)

        return np.asarray(state), reward, max_distance, adviseSpeed, adviseRotate, heading, current_distance

    def registerParameter(self):
        """"""
        """重启导航模块后 清空已有参数列表"""
        self.ros_ctrl.initParameterList()
        """adjust parameter data update by topic thread"""

        """导航点有效半径，船位于此半径内即为达到导航点"""
        name = '/usv/auto/plan/attainRadius'
        attainRadius = Parameter(name=name, dataType='float', defaultData=2)
        result = self.ros_ctrl.registParameter(attainRadius)
        if result:
            self.global_data.parameterAdjustMap.update({name: attainRadius})
            self.routePlaneService.wayPointRadius = attainRadius.value

        param2 = Parameter(name='/usv/auto/param2', dataType='int', defaultData=[20])
        result = self.ros_ctrl.registParameter(param2)
        if result:
            self.global_data.parameterAdjustMap.update({'/usv/auto/param2': param2})

        param3 = Parameter(name='/usv/auto/param3', dataType='int', defaultData=[30, 31])
        result = self.ros_ctrl.registParameter(param3)
        if result:
            self.global_data.parameterAdjustMap.update({'/usv/auto/param3': param3})

        param4 = Parameter(name='/usv/auto/param4', dataType='float', defaultData=[40.1, 40.2])
        result = self.ros_ctrl.registParameter(param4)
        if result:
            self.global_data.parameterAdjustMap.update({'/usv/auto/param4': param4})

        param6 = Parameter(name='/usv/auto/param6', dataType='str', defaultData='some msg')
        result = self.ros_ctrl.registParameter(param6)
        if result:
            self.global_data.parameterAdjustMap.update({'/usv/auto/param6': param6})

        """monitor parameter """
        # TODO

        pass

    def setMonitorParameterValue(self):
        """monitor parameter """

        pass

    def __loadVehiclePoseInfo(self):
        """
        isAuto  : True 自动导航;False 非自动导航
        isReturn:True 自动返航;False 非自动返航
        lng     : 航行器当前经度
        lat     : 航行器当前纬度
        realSpeed:航行器当前速度 m/s
        realRotateSpeed:航行器当前角速度 °/s
        """
        workModel = self.global_data.device_data.work_model
        isReturn = workModel == Constants.WorkMode.AutoReturn
        isAuto = workModel == Constants.WorkMode.Auto

        # lng = global_data.getInstance().gpsInfo.lng
        # lat = global_data.getInstance().gpsInfo.lat
        # heading = global_data.getInstance().gpsInfo.heading
        # realSpeed = global_data.getInstance().gpsInfo.speed
        # realRotateSpeed = 0.0 # TODO °/s

        # if global_data.getInstance().autoCommand.poseModel == Constants.PoseMode.UintyPose:
        pose = self.global_data.scada_data.pose
        lng = pose.lng
        lat = pose.lat
        heading = pose.yaw
        if heading > 180:
            heading = heading - 360
        realSpeed = pose.speed
        realRotateSpeed = pose.rotate_speed

        return isAuto, isReturn, lng, lat, heading, realSpeed, realRotateSpeed

    def __reloadNavigationRoute(self, route):
        """return True:new route"""
        """更新route"""
        self.route = route  # load route info from global data
        self.routePlaneService.reset(route=self.route)
        return True

    def get_laser_scan(self,timeout):
        laser_start_time = time.time()
        """
        self.global_data.laser_data : ros2 提供的最新的激光雷达数据
        self.last_laser_scan ： 算法上次处理的激光雷达数据
        是同一个对象则未收到新的数据，超时返回None
        """
        while self.global_data.laser_data == self.last_laser_scan:
            time.sleep(0.1)
            get_laser_use_time = time.time() - laser_start_time
            if get_laser_use_time >timeout:
                return None
            pass

        laser_scan = self.global_data.laser_data
        return laser_scan

    def navigationHandler(self,state,episode,step):
        """
        导航算法:
        """
        try:

            """
            获取航行器姿态和位置
            isAuto  : True 自动导航;False 非自动导航
            isReturn:True 自动返航;False 非自动返航
            lng     : 航行器当前经度
            lat     : 航行器当前纬度
            realSpeed:航行器当前速度 m/s
            realRotateSpeed:航行器当前角速度 °/s
            """
            isAuto, isReturn, lng, lat, heading, realSpeed, realRotateSpeed = self.__loadVehiclePoseInfo()

            """路径规划：航线中多个点进行有序导航"""
            self.routePlaneService.setCurrentPos(lng, lat, isReturn)

            """获取路径规划结果"""
            try:
                nextPointIndex = self.routePlaneService.curNextIndex
                prevPointIndex = self.routePlaneService.curPrevIndex

                nextPoint = self.route.points[nextPointIndex]
                if prevPointIndex < 0 or prevPointIndex >= len(self.route.points):
                    prevPointIndex = nextPointIndex

                if nextPointIndex != self.destPointIndex or nextPoint != self.destPoint:
                    """切换目标点,记录此时到目标点的距离"""
                    self.max_distance = self.routePlaneService.distanceMetersShip2NextWP
                    LogUtil.info(f"Goto point[{nextPointIndex}] = {nextPoint}, distance = {self.max_distance}")

                """当前导航目标点"""
                self.destPointIndex = nextPointIndex
                self.destPoint = self.route.points[self.destPointIndex]  # 目标点
                self.prevPointIndex = prevPointIndex
                self.prevPoint = self.route.points[self.prevPointIndex]  # 目标点
            except Exception as e:
                LogUtil.error(e)

            if self.destPointIndex == -1:
                """无导航目标点"""
                return

            """自动导航相关数据"""
            destLng = self.destPoint.lng
            destLat = self.destPoint.lat
            prevLng = self.prevPoint.lng
            prevLat = self.prevPoint.lat
            shipToNextWPDistance = self.routePlaneService.distanceMetersShip2NextWP

            """等待新的激光雷达数据"""
            laser_scan = self.get_laser_scan(timeout=2)     # 2s 超时
            if laser_scan is None:
                print("Get 2d laser scan timeout.")
                return False
            self.last_laser_scan = laser_scan

            """*******************自定义导航算法 Start***************************"""
            """第一次进入导航"""
            if state is None:
                state = self.getState(laser_scan, heading, shipToNextWPDistance)

            state = torch.FloatTensor(state).to(device)
            action = self.ppo_agent.select_action(state)
            self.next_state, reward, current_distance,adviseSpeed,adviseRotate,advisedHeading,current_distance =\
                self.step(state.cpu().numpy().tolist(),action,laser_scan,heading, shipToNextWPDistance, self.max_distance)

            self.ppo_agent.buffer.rewards.append(reward)
            self.ppo_agent.buffer.is_terminals.append(self.done)
            self.episode_reward_sum += reward

            """算法输出结果保存到global_data"""
            self.global_data.updateAlgorithmOutput(episode, step, int(self.episode_reward_sum), reward, MAX_EPOCH, 2)
            self.global_data.updateThrottleRudderOutput(adviseSpeed, adviseRotate, advisedHeading, nextPointIndex,
                                                        shipToNextWPDistance)
            print(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {self.done}")

            if self.arrive or self.done:
                print(f"Episode ended at step {step}, total reward: {self.episode_reward_sum}")
                return True
            """---------------------------自定义导航算法 END-------------------------"""
            """get result"""
            LogUtil.info("Goto index = %s, heading : %s = > %s .distance=%s .speed = %s , rotate = %s ." % (
                self.destPointIndex, heading, advisedHeading, shipToNextWPDistance, adviseSpeed, adviseRotate))
            pass
        except Exception as e:
            LogUtil.error(e)
        return False

    def setReward(self, state, action, max_distance):       # 少一个方向reward    scanreward和obreward只能有一个好像
        yaw_reward = []
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]
        scan_reward = 0
        ob_reward = 0

        if current_distance > 1:
            scan_reward = 1 - (current_distance / max_distance)
            if scan_reward < 0:
                scan_reward = scan_reward * 2
            else:
                scan_reward = scan_reward * 5

        if current_distance > max_distance:
            self.max_distance = current_distance

        # 从五个方向上给出的奖励值，也就是对应了五个action 这里随着action要变动
        pi = math.pi
        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.max_distance)    # self.dis是全局的距离吗 这里的reward值都是正数
        forward_reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if obstacle_min_range < 3:
            ob_reward = -5
        if current_distance < 3:
            ob_reward = 1
        # else:
        #     ob_reward = 1                                       #考虑这里就不要这个了

        # reward = scan_reward * 0.5  + forward_reward * 0.3 + ob_reward * 0.2        #尝试使用加权奖励
        reward = scan_reward * 0.6 + ob_reward * 0.2 + forward_reward * 0.2

        if self.arrive:
            LogUtil.info("Goal!!")
            reward += 1000
        elif self.done:
            LogUtil.info("Collision!!")
            reward += -500
        return reward