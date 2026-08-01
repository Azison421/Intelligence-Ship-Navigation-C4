"""
install pytorch in window10 cpu
python3.9 -m pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt

from usvlib4ros.dqn.dqn_env_test import Env_test
from usvlib4ros.dqn.dqn_ros_service import DQN_ROS_Service
from usvlib4ros.usvRosUtil import LogUtil,USVRosbridgeClient
from usvlib4ros.dqn.twist import Twist

#device 如果有cuda更好，看能否加快计算速度
device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

N_ACTIONS = 5                           # 油门, 推杆 同时也是网络输出的通道数,这个要改一下    5 and 28  25 and 256
N_STATES = 30                           # 状态个数 (4个) 365       想办法给state降维
MEMORY_CAPACITY = 2000                  # 记忆库容量
BATCH_SIZE = 128                        # 样本数量 原来是128
LR = 0.001                              # 学习率
GAMMA = 0.9                             # reward discount，折扣因子，防止学习过快，关键参数
TARGET_REPLACE_ITER = 10
MAX_Epoch = 10000                       # 在test模式里，要确定一次给出的导航点的个数
reward_list = []                        # 评估用 数据存储和绘图
MODEL_PATH = 'a_100.pt'

# 定义Net类 (定义网络)
class Net(nn.Module):

    def __init__(self):                                                         # 定义Net的一系列属性 网络的架构是两个全连接层带激活函数 输出通道是action
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 128)                                     # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 =nn.Linear(128,128)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(128, N_ACTIONS)                                    # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        x = F.relu(self.fc2(x))
        x = F.dropout(self.fc2(x))
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value
    pass

class DQN:

    def __init__(self,loadModel = True):
        self.eval_net = Net()       # 评估网络
        self.target_net = Net()     # 目标网络
        self.memory_counter = 0     # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, 62))  # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.start_epoch = 0
        self.epsilon = 1               # 贪心值，有1-ep的概率探索路径 这个影响很大 原来数据为1.0，贪心策略会自动更新
        self.epsilon_min = 0.1         # 若加载模型，贪心策略设置为0，不在探索路径 之前是0.01，一直学习到0.99
        self.load_ep = 780             # 可以理解为超参数
        self.loss = 0
        self.q_eval = 0
        self.q_target = 0
        self.learn_step_counter = 0
        if loadModel:
            self.epsilon = 0           # 贪心为0，都给出最优策略
            self.start_epoch = self.load_ep
            checkpoint = torch.load(MODEL_PATH)
            print(checkpoint.keys())
            print(checkpoint['epoch'])
            print(checkpoint)

            self.target_net.load_state_dict(checkpoint['target_net'])
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = int(checkpoint['epoch']) + 1
            print("load model")

            pass

    def choose_action(self, x):  # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() > self.epsilon:  # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)  # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式

            action = action[0]  # 输出action的第一个数
        else:  # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)  # 这里action随机等于0或1 (N_ACTIONS = 2)
        return action  # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))  # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY  # 获取transition要置入的行数
        self.memory[index, :] = transition  # 置入transition
        self.memory_counter += 1  # memory_counter自加1

    def learn(self):  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每10步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1  # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]  # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)     # 目标和训练网络的均方差
        self.loss = torch.max(loss)
        self.q_eval = torch.max(q_eval)
        self.q_target = torch.max(q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数

if __name__ == "__main__":
    try:
        USVRosbridgeClient.Host = "192.168.3.119"
        env = Env_test(action_size=N_ACTIONS, rosHost=USVRosbridgeClient.Host)
        dqn = DQN()                             # 在初始化的时候已经完成模型的load
        e = dqn.start_epoch
        print("wait button trigger ...")

        # 导航点的数量
        MAX_Epoch = env.getMaxRoute()
        episode_reward_sum = 0                  # 总奖励值
        goal_count = 0

        print("testing load model %d ..." % (e))

        while True:
            currentTarget = env.getIndex()      # 保存当前导航点

            if currentTarget > MAX_Epoch:       # 标识最后一个导航点
                break

            env.getIndex()                      # 初始化start 先初始化start
            s = env.reset()                     # 初始化一个state

            max_distance = env.getTrueDistance()

            # episode_reward_sum = 0            # 初始化该循环对应的episode的总奖励
            done = False
            start = time.time()

            while not env.get_goalbox:          # 到达之前一直循环 步数不限

                a = dqn.choose_action(s)  # 输入该步对应的状态s，选择动作
                s_, r, done, max_distance = env.step(s, a, max_distance)  # 执行动作，获得反馈

                dqn.store_transition(s, a, r, s_)  # 存储样本
                episode_reward_sum += r   # 逐步加上一个episode内每个step的reward 观测值
                s = s_  # 更新状态
                DQN_ROS_Service.updateTrainStatus(e, 1, int(episode_reward_sum), goal_count, MAX_Epoch, 2)  # 最后是个标志位

                now = time.time()
                temp = now - start
                start = now

                # 指定位置，index是否更新
                newIndex = env.getIndex()
                if newIndex != currentTarget:
                    # 立刻返回当前要去的目标 速度上可能还需要调整
                    DQN_ROS_Service.updateVehicleAction(twist=Twist(), heading=0, distance=0,
                                                        index=newIndex)
                    break

                if done or env.get_goalbox:

                    reward_list.append(r)
                    env.updateIndex()

                    break

    # episodes_list = list(range(len(reward_list)))
    # plt.plot(episodes_list, reward_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('DQN Returns')
    # plt.show()

    except Exception as e:
        LogUtil.error(e)
    pass