"""
install pytorch in window10 cpu
python3.9 -m pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
"""

import sys
sys.path.append(r"D:\\318zero\\USV\\code\\usvlib4ros_project")
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, MultivariateNormal
from usvlib4ros.dqn.dqn_env import Env
from usvlib4ros.dqn.dqn_ros_service import DQN_ROS_Service
from usvlib4ros.usvRosUtil import LogUtil, USVRosbridgeClient

# Device setup
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

# Rollout Buffer
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1) if not has_continuous_action_space else nn.Identity()
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def act(self, state):
        state = state.to(device)
        # Check for NaN or Inf in state and handle it
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("Warning: State contains NaN or Inf values. Replacing with zeros.")
            state = torch.where(torch.isnan(state) | torch.isinf(state), torch.zeros_like(state), state)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            # Ensure action_probs is valid
            action_probs = torch.clamp(action_probs, min=1e-6, max=1 - 1e-6)  # Avoid zero, negative, or extreme values
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                print("Warning: Action probabilities contain NaN or Inf values. Replacing with uniform distribution.")
                action_probs = torch.ones_like(action_probs) / action_probs.size(0)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        print(f"Action probabilities: {action_probs}, Selected action: {action}")
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        state = state.to(device)
        print(f"Evaluating state input: {state}")  # Added print statement to debug state input
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            # Ensure action_probs is valid
            action_probs = torch.clamp(action_probs, min=1e-6, max=1 - 1e-6)  # Avoid zero, negative, or extreme values
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                print(f"Warning: Action probabilities contain invalid values: {action_probs}")
                action_probs = torch.ones_like(action_probs) / action_probs.size(0)  # Replace invalid probs with uniform
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        print(f"Evaluating state: {state}, Action: {action}, LogProbs: {action_logprobs}, State values: {state_values}, Action probs: {action_probs}")  # Updated print statement to include action_probs
        return action_logprobs, state_values, dist_entropy

# PPO Class
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init):
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = state.to(device)
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state.cpu().numpy().tolist())  # Convert Tensor to list for JSON serialization
        self.buffer.actions.append(action.cpu().numpy().tolist())
        self.buffer.logprobs.append(action_logprob.cpu().numpy().tolist())
        print(f"Selected action: {action}, LogProb: {action_logprob}")
        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack([torch.tensor(s, dtype=torch.float32) for s in self.buffer.states], dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack([torch.tensor(a, dtype=torch.float32) for a in self.buffer.actions], dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack([torch.tensor(lp, dtype=torch.float32) for lp in self.buffer.logprobs], dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            if state_values.dim() == 0:
                state_values = state_values.unsqueeze(0)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            print(f"Loss: {loss.mean().item()}, Ratios: {ratios}, Advantages: {advantages}")

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)  # Gradient clipping to prevent explosion
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)


class DQN_NAV:

    def __init__(self):

        pass

    def initNav(self):
        env = Env(action_size=N_ACTIONS, rosHost=USVRosbridgeClient.Host,
                  deviceId="27279d7a29f4fa6e95f630f94c76eb12f6fbb230")
        ppo_agent = PPO(N_STATES, N_ACTIONS, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP, False, ACTION_STD_INIT)
        pass



    pass

if __name__ == "__main__":
    try:
        # USVRosbridgeClient.Host = "121.41.106.238"
        # USVRosbridgeClient.Port = 8236
        USVRosbridgeClient.Host = "192.168.3.35"
        USVRosbridgeClient.Port = 9090
        #69b7a7d6ddf6c61f0af03f66a93d32a9723b23dd    27279d7a29f4fa6e95f630f94c76eb12f6fbb230
        env = Env(action_size=N_ACTIONS, rosHost=USVRosbridgeClient.Host,deviceId="27279d7a29f4fa6e95f630f94c76eb12f6fbb230")
        ppo_agent = PPO(N_STATES, N_ACTIONS, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP, False, ACTION_STD_INIT)

        # Load pretrained model
        # ppo_agent.load("./PPO_Ship_obstacle_zero_4.0w.pth")

        print("wait train button trigger ...")
        while env.isTrainActionTrigger() == 0:
            time.sleep(1)
            pass

        for e in range(MAX_EPOCH):
            if env.isTrainActionTrigger() == 0:
                print("Stop train ...")
                break

            print(f"train {e} ...")
            state = env.reset()
            max_distance = env.goal_distance
            print(f"Initial distance to goal: {max_distance}")  # Print initial distance to goal
            print(f"Raw LiDAR data on reset: {state}")  # Print raw LiDAR data before processing
            done = False
            episode_reward_sum = 0
            startTime = time.time()
            for t in range(3000):
                if env.isTrainActionTrigger() == 0:
                    print("Stop train step ...")
                    break
                tempTime = time.time() - startTime
                if tempTime > 10:
                    break;
                state = torch.FloatTensor(state).to(device)
                # print(f"State before action: {state}")
                action = ppo_agent.select_action(state)
                next_state, reward, done, max_distance = env.step(state.cpu().numpy().tolist(), action, max_distance)
                print(f"Distance to goal after step: {max_distance}")  # Print distance to goal after each step
                print(f"Raw LiDAR data after step: {next_state}")  # Print raw LiDAR data after each step

                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)
                episode_reward_sum += reward
                DQN_ROS_Service.updateTrainStatus(e, 1, int(episode_reward_sum), e, MAX_EPOCH, 2)  # 最后是个标志位

                print(f"Step: {t}, Action: {action}, Reward: {reward}, Done: {done}")

                if done:
                    print(f"Episode ended at step {t}, total reward: {episode_reward_sum}")
                    break

                if t % 2000 == 0:
                    ppo_agent.update()

                state = next_state

            if e % 100 == 0:
                checkpoint_path = f"./PPO_ship_obstacle_{e}.pth"
                # ppo_agent.save(checkpoint_path)

    except Exception as e:
        LogUtil.error(e)