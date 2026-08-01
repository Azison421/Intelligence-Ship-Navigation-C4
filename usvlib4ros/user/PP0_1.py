"""
PPO (Proximal Policy Optimization) Algorithm for USV Navigation.
install pytorch: pip install torch torchvision torchaudio
"""

import logging
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RolloutBuffer:
    """经验回放缓冲区，存储PPO训练所需的转移数据。"""

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        """清空缓冲区。"""
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()


class ActorCritic(nn.Module):
    """Actor-Critic网络：Actor输出策略分布，Critic估计状态值。"""

    def __init__(self, state_dim: int, action_dim: int,
                 has_continuous_action_space: bool, action_std_init: float):
        super().__init__()
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

    @staticmethod
    def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
        """将NaN/Inf替换为安全值。"""
        return torch.where(
            torch.isnan(t) | torch.isinf(t), torch.zeros_like(t), t
        )

    def act(self, state: torch.Tensor):
        """根据状态采样动作，返回(action, log_prob)。"""
        state = self._sanitize_tensor(state.to(device))

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = torch.clamp(self.actor(state), min=1e-6, max=1 - 1e-6)
            action_probs = self._sanitize_tensor(action_probs)
            # 归一化确保合法概率分布
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            dist = Categorical(action_probs)

        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """评估给定(state, action)对的(log_prob, state_value, entropy)。"""
        state = self._sanitize_tensor(state.to(device))

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = torch.clamp(self.actor(state), min=1e-6, max=1 - 1e-6)
            action_probs = self._sanitize_tensor(action_probs)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            dist = Categorical(action_probs)

        return dist.log_prob(action), self.critic(state), dist.entropy()


class PPO:
    """PPO算法实现。"""

    def __init__(self, state_dim: int, action_dim: int,
                 lr_actor: float, lr_critic: float,
                 gamma: float, K_epochs: int, eps_clip: float,
                 has_continuous_action_space: bool, action_std_init: float):
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
        self.mse_loss = nn.MSELoss()

    def select_action(self, state) -> int:
        """选择动作并记录到buffer，返回标量action值。"""
        state = state.to(device)
        with torch.no_grad():
            action, logprob = self.policy_old.act(state)
        # 存储为list以兼容JSON序列化
        self.buffer.states.append(state.cpu().numpy().tolist())
        self.buffer.actions.append(action.cpu().numpy().tolist())
        self.buffer.logprobs.append(logprob.cpu().numpy().tolist())
        return action.item()

    def update(self):
        """使用收集的经验更新策略网络。"""
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack([
            torch.tensor(s, dtype=torch.float32) for s in self.buffer.states
        ], dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack([
            torch.tensor(a, dtype=torch.float32) for a in self.buffer.actions
        ], dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack([
            torch.tensor(lp, dtype=torch.float32) for lp in self.buffer.logprobs
        ], dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            if state_values.dim() == 0:
                state_values = state_values.unsqueeze(0)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def load(self, checkpoint_path: str):
        """加载模型权重。"""
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))

    def save(self, checkpoint_path: str):
        """保存模型权重。"""
        torch.save(self.policy.state_dict(), checkpoint_path)
