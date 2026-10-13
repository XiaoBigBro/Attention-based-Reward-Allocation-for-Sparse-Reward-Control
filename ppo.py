from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple


# ============================================================================
# 工具函数
# ============================================================================

def unpack_obs(obs: dict) -> np.ndarray:
    """gymnasium dict obs → 1D numpy"""
    components = [
        obs.get("observation", []),
        obs.get("achieved_goal", []),
        obs.get("desired_goal", []),
    ]
    return np.concatenate([c.flatten() for c in components if len(c) > 0])

# ============================================================================
# 网络
# ============================================================================
class Actor(nn.Module):
    """Gaussian 策略网络，输出 tanh 压缩的动作和 log_std"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int],
                 log_std_min: float, log_std_max: float, log_std_init: float):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 tanh 动作、原始动作的 log_prob、log_std,采样用"""
        features = self.net(obs)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        # 采样
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()  # 原始动作
        action = torch.tanh(raw_action)  # tanh 压缩到 [-1, 1]

        # 修正 log_prob（补偿 tanh 压缩带来的雅可比行列式）
        # log π(a|o) = log π_raw(u|o) - Σ log(1 - tanh²(u) + ε)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        action_safe = torch.clamp(action, -0.999, 0.999)
        log_prob -= (1.0 - action_safe.pow(2) + 1e-6).log().sum(dim=-1)

        return action, log_prob, log_std, mean

    def get_log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """给定动作，计算 log_prob（用于训练）"""
        features = self.net(obs)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        # 反 tanh：从 action 得到 raw_action
        # tanh(u) = a => u = atanh(a) = 0.5 * ln((1+a)/(1-a))
        # 使用 log1p 公式，数值更稳定
        actions_safe = torch.clamp(actions, -0.999, 0.999)
        raw_actions = 0.5 * (torch.log1p(actions_safe) - torch.log1p(-actions_safe))

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        log_prob -= (1.0 - actions_safe.pow(2) + 1e-6).log().sum(dim=-1)

        return log_prob

class Critic(nn.Module):
    """状态价值网络 V(s)"""
    def __init__(self, obs_dim: int):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

class PPONetwork(nn.Module):
    """Actor-Critic 整合网络，含 target_critic 软更新"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int],
                 log_std_min: float, log_std_max: float, log_std_init: float):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = Actor(obs_dim, action_dim, hidden_dims, log_std_min, log_std_max, log_std_init)
        self.critic = Critic(obs_dim)
        self.target_critic = Critic(obs_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

    #软更新
    def _soft_update_target(self, tau: float):
        # 一次性更新所有参数
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.copy_(tau * param + (1 - tau) * target_param)

    #获取动作
    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作及相关信息（tanh 压缩后）"""
        action, log_prob, log_std, mean= self.actor(obs)
        return action, log_prob, log_std, mean

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_target_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.target_critic(obs)

# ============================================================================
# 缓冲区
# ============================================================================

@dataclass
class RawTrajectoryBuffer:
    """
    存储一个完整回合的原始数据。

    长度约定（episode 共 N 个 transition，observations 共 N+1 个）:
        observations: N + 1  (s_0 ... s_N，包含 terminal state)
        actions:      N + 1  (shifted: [zeros] + [a_0 ... a_{N-1}])
        means:        N + 1
        log_stds:     N + 1
        log_probs:    N      (每个 transition 的 log π(a_t | s_t))
        values:       N      (每个 transition 的 V(s_t))
        terminal_reward: 标量
        step_rewards: List[float] = field(default_factory=list)#每个步骤的奖励
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    means: List[np.ndarray] = field(default_factory=list)
    log_stds: List[np.ndarray] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    terminal_reward: float = 0.0
    step_rewards: List[float] = field(default_factory=list)

    def add(self, obs, action, mean, log_std, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.means.append(mean)
        self.log_stds.append(log_std)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def finalize(self, terminal_reward: float, step_rewards: List[float]):
        self.terminal_reward = terminal_reward
        self.step_rewards = step_rewards

    def get_action_dist(self) -> np.ndarray:
        """返回 concat(mean, log_std)，形状 [N+1, action_dim * 2]（shifted）"""
        return np.concatenate([self.means, self.log_stds], axis=-1)



    def __len__(self) -> int:
        return len(self.observations)


@dataclass
class TrainingBuffer:
    """
    存储一个 episode 的 N 个 transition 的训练数据。

    长度约定: states/actions/next_states/rewards/advantages/returns/old_log_probs 全部为 N
    """
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    old_log_probs: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)


# ============================================================================
# GAE 优势估计
# ============================================================================

def compute_gae(
    rewards: List[float],
    values: List[float],
    next_values: List[float],
    dones: List[bool],
    gamma: float ,
    lam: float ,
) -> Tuple[List[float], List[float]]:
    """
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    A_t = δ_t + γλ * A_{t+1}
    G_t = A_t + V(s_t)
    所有输入长度必须一致: rewards, values, next_values, dones
    """
    T = len(rewards)
    assert len(values) == T, f"values length {len(values)} != rewards length {T}"
    assert len(next_values) == T, f"next_values length {len(next_values)} != rewards length {T}"
    assert len(dones) == T, f"dones length {len(dones)} != rewards length {T}"

    advantages = [0.0] * T
    returns = [0.0] * T

    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1 - int(dones[t])) - values[t]
        gae = delta + gamma * lam * (1 - int(dones[t])) * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns
