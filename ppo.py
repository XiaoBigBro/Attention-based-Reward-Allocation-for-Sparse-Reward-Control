from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import torch.optim as optim


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
    """Gaussian 策略网络，输出 mean 和 log_std"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int],
                 log_std_min: float, log_std_max: float):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 2.0)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class Critic(nn.Module):
    """状态价值网络 V(s)"""
    def __init__(self, obs_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.net = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.net(obs))


class PPONetwork(nn.Module):
    """Actor-Critic 整合网络，含 target_critic 软更新"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int],
                 log_std_min: float, log_std_max: float):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = Actor(obs_dim, action_dim, hidden_dims, log_std_min, log_std_max)
        self.critic = Critic(obs_dim, hidden_dims)
        self.target_critic = Critic(obs_dim, hidden_dims)
        self.target_critic.load_state_dict(self.critic.state_dict())

#软更新
    def _soft_update_target(self, tau: float = 0.005):
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

#获取动作
    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作及相关信息"""
        mean, log_std = self.actor(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs)
        return action, log_prob, value, mean, log_std

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_target_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.target_critic(obs)


# ============================================================================
# 缓冲区
# ============================================================================

@dataclass
class RawTrajectoryBuffer:
    """存储一个完整回合的原始数据: obs, action, mean, log_std, log_prob, value, terminal_reward"""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    means: List[np.ndarray] = field(default_factory=list)      # 动作分布均值
    log_stds: List[np.ndarray] = field(default_factory=list)   # 动作分布对数标准差
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    terminal_reward: float = 0.0

    def add(self, obs, action, mean, log_std, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.means.append(mean)
        self.log_stds.append(log_std)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def finalize(self, terminal_reward: float):
        self.terminal_reward = terminal_reward

    def get_action_dist(self) -> np.ndarray:
        """返回 concat(mean, log_std) 形状 [T, action_dim * 2]"""
        return np.concatenate([self.means, self.log_stds], axis=-1)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.means.clear()
        self.log_stds.clear()
        self.log_probs.clear()
        self.values.clear()
        self.terminal_reward = 0.0

    def __len__(self) -> int:
        return len(self.observations)


@dataclass
class TrainingBuffer:
    """存储训练数据: s_t, a_t, r_t, s_{t+1}, log_prob, advantage, returns"""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    old_log_probs: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.next_states.clear()
        self.rewards.clear()
        self.advantages.clear()
        self.returns.clear()
        self.old_log_probs.clear()

    def to_tensors(self, device: str = "cuda") -> Dict[str, torch.Tensor]:
        return {
            "states": torch.FloatTensor(np.array(self.states)).to(device),
            "actions": torch.FloatTensor(np.array(self.actions)).to(device),
            "next_states": torch.FloatTensor(np.array(self.next_states)).to(device),
            "rewards": torch.FloatTensor(self.rewards).to(device),
            "advantages": torch.FloatTensor(self.advantages).to(device),
            "returns": torch.FloatTensor(self.returns).to(device),
            "old_log_probs": torch.FloatTensor(self.old_log_probs).to(device),
        }


# ============================================================================
# GAE 优势估计
# ============================================================================

def compute_gae(
    rewards: List[float],
    values: List[float],
    next_values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    A_t = δ_t + γλ * A_{t+1}
    G_t = A_t + V(s_t)
    """
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T

    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1 - int(dones[t])) - values[t]
        gae = delta + gamma * lam * (1 - int(dones[t])) * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


# ============================================================================
# PPO 损失
# ============================================================================

def compute_ppo_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                     advantages: torch.Tensor, clip_eps: float = 0.2) -> torch.Tensor:
    """L^{CLIP}(θ) = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]"""
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    return -torch.min(ratio * advantages, clipped * advantages).mean()


def compute_value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Huber 损失"""
    return F.smooth_l1_loss(values.squeeze(-1), returns)


def compute_entropy_loss(log_std: torch.Tensor) -> torch.Tensor:
    """Gaussian 策略熵 H = 0.5 * log(2πe * σ²)"""
    return -0.5 * (1 + 2 * log_std + np.log(2 * np.pi)).sum()


# ============================================================================
# 采集轨迹
# ============================================================================

def collect_trajectory(env, policy_fn, device: str = "cuda", buffer_size: int = 200) -> RawTrajectoryBuffer:
    """采集一个完整回合的数据"""
    buffer = RawTrajectoryBuffer()
    obs, info = env.reset()
    obs = unpack_obs(obs)

    done = False
    while not done:
        action, log_prob, value = policy_fn(obs)

        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = action

        next_obs, _, terminated, truncated, info = env.step(action_np)
        next_obs = unpack_obs(next_obs)
        buffer.add(obs, action_np, log_prob, value)
        done = truncated or (len(buffer) >= buffer_size)
        obs = next_obs

    # 存储最后一个 terminal 状态（用于奖励分配网络的错位对齐）
    buffer.observations.append(obs)
    # 添加零填充的 terminal action（保持 states 和 actions 长度一致）
    buffer.actions.append(np.zeros_like(buffer.actions[-1]))

    buffer.finalize(info.get("terminal_reward", 0.0))
    return buffer


# ============================================================================
# 数据准备
# ============================================================================

def prepare_for_training(
    raw_buffer: RawTrajectoryBuffer,
    network: PPONetwork,
    allocated_rewards: List[float],
    device: str = "cuda",
    gamma: float = 0.99,
    lam: float = 0.95,
) -> TrainingBuffer:
    """原始轨迹 → 训练数据（含 GAE）"""
    T = len(raw_buffer)
    if T == 0:
        return TrainingBuffer()

    observations = raw_buffer.observations
    obs_tensor = torch.FloatTensor(np.array(observations)).to(device)

    with torch.no_grad():
        critic_values = network.get_value(obs_tensor).squeeze(-1).cpu().numpy().tolist()
        target_values = network.get_target_value(obs_tensor).squeeze(-1).cpu().numpy().tolist()

    values = critic_values  # 长度 T+1 (包含 terminal 状态的价值)
    next_values = target_values[1:] + [0.0]  # 长度 T+1
    dones = [False] * T
    dones[-1] = True
    # allocated_rewards 长度 T+1，但 GAE 只用前 T 个
    advantages, returns = compute_gae(allocated_rewards[:-1], values, next_values, dones, gamma, lam)

    train_buffer = TrainingBuffer()
    for i in range(T):
        train_buffer.states.append(observations[i])
        train_buffer.actions.append(raw_buffer.actions[i])
        train_buffer.rewards.append(allocated_rewards[i])
        train_buffer.advantages.append(advantages[i])
        train_buffer.returns.append(returns[i])
        train_buffer.old_log_probs.append(raw_buffer.log_probs[i])
        train_buffer.next_states.append(observations[i + 1] if i < T - 1 else observations[i])

    return train_buffer
