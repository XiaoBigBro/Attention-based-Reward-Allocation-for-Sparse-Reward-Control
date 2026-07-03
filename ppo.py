"""
PPO 算法主体实现
数据流: env.step() → RawTrajectoryBuffer → TrainingBuffer → 网络更新

=============================================================================
可配置超参数说明
=============================================================================

【网络结构】
- obs_dim, action_dim         : 输入状态维数、动作维数（在 PPONetwork 创建时指定）
- hidden_dims                 : Actor/Critic 共享隐藏层维度列表，默认 [256, 256]
- log_std_min, log_std_max    : 策略网络 log_std 的裁剪范围，防止 std 过大/过小

【GAE (Generalized Advantage Estimation)】
- gamma : 折扣因子，默认 0.99，越大越看重远期奖励
- lam   : GAE λ 参数，默认 0.95，越接近 1 越偏向蒙特卡洛估计，越接近 0 越偏向 TD 估计

【PPO 损失】
- clip_eps          : PPO 裁剪范围，默认 0.2，越大允许策略更新幅度越大
- value_coef        : 价值损失权重，默认 0.5
- entropy_coef      : 熵正则项系数，默认 0.01，越大鼓励更多探索
- max_grad_norm     : 梯度裁剪阈值，默认 0.5，防止梯度爆炸

【训练过程】
- ppo_epochs        : 每个回合数据重复训练次数，默认 4
- buffer_size       : 采集轨迹的最大步数，默认 200
- tau               : 目标网络软更新系数，默认 0.005，越小更新越慢越稳定

【其他】
- device            : 计算设备，默认 "cuda"，可改为 "cpu"

【熵系数衰减】
熵系数的退火逻辑在 train_ppo_allocator.py 的 PPOTrainerWithAllocator.get_entropy_coef() 中实现：
- 线性两阶段衰减：1.5 -> 1.0 (2000回合) -> 0.1 (12000回合)
- 每次课程升级时重置
"""

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

    def _soft_update_target(self, tau: float = 0.005):
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.actor(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs)
        return action, log_prob, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_target_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.target_critic(obs)


# ============================================================================
# 缓冲区
# ============================================================================

@dataclass
class RawTrajectoryBuffer:
    """存储一个完整回合的原始数据: obs, action, log_prob, value, terminal_reward"""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    terminal_reward: float = 0.0

    def add(self, obs, action, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def finalize(self, terminal_reward: float):
        self.terminal_reward = terminal_reward

    def clear(self):
        self.observations.clear()
        self.actions.clear()
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

    values = critic_values + [0.0]
    next_values = target_values[1:] + [0.0]
    dones = [False] * T
    dones[-1] = True

    advantages, returns = compute_gae(allocated_rewards, values, next_values, dones, gamma, lam)

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


# ============================================================================
# 策略包装器
# ============================================================================

class PolicyWrapper:
    """将 PPONetwork 转换为采集函数"""
    def __init__(self, network: PPONetwork, device: str = "cuda"):
        self.network = network
        self.device = device

    def __call__(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(obs_t)
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()


# ============================================================================
# 训练函数
# ============================================================================

def train_one_episode(
    network: PPONetwork,
    optimizer_actor: optim.Optimizer,
    optimizer_critic: optim.Optimizer,
    train_buffer: TrainingBuffer,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    tau: float = 0.005,
    device: str = "cuda",
) -> Dict[str, float]:
    """训练一个回合的数据"""
    batch = train_buffer.to_tensors(device)
    states = batch["states"]
    actions = batch["actions"]
    advantages = (batch["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + 1e-8)
    returns = batch["returns"]
    old_log_probs = batch["old_log_probs"]

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

    for _ in range(ppo_epochs):
        # Actor 更新
        optimizer_actor.zero_grad()
        _, new_log_probs, _ = network.get_action(states)
        _, log_std = network.actor(states)
        policy_loss = compute_ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps)
        entropy_loss = compute_entropy_loss(log_std)
        actor_loss = (policy_loss - entropy_coef * entropy_loss) / ppo_epochs
        actor_loss.backward()
        nn.utils.clip_grad_norm_(network.actor.parameters(), max_grad_norm)
        optimizer_actor.step()

        # Critic 更新
        optimizer_critic.zero_grad()
        values = network.get_value(states)
        value_loss = compute_value_loss(values, returns) * value_coef / ppo_epochs
        value_loss.backward()
        nn.utils.clip_grad_norm_(network.critic.parameters(), max_grad_norm)
        optimizer_critic.step()

        metrics["policy_loss"] += policy_loss.item()
        metrics["value_loss"] += value_loss.item()
        metrics["entropy_loss"] += entropy_loss.item()
        metrics["total_loss"] += (policy_loss + value_loss).item()

    network._soft_update_target(tau=tau)

    for k in metrics:
        metrics[k] /= ppo_epochs

    return metrics
