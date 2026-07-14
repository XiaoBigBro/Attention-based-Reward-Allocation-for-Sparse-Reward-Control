"""
PPO + 奖励分配网络 + 自适应课程学习 完整训练脚本
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import ppo
import curriculum
import fetch_env
import reward_allocator
from curriculum import AdaptiveCurriculumManager
from fetch_env import make_fetch_env, EnvCurriculumParams, CURRICULUM_PRESETS, get_env_obs_dim, get_env_action_dim
from ppo import (
    Actor,
    Critic,
    PPONetwork,
    RawTrajectoryBuffer,
    TrainingBuffer,
    compute_gae,
    unpack_obs,
)
from reward_allocator import RewardAllocatorAgent, RewardAllocatorConfig


# =============================================================================
# 统一配置 - 所有可调参数集中于此
# =============================================================================

# 环境
ENV_ID = "FetchReach-v3"              # Gym 环境 ID
SEED = 123                             # 随机数种子（用于复现实验）

# PPO 网络
HIDDEN_DIMS = [256, 128]               # Actor/Critic 隐藏层维度
LOG_STD_MIN = -5.0                     # 策略标准差的对数下限（防止方差过小）
LOG_STD_MAX = 2.0                      # 策略标准差的对数上限（防止方差过大）

# PPO 训练
GAMMA = 0.99                           # 折扣因子（未来奖励的重要性）
LAM = 0.95                             # GAE 的 lambda（平衡偏差与方差）
CLIP_EPS = 0.2                         # PPO 裁剪范围（防止策略更新过大）
VALUE_COEF = 0.5                       # 价值损失系数
ENTROPY_COEF = 1.5                     # 熵系数初始值（鼓励探索）
ENTROPY_COEF_END = 0.1                 # 熵系数最终值（后期减少探索）
ENTROPY_DECAY_MID = 2000               # 熵系数衰减到 1.0 的 episode 数
ENTROPY_DECAY_FINAL = 12000            # 熵系数衰减到终值的 episode 数
ACTOR_LR = 5e-4                        # 策略网络学习率
CRITIC_LR = 1e-3                       # 价值网络学习率（通常比策略网络大）
LR_DECAY_FINAL = 80000                 # 学习率衰减到达终值的 episode 数
LR_END_RATIO = 0.1                     # 学习率衰减到的比例（衰减到初始值的 10%）
PPO_EPOCHS = 6                         # 每次更新的策略网络 epoch 数
CRITIC_UPDATES_PER_EPOCH = 2           # 每个 epoch 中价值网络更新次数（相对于策略网络）
BATCH_SIZE = 128                       # 每次更新的 mini-batch 大小
MAX_GRAD_NORM = 0.5                    # 梯度裁剪的最大范数
TAU = 0.005                            # 目标网络软更新系数

# 奖励分配网络
ALLOC_D_MODEL = 64                     # Transformer 隐层维度
ALLOC_NUM_HEADS = 4                    # 多头注意力头数
ALLOC_NUM_LAYERS = 2                   # Transformer 层数
ALLOC_D_FF = 256                       # 前馈网络隐层维度
ALLOC_DROPOUT = 0.1                    # Dropout 概率
ALLOC_ENTROPY_COEF = 0.01              # 分配网络熵系数
ALLOC_LR = 3e-4                        # 分配网络学习率

# 课程学习
CURRICULUM_WINDOW_SIZE = 50            # 计算平均奖励的窗口大小
PROMOTE_THRESHOLD = 200.0              # 升级阈值（平均奖励 > 此值则升级）
DEMOTE_THRESHOLD = 20.0                # 降级阈值（平均奖励 < 此值则降级）
PROMOTE_COOLDOWN = 150                 # 升级冷却期（防止频繁升级）
DEMOTE_COOLDOWN = 50                   # 降级冷却期（快速响应降级）
REPLAY_INTERVAL = 100                   # 回放间隔（每隔多少 episode 回放一次）

# 训练流程
TOTAL_EPISODES = 100000                # 总训练 episode 数
LOG_INTERVAL = 10                      # 日志打印间隔
SAVE_INTERVAL = 2000                   # 模型保存间隔
SAVE_DIR = "./checkpoints/seed{seed}"  # 模型保存目录（{seed} 会替换为实际种子）
DEVICE = "cuda:0"                      # 训练设备


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class TrainingConfig:
    """训练配置：包含所有可调超参数，供 PPOTrainerWithAllocator 使用"""
    # 环境
    env_id: str
    seed: int

    # PPO 网络
    hidden_dims: List[int]
    log_std_min: float
    log_std_max: float

    # PPO 训练
    gamma: float
    lam: float
    clip_eps: float
    value_coef: float
    entropy_coef: float
    entropy_coef_end: float
    entropy_decay_mid: int
    entropy_decay_final: int
    actor_lr: float
    critic_lr: float
    lr_decay_final: int
    lr_end_ratio: float
    ppo_epochs: int
    critic_updates_per_epoch: int
    batch_size: int
    max_grad_norm: float
    tau: float

    # 奖励分配网络
    alloc_d_model: int
    alloc_num_heads: int
    alloc_num_layers: int
    alloc_d_ff: int
    alloc_dropout: float
    alloc_entropy_coef: float
    alloc_lr: float

    # 课程学习
    curriculum_window_size: int
    curriculum_promote_threshold: float
    curriculum_demote_threshold: float
    curriculum_promote_cooldown: int
    curriculum_demote_cooldown: int
    curriculum_replay_interval: int

    # 训练流程
    total_episodes: int
    log_interval: int
    save_interval: int
    save_dir: str
    device: str


@dataclass
class TrainingMetrics:
    """训练指标：记录每个 episode 的各项损失和奖励，用于日志和绘图"""
    # 基础信息
    episode: int = 0
    episode_reward: float = 0.0
    terminal_reward: float = 0.0
    episode_length: int = 0

    # PPO 损失
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    ppo_loss: float = 0.0

    # 分配网络损失
    alloc_td_loss: float = 0.0
    alloc_entropy_loss: float = 0.0
    alloc_total_loss: float = 0.0

    # 课程学习状态
    curriculum_level: int = 0
    curriculum_changed: str = ""
    learning_rate: float = 0.0


# ============================================================================
# 1. 训练器
# ============================================================================

class PPOTrainerWithAllocator:
    """
    PPO 训练器（集成奖励分配网络和课程学习）

    训练流程:
        1. 采集轨迹
        2. 奖励分配
        3. 计算 GAE
        4. 训练 PPO + 分配网络
        5. 更新课程学习
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # 解析 save_dir 中的 {seed} 占位符
        save_dir = config.save_dir.format(seed=config.seed)
        self.save_dir = save_dir

        # 奖励记录文件（按种子区分）
        self.rewards_file = f"{self.save_dir}/training_rewards.txt"
        os.makedirs(save_dir, exist_ok=True)
        with open(self.rewards_file, 'w') as f:
            f.write("episode,terminal_reward,curriculum_level\n")

        # 熵系数衰减状态（每个课程等级独立衰减）
        self._entropy_coef = config.entropy_coef
        self._entropy_start_episode = 0  # 当前等级的起始回合
        self._last_curriculum_level = 0  # 上一次的课程等级

        # 创建设备
        self.device = torch.device(config.device)
        print(f"使用设备: {self.device}")

        # 自动查询环境维度（不创建渲染窗口）
        obs_dim = get_env_obs_dim(config.env_id)
        action_dim = get_env_action_dim(config.env_id)
        print(f"环境维度: obs={obs_dim}, action={action_dim}")

        print("\n[1/5] 初始化 PPO 网络...")
        self.network = PPONetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
        ).to(self.device)

        # 优化器（策略和价值网络分开设置学习率）
        self.optimizer_actor = optim.Adam(
            self.network.actor.parameters(),
            lr=config.actor_lr,
        )
        self.optimizer_critic = optim.Adam(
            self.network.critic.parameters(),
            lr=config.critic_lr,
        )

        # 学习率调度器 (暂时停用)

        print("[2/5] 初始化奖励分配网络...")
        alloc_config = RewardAllocatorConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=config.alloc_d_model,
            num_heads=config.alloc_num_heads,
            num_layers=config.alloc_num_layers,
            d_ff=config.alloc_d_ff,
            dropout=config.alloc_dropout,
            entropy_coef=config.alloc_entropy_coef,
            learning_rate=config.alloc_lr,
        )
        self.allocator_agent = RewardAllocatorAgent(alloc_config, self.device)

        print("[3/5] 初始化课程学习管理器...")
        self.curriculum_manager = AdaptiveCurriculumManager(
            window_size=config.curriculum_window_size,
            promote_threshold=config.curriculum_promote_threshold,
            demote_threshold=config.curriculum_demote_threshold,
            promote_cooldown=config.curriculum_promote_cooldown,
            demote_cooldown=config.curriculum_demote_cooldown,
            replay_interval=config.curriculum_replay_interval,
        )

        print("[4/5] 初始化环境...")
        self.env = make_fetch_env(
            env_id=config.env_id,
            curriculum_level=0,  # 从最简单的难度开始
        )

        print("[5/5] 准备完成!")
        # 全局训练计数器（用于学习率衰减、熵系数衰减）
        self.episode_count = 0
        self.step_count = 0  # 总步数统计

    def _policy_fn(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """策略函数，用于采集数据"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.network.get_action(obs_t)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def get_entropy_coef(self) -> float:
        """
        获取当前衰减后的熵系数（线性衰减）

        衰减曲线（基于配置中的实际值）：
        - 0 回合:              entropy = entropy_coef (1.5)
        - entropy_decay_mid:    entropy = 1.0 (中间值)
        - entropy_decay_final:  entropy = entropy_coef_end (0.1)
        - 每次课程升级重置
        """
        episodes_in_level = self.episode_count - self._entropy_start_episode

        if episodes_in_level <= 0:
            return self.config.entropy_coef

        if episodes_in_level >= self.config.entropy_decay_final:
            return self.config.entropy_coef_end

        mid_entropy = 1.0  # 中间锚点

        if episodes_in_level <= self.config.entropy_decay_mid:
            progress = episodes_in_level / self.config.entropy_decay_mid
            return self.config.entropy_coef + (mid_entropy - self.config.entropy_coef) * progress
        else:
            progress = (episodes_in_level - self.config.entropy_decay_mid) / \
                       (self.config.entropy_decay_final - self.config.entropy_decay_mid)
            return mid_entropy + (self.config.entropy_coef_end - mid_entropy) * progress

    def get_lr(self) -> Tuple[float, float]:
        """
        获取当前衰减后的学习率（线性衰减，不重置）

        衰减曲线：
        - 0 回合:              lr = initial_lr
        - lr_decay_final:      lr = initial_lr * lr_end_ratio
        - 课程升级时不重置（全局衰减）
        """
        if self.episode_count >= self.config.lr_decay_final:
            return (
                self.config.actor_lr * self.config.lr_end_ratio,
                self.config.critic_lr * self.config.lr_end_ratio,
            )

        progress = self.episode_count / self.config.lr_decay_final
        return (
            self.config.actor_lr * (1 - (1 - self.config.lr_end_ratio) * progress),
            self.config.critic_lr * (1 - (1 - self.config.lr_end_ratio) * progress),
        )

    def update_lr(self):
        """更新优化器的学习率"""
        actor_lr, critic_lr = self.get_lr()
        for param_group in self.optimizer_actor.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.optimizer_critic.param_groups:
            param_group['lr'] = critic_lr

    def collect_trajectory(self) -> RawTrajectoryBuffer:
        """采集一个回合的轨迹"""
        buffer = RawTrajectoryBuffer()

        obs, info = self.env.reset()
        obs = unpack_obs(obs)

        done = False
        while not done:
            # 获取动作
            action, log_prob, value = self._policy_fn(obs)

            # 与环境交互
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs = unpack_obs(next_obs)

            # 存储
            buffer.add(obs, action, log_prob, value)

            # 检查结束
            done = truncated
            obs = next_obs

        # 存储最终奖励
        terminal_reward = info.get("terminal_reward", 0.0)
        buffer.finalize(terminal_reward)

        return buffer

    def allocate_rewards(self, buffer: RawTrajectoryBuffer) -> np.ndarray:
        """使用奖励分配网络分配奖励"""
        T = len(buffer)
        if T == 0:
            return np.array([])

        # 转为数组 (序列长度固定，不需要 padding)
        obs_padded = np.array(buffer.observations, dtype=np.float32)
        act_padded = np.array(buffer.actions, dtype=np.float32)

        # 获取价值估计
        obs_t = torch.FloatTensor(obs_padded).to(self.device)
        with torch.no_grad():
            v_online = self.network.get_value(obs_t).squeeze(-1).cpu().numpy()
            v_target_next = self.network.get_target_value(obs_t).squeeze(-1).cpu().numpy()

        # 分配奖励
        weights, metrics = self.allocator_agent.allocate(
            states=obs_padded,
            actions=act_padded,
        )

        # 计算分配后的奖励
        terminal_reward = buffer.terminal_reward
        allocated_rewards = weights * terminal_reward

        return allocated_rewards

    def compute_gae_and_train_buffer(
        self,
        buffer: RawTrajectoryBuffer,
        allocated_rewards: np.ndarray,
    ) -> TrainingBuffer:
        """计算 GAE 并构建训练缓冲区"""
        T = len(buffer)
        if T == 0:
            return TrainingBuffer()

        observations = buffer.observations
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)

        with torch.no_grad():
            # Critic 的值
            critic_values = self.network.get_value(obs_tensor).squeeze(-1).cpu().numpy().tolist()
            # target_critic 的值
            target_values = self.network.get_target_value(obs_tensor).squeeze(-1).cpu().numpy().tolist()

        # 构建 values 列表
        values = critic_values + [0.0]  # V(s_1), ..., V(s_T), V(s_{T+1}=0)

        # 构建 next_values (使用 target_critic)
        next_values = target_values[1:] + [0.0]  # V_target(s_2), ..., V_target(s_T), V_target(s_{T+1}=0)

        # 所有 done 都是 False（FetchReach 强制跑满 T 步，用 truncated 判断回合结束）
        # 末步的价值用 0（terminal）替代（已通过 next_values 构建好）
        dones = [False] * T

        # 计算 GAE
        advantages, returns = compute_gae(
            allocated_rewards.tolist(),
            values,
            next_values,
            dones,
            self.config.gamma,
            self.config.lam,
        )

        # 构建训练缓冲区
        train_buffer = TrainingBuffer()

        for i in range(T):
            train_buffer.states.append(observations[i])
            train_buffer.actions.append(buffer.actions[i])
            train_buffer.rewards.append(allocated_rewards[i])
            train_buffer.advantages.append(advantages[i])
            train_buffer.returns.append(returns[i])
            train_buffer.old_log_probs.append(buffer.log_probs[i])

            if i < T - 1:
                train_buffer.next_states.append(observations[i + 1])
            else:
                train_buffer.next_states.append(observations[i])

        return train_buffer

    def train_ppo_step(self, train_buffer: TrainingBuffer) -> Dict[str, float]:
        """训练 PPO 网络：策略网络更新 ppo_epochs 次，价值网络更新 critic_updates_per_epoch * ppo_epochs 次"""
        if len(train_buffer) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0, "ppo_loss": 0}

        # 获取当前衰减后的熵系数
        current_entropy_coef = self.get_entropy_coef()

        # 转为 tensor
        states = torch.FloatTensor(np.array(train_buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(train_buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(train_buffer.old_log_probs).to(self.device)
        advantages = torch.FloatTensor(train_buffer.advantages).to(self.device)
        returns = torch.FloatTensor(train_buffer.returns).to(self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 累计多 epoch 的损失
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_critic_updates = 0

        # 策略网络更新 ppo_epochs 次，价值网络更新 critic_updates_per_epoch * ppo_epochs 次
        for _ in range(self.config.ppo_epochs):
            # ========== Actor 更新 ==========
            self.optimizer_actor.zero_grad()
            mean, log_std = self.network.actor(states)
            dist = torch.distributions.Normal(mean, log_std.exp())
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -current_entropy_coef * entropy
            actor_loss = policy_loss + entropy_loss

            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.config.max_grad_norm)
            self.optimizer_actor.step()

            # ========== Critic 更新（多次） ==========
            for _ in range(self.config.critic_updates_per_epoch):
                self.optimizer_critic.zero_grad()
                values = self.network.get_value(states).squeeze(-1)
                value_loss = F.mse_loss(values, returns)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.config.max_grad_norm)
                self.optimizer_critic.step()

                total_value_loss += value_loss.item()
                total_critic_updates += 1

            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item()

        # 软更新 target_critic
        self.network._soft_update_target(self.config.tau)

        n = self.config.ppo_epochs
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / total_critic_updates,
            "entropy_loss": total_entropy_loss / n,
            "ppo_loss": (total_policy_loss + total_entropy_loss) / n + total_value_loss / total_critic_updates,
        }

    def train_allocator_step(
        self,
        buffer: RawTrajectoryBuffer,
        allocated_rewards: np.ndarray,
    ) -> Dict[str, float]:
        """训练奖励分配网络"""
        T = len(buffer)
        if T == 0:
            return {"alloc_td_loss": 0, "alloc_entropy_loss": 0, "alloc_total_loss": 0}

        # 转为数组 (序列长度固定，不需要 padding)
        obs_padded = np.array(buffer.observations, dtype=np.float32)
        act_padded = np.array(buffer.actions, dtype=np.float32)

        # 获取价值估计
        obs_t = torch.FloatTensor(obs_padded).to(self.device)
        with torch.no_grad():
            v_online = self.network.get_value(obs_t).squeeze(-1)  # [T]
            v_target = self.network.get_target_value(obs_t).squeeze(-1)  # [T]
            # v_target_next 需要 [T+1]，在末尾添加终端状态价值 0
            v_target_next = torch.cat([v_target, torch.zeros(1).to(self.device)], dim=0)  # [T+1]

        # 训练分配网络
        metrics = self.allocator_agent.train_step(
            states=obs_padded,
            actions=act_padded,
            terminal_reward=buffer.terminal_reward,
            gamma=self.config.gamma,
            v_online=v_online.cpu().numpy(),
            v_target_next=v_target_next.cpu().numpy(),
        )

        return {
            "alloc_td_loss": metrics.get("td_loss", 0),
            "alloc_entropy_loss": metrics.get("entropy_loss", 0),
            "alloc_total_loss": metrics.get("total_loss", 0),
        }

    def train_one_episode(self) -> TrainingMetrics:
        """训练一个回合"""
        # 1. 采集轨迹
        raw_buffer = self.collect_trajectory()

        # 2. 奖励分配
        allocated_rewards = self.allocate_rewards(raw_buffer)

        # 3. 计算 GAE 并构建训练缓冲区
        train_buffer = self.compute_gae_and_train_buffer(raw_buffer, allocated_rewards)

        # 4. 训练 PPO
        ppo_metrics = self.train_ppo_step(train_buffer)

        # 5. 训练分配网络
        alloc_metrics = self.train_allocator_step(raw_buffer, allocated_rewards)

        # 6. 更新课程学习
        terminal_reward = raw_buffer.terminal_reward
        curriculum_changed = self.curriculum_manager.on_episode_end(terminal_reward)
        current_level = self.curriculum_manager.get_current_level()

        if curriculum_changed:
            self.env.set_curriculum_level(current_level)
            # 课程升级时重置熵系数衰减
            self._entropy_start_episode = self.episode_count
            self._last_curriculum_level = current_level

        # 7. 学习率衰减
        self.update_lr()

        # 构建指标
        metrics = TrainingMetrics(
            episode=self.episode_count,
            episode_reward=sum(allocated_rewards) if len(allocated_rewards) > 0 else 0,
            terminal_reward=terminal_reward,
            episode_length=len(raw_buffer),
            curriculum_level=current_level,
            curriculum_changed=curriculum_changed,
            learning_rate=self.get_lr()[0],
            **{**ppo_metrics, **alloc_metrics},
        )

        self.episode_count += 1

        # 每次回合结束都追加写入文件（课程切换时实时保存）
        with open(self.rewards_file, 'a') as f:
            f.write(f"{self.episode_count},{terminal_reward},{current_level}\n")

        return metrics

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        # 统计
        episode_rewards = []
        terminal_rewards = []
        curriculum_levels = []

        # 主循环
        pbar = tqdm(total=self.config.total_episodes, desc="Training")
        start_time = time.time()

        for episode in range(self.config.total_episodes):
            # 训练一个回合
            metrics = self.train_one_episode()

            # 记录
            episode_rewards.append(metrics.episode_reward)
            terminal_rewards.append(metrics.terminal_reward)
            curriculum_levels.append(metrics.curriculum_level)

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "Level": metrics.curriculum_level,
                "TR": f"{metrics.terminal_reward:.1f}",
                "ER": f"{metrics.episode_reward:.1f}",
                "PPO": f"{metrics.ppo_loss:.3f}",
                "Alloc": f"{metrics.alloc_total_loss:.3f}",
            })

            # 日志
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-self.config.log_interval:])
                current_level = self.curriculum_manager.get_current_level()
                is_replay = self.curriculum_manager.should_replay()
                replay_status = "是" if is_replay else "否"
                window_mean = self.curriculum_manager.reward_window.mean() if self.curriculum_manager.reward_window.count() > 0 else 0.0
                current_entropy_coef = self.get_entropy_coef()

                # 获取当前 log_std（直接从 actor 参数）
                log_std_mean = self.network.actor.log_std.mean().item()

                pbar.write(f"[Episode {episode + 1}]")
                pbar.write(f"  课程等级: {current_level}")
                pbar.write(f"  熵系数: {current_entropy_coef:.4f}, log_std: {log_std_mean:.4f}")
                pbar.write(f"  课程回放: {replay_status}")
                pbar.write(f"  平均最终奖励 (最近10): {avg_reward:.2f}")
                pbar.write(f"  滑动缓冲区均值: {window_mean:.2f}")
                pbar.write(f"  Actor损失: {metrics.policy_loss:.4f}")
                pbar.write(f"  Critic损失: {metrics.value_loss:.4f}")
                pbar.write(f"  分配网络损失: {metrics.alloc_total_loss:.4f}")

            # 课程升级时保存
            if metrics.curriculum_changed == "promote":
                save_path = f"{self.save_dir}/level_up_{episode + 1}.pt"
                self.save(save_path)
                pbar.write(f"*** 课程升级! 模型已保存: {save_path} ***")

            # 定时保存
            if (episode + 1) % self.config.save_interval == 0:
                save_path = f"{self.save_dir}/checkpoint_{episode + 1}.pt"
                self.save(save_path)
                pbar.write(f"[Episode {episode + 1}] 模型已保存: {save_path}")

        pbar.close()

        # 训练结束
        elapsed = time.time() - start_time
        print(f"\n训练完成! 耗时: {elapsed:.1f}秒")
        print(f"平均奖励: {np.mean(episode_rewards):.2f}")
        print(f"最高奖励: {np.max(episode_rewards):.2f}")
        print(f"奖励数据已保存到: {self.rewards_file}")

        # 保存最终模型
        self.save(f"{self.save_dir}/final_model.pt")

    def save(self, path: str):
        """保存模型"""
        torch.save({
            "ppo_network": self.network.state_dict(),
            "allocator": self.allocator_agent.network.state_dict(),
            "curriculum": self.curriculum_manager.get_stats(),
            "config": self.config,
            "episode_count": self.episode_count,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["ppo_network"])
        self.allocator_agent.network.load_state_dict(checkpoint["allocator"])
        self.episode_count = checkpoint["episode_count"]


# ============================================================================
# 2. 主函数
# ============================================================================

def main():
    """主函数"""
    # 设置随机数种子
    import random
    random.seed(SEED)
    import numpy as np
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("=" * 60)
    print("PPO + 奖励分配网络 + 自适应课程学习")
    print("=" * 60)

    # 配置
    config = TrainingConfig(
        env_id=ENV_ID,
        seed=SEED,
        hidden_dims=HIDDEN_DIMS,
        log_std_min=LOG_STD_MIN,
        log_std_max=LOG_STD_MAX,
        gamma=GAMMA,
        lam=LAM,
        clip_eps=CLIP_EPS,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        entropy_coef_end=ENTROPY_COEF_END,
        entropy_decay_mid=ENTROPY_DECAY_MID,
        entropy_decay_final=ENTROPY_DECAY_FINAL,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        lr_decay_final=LR_DECAY_FINAL,
        lr_end_ratio=LR_END_RATIO,
        ppo_epochs=PPO_EPOCHS,
        critic_updates_per_epoch=CRITIC_UPDATES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        max_grad_norm=MAX_GRAD_NORM,
        tau=TAU,
        alloc_d_model=ALLOC_D_MODEL,
        alloc_num_heads=ALLOC_NUM_HEADS,
        alloc_num_layers=ALLOC_NUM_LAYERS,
        alloc_d_ff=ALLOC_D_FF,
        alloc_dropout=ALLOC_DROPOUT,
        alloc_entropy_coef=ALLOC_ENTROPY_COEF,
        alloc_lr=ALLOC_LR,
        curriculum_window_size=CURRICULUM_WINDOW_SIZE,
        curriculum_promote_threshold=PROMOTE_THRESHOLD,
        curriculum_demote_threshold=DEMOTE_THRESHOLD,
        curriculum_promote_cooldown=PROMOTE_COOLDOWN,
        curriculum_demote_cooldown=DEMOTE_COOLDOWN,
        curriculum_replay_interval=REPLAY_INTERVAL,
        total_episodes=TOTAL_EPISODES,
        log_interval=LOG_INTERVAL,
        save_interval=SAVE_INTERVAL,
        save_dir=SAVE_DIR,
        device=DEVICE,
    )

    # 打印关键超参数
    print("\n" + "=" * 60)
    print("关键超参数")
    print("=" * 60)
    print(f"  PPO:")
    print(f"    策略网络学习率:       {config.actor_lr}")
    print(f"    价值网络学习率:       {config.critic_lr}")
    print(f"    学习率衰减:          {config.actor_lr} -> {config.actor_lr * config.lr_end_ratio:.2e} ({config.lr_decay_final} 回合)")
    print(f"    熵系数:              {config.entropy_coef}")
    print(f"    ppo_epochs:          {config.ppo_epochs}")
    print(f"    critic_updates:      {config.critic_updates_per_epoch} 次/epoch (共 {config.ppo_epochs * config.critic_updates_per_epoch} 次)")
    print(f"    batch_size:          {config.batch_size}")
    print(f"    gamma:               {config.gamma}")
    print(f"    lambda (GAE):        {config.lam}")
    print(f"    clip_eps:            {config.clip_eps}")
    print(f"    value_coef:          {config.value_coef}")
    print(f"    max_grad_norm:       {config.max_grad_norm}")
    print(f"    entropy_coef:        {config.entropy_coef} -> {config.entropy_coef_end} (衰减: {config.entropy_decay_mid}回合到1.0, {config.entropy_decay_final}回合到{config.entropy_coef_end})")
    print(f"  奖励分配网络:")
    print(f"    学习率:              {config.alloc_lr}")
    print(f"    熵系数:              {config.alloc_entropy_coef}")
    print(f"    d_model:             {config.alloc_d_model}")
    print(f"    num_heads:           {config.alloc_num_heads}")
    print(f"    num_layers:          {config.alloc_num_layers}")
    print(f"  课程学习:")
    print(f"    window_size:         {config.curriculum_window_size}")
    print(f"    promote_threshold:    {config.curriculum_promote_threshold}")
    print(f"    demote_threshold:    {config.curriculum_demote_threshold}")
    print(f"    promote_cooldown:    {config.curriculum_promote_cooldown}")
    print(f"    demote_cooldown:    {config.curriculum_demote_cooldown}")
    print(f"    replay_interval:    {config.curriculum_replay_interval}")
    print(f"  训练:")
    print(f"    total_episodes:      {config.total_episodes}")
    print(f"    seed:                {config.seed}")
    print("=" * 60)

    # 创建训练器
    trainer = PPOTrainerWithAllocator(config)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
