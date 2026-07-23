"""
PPO + 奖励分配网络 + 自适应课程学习 完整训练脚本
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from curriculum import AdaptiveCurriculumManager
from fetch_env import make_fetch_env, get_env_obs_dim, get_env_action_dim
from ppo import (
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

@dataclass
class TrainingConfig:
    """训练配置：包含所有可调超参数，供 PPOTrainerWithAllocator 使用"""
    # 环境
    env_id: str = "FetchReach-v4"
    seed: int = 42

    # PPO 网络
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])  # 策略网络隐藏层维度
    log_std_min: float = -2.0#策略网络log标准差最小值
    log_std_max: float = 3.0#策略网络log标准差最大值
    log_std_init: float = 2.0#策略网络log标准差初始值

    # PPO 训练
    gamma: float = 0.99#折扣因子
    lam: float = 0.98#GAE参数
    clip_eps: float = 0.2#PPO裁剪参数
    entropy_coef: float = 1.0  # 熵正则化系数
    actor_lr: float = 3e-4  # 策略网络学习率
    critic_lr: float = 1e-4  # 价值网络学习率
    ppo_epochs: int = 6#PPO训练epochs
    critic_updates_per_epoch: int = 2#价值网络更新次数
    max_grad_norm: float = 0.5#最大梯度范数，防止梯度爆炸
    tau: float = 0.01#目标网络更新参数

    # 奖励分配网络
    alloc_d_model: int = 128#奖励分配网络隐藏层维度
    alloc_num_heads: int = 4#奖励分配网络头数
    alloc_num_layers: int = 2#奖励分配网络层数
    alloc_d_ff: int = 256#奖励分配网络前馈网络隐藏层维度
    alloc_dropout: float = 0.1#奖励分配网络dropout率
    alloc_entropy_coef: float = 0.001#奖励分配网络熵正则化系数
    alloc_lr: float = 3e-4#奖励分配网络学习率
    max_seq_len: int = 200#奖励分配网络最大序列长度

    # 课程学习
    curriculum_window_size: int = 50#课程学习窗口大小
    curriculum_promote_threshold: float = 200.0#课程学习提升阈值
    curriculum_demote_threshold: float = 20.0#课程学习降级阈值
    curriculum_promote_cooldown: int = 150#课程学习提升冷却期
    curriculum_demote_cooldown: int = 50#课程学习降级冷却期
    curriculum_replay_interval: int = 100#课程学习重放间隔

    # 训练流程
    total_episodes: int = 100000
    log_interval: int = 10#日志间隔
    save_interval: int = 2000#保存间隔
    save_dir: str = "./checkpoints/seed{seed}"#保存目录，只是普通的字符串
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 设备（自动检测）


@dataclass
class TrainingMetrics:
    """训练指标：记录每个 episode 的各项损失和奖励，用于日志和绘图"""
    # 基础信息
    episode: int = 0#回合数
    episode_reward: float = 0.0#回合奖励
    terminal_reward: float = 0.0#回合结束奖励
    episode_length: int = 0#回合长度

    # PPO 损失
    policy_loss: float = 0.0#策略网络损失
    value_loss: float = 0.0#价值网络损失
    entropy_loss: float = 0.0#熵正则化损失
    ppo_loss: float = 0.0#PPO损失

    # 分配网络损失
    alloc_td_loss: float = 0.0#分配网络TD损失
    alloc_entropy_loss: float = 0.0#分配网络熵正则化损失
    alloc_total_loss: float = 0.0#分配网络总损失

    # 课程学习状态
    curriculum_level: int = 0#课程学习等级
    curriculum_changed: str = ""#课程学习变化
    learning_rate: float = 0.0#学习率


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

        # 只有显式调用 .format() 时，才会被替换，否则只是普通的字符串
        save_dir = config.save_dir.format(seed=config.seed)
        self.save_dir = save_dir

        # 如果文件夹已存在，删除旧内容以确保干净的训练状态
        if os.path.exists(save_dir):
            import shutil
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        # 奖励记录文件（按种子区分）
        self.rewards_file = f"{self.save_dir}/training_rewards.txt"
        with open(self.rewards_file, 'w') as f:
            f.write("episode,terminal_reward,curriculum_level\n")

        # 创建设备
        self.device = torch.device(config.device)
        print(f"设备: device={self.device}")
        
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
            log_std_init=config.log_std_init,
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
            max_seq_len=config.max_seq_len,
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
        # 全局训练计数器
        self.episode_count = 0#回合数计数器

    def _policy_fn(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        """策略函数，用于采集数据"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value, mean, log_std = self.network.get_action(obs_t)

        return (
            action.squeeze(0).cpu().numpy(),
            mean.squeeze(0).cpu().numpy(),
            log_std.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def get_entropy_coef(self) -> float:
        """获取熵系数（固定值）"""
        return self.config.entropy_coef

    def get_lr(self) -> Tuple[float, float]:
        """获取学习率（固定值）"""
        return (self.config.actor_lr, self.config.critic_lr)

    def collect_trajectory(self) -> RawTrajectoryBuffer:
        """采集一个回合的轨迹"""
        buffer = RawTrajectoryBuffer()

        obs, info = self.env.reset()
        obs = unpack_obs(obs)

        done = False
        while not done:
            # 获取动作
            action, mean, log_std, log_prob, value = self._policy_fn(obs)

            # 与环境交互
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs = unpack_obs(next_obs)

            # 存储
            buffer.add(obs, action, mean, log_std, log_prob, value)

            # 检查结束
            done = truncated
            obs = next_obs

        # 存储最后一个 terminal 状态（用于奖励分配网络的错位对齐）
        buffer.observations.append(obs) #存储最后一个 terminal 状态长度为T+1
        #错位对齐
        buffer.actions.insert(0, np.zeros_like(buffer.actions[0]))#插入一个零动作，长度为T+1
        buffer.means.insert(0, np.zeros_like(buffer.means[0]))#插入一个零均值，长度为T+1
        buffer.log_stds.insert(0, np.zeros_like(buffer.log_stds[0]))#插入一个零对数标准差，长度为T+1

        # 存储最终奖励
        terminal_reward = info.get("terminal_reward", 0.0)
        buffer.finalize(terminal_reward)

        return buffer

    def allocate_rewards(self, buffer: RawTrajectoryBuffer) -> np.ndarray:
        """使用奖励分配网络分配奖励"""
        T = len(buffer)  # 长度为 N+1
        if T == 0:
            return np.array([])

        # 转为数组 (序列长度固定，不需要 padding)
        obs_padded = np.array(buffer.observations, dtype=np.float32)
        act_padded = np.array(buffer.actions, dtype=np.float32)

        # 分配奖励（不传 action_dist，保持训练/推理一致）
        weights, metrics = self.allocator_agent.allocate(
            states=obs_padded,
            actions=act_padded,
            actions_dist=buffer.get_action_dist(),
        )

        # 计算分配后的奖励
        terminal_reward = buffer.terminal_reward
        allocated_rewards = weights * terminal_reward

        return allocated_rewards,weights#长度为N

    def compute_gae_and_train_buffer(
        self,
        buffer: RawTrajectoryBuffer,#长度为N+1
        allocated_rewards: np.ndarray,#长度为N
        weights: np.ndarray,#长度为N
    ) -> TrainingBuffer:
        """计算 GAE 并构建训练缓冲区"""
        # 实际 transition 数 N (observations 有 N+1 个，包含 terminal state)
        N = len(buffer.observations) - 1#长度为N
        if N == 0:
            return TrainingBuffer()

        observations = buffer.observations  # 长度 N+1: s_0, ..., s_N

        # 重新计算价值（使用 observations 序列）
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        with torch.no_grad():
            critic_values = self.network.get_value(obs_tensor).squeeze(-1).cpu().numpy().tolist()
            target_values = self.network.get_target_value(obs_tensor).squeeze(-1).cpu().numpy().tolist()

        values = critic_values[:-1]  # 长度 N，取出V(s_0), V(s_1), ..., V(s_{N-1})
        next_values = target_values[1:]  # 长度 N，取出V(s_1), V(s_2), ..., V(s_{N})
        dones = [False] * (N - 1) + [True]  # 长度 N，最后一步是 terminal

        # 计算 GAE（所有输入长度必须一致为 N）
        advantages, returns = compute_gae(
            allocated_rewards.tolist(),  # 长度 N
            values,  # 长度 N
            next_values,  # 长度 N
            dones,  # 长度 N
            self.config.gamma,
            self.config.lam,
        )

        # 构建训练缓冲区（长度为 N）
        train_buffer = TrainingBuffer()
        train_buffer.terminal_reward = buffer.terminal_reward
        train_buffer.weights = weights
        for i in range(N):
            train_buffer.states.append(observations[i])  # s_i
            train_buffer.actions.append(buffer.actions[i + 1])  # a_i (shifted)
            train_buffer.rewards.append(allocated_rewards[i])  # r_i
            train_buffer.advantages.append(advantages[i])
            train_buffer.returns.append(returns[i])
            train_buffer.old_log_probs.append(buffer.log_probs[i])  # log π(a_i|s_i)
            train_buffer.next_states.append(observations[i + 1])  # s_{i+1}

        # 长度一致性检查
        train_buffer._check_consistency()

        return train_buffer

    def train_ppo_step(self, train_buffer: TrainingBuffer) -> Dict[str, float]:
        """训练 PPO 网络：策略网络更新 ppo_epochs 次，价值网络更新 critic_updates_per_epoch * ppo_epochs 次"""
        if len(train_buffer) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0, "ppo_loss": 0}

        # 获取当前的熵系数
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
            entropy = dist.entropy().sum(dim=-1).mean()#熵损失

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -current_entropy_coef * entropy#熵损失
            actor_loss = policy_loss + entropy_loss

            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.config.max_grad_norm)
            self.optimizer_actor.step()

            # ========== Critic 更新（多次） ==========
            for _ in range(self.config.critic_updates_per_epoch):
                self.optimizer_critic.zero_grad()
                values = self.network.get_value(states).squeeze(-1)
                value_loss = F.smooth_l1_loss(values, returns)
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
        train_buffer: TrainingBuffer,
        critic_network,
    ) -> Dict[str, float]:
        """训练奖励分配网络"""
        T = len(train_buffer)  # N 个 transition
        if T == 0:
            return {"alloc_td_loss": 0, "alloc_entropy_loss": 0, "alloc_total_loss": 0}

        # 构造 N+1 个 states: [s_0, ..., s_{N-1}] + [s_N]
        states_list = list(train_buffer.states) + [train_buffer.next_states[-1]]
        states_padded = np.array(states_list, dtype=np.float32)

        # 获取价值估计（长度为 N+1）
        obs_t = torch.FloatTensor(states_padded).to(self.device)
        with torch.no_grad():
            v_all = critic_network.get_value(obs_t).squeeze(-1)  # [N+1]: V(s_0)...V(s_N)
            v_target_all = critic_network.get_target_value(obs_t).squeeze(-1)  # [N+1]

        # allocator 前向输出 weights 为 N（position 1...N 对应 transition 0...N-1）
        # TD target 使用对应位置的 value: V(s_1)...V(s_N)
        v_online = v_all[:-1].cpu().numpy()  # [N]
        v_target_next = v_target_all[1:].cpu().numpy()  # [N]，最后一步 V(s_N) 用于 bootstrap

        # 训练分配网络
        metrics = self.allocator_agent.train_step(
            train_buffer=train_buffer,
            gamma=self.config.gamma,
            v_online=v_online,
            v_target_next=v_target_next,
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
        allocated_rewards,weights = self.allocate_rewards(raw_buffer)#长度为N

        # 3. 计算 GAE 并构建训练缓冲区
        train_buffer = self.compute_gae_and_train_buffer(raw_buffer, allocated_rewards,weights)

        # 4. 训练 PPO
        ppo_metrics = self.train_ppo_step(train_buffer)

        # 5. 训练分配网络
        alloc_metrics = self.train_allocator_step(train_buffer, self.network)

        # 6. 更新课程学习
        terminal_reward = raw_buffer.terminal_reward
        curriculum_changed = self.curriculum_manager.on_episode_end(terminal_reward)
        current_level = self.curriculum_manager.get_current_level()

        if curriculum_changed:
            self.env.set_curriculum_level(current_level)

        # 构建指标
        metrics = TrainingMetrics(
            episode=self.episode_count,
            episode_reward=sum(allocated_rewards) if len(allocated_rewards) > 0 else 0,
            terminal_reward=terminal_reward,
            episode_length=len(raw_buffer) - 1,  # observations 有 N+1 个，transition 数为 N
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
        episode_rewards = []#回合奖励
        terminal_rewards = []#最终奖励
        curriculum_levels = []#课程等级

        # 主循环，会在终端中显示一个动态进度条
        pbar = tqdm(total=self.config.total_episodes, desc="Training")
        start_time = time.time()

        trained_episodes = 0  # 有效训练回合数（不含回放）

        while trained_episodes < self.config.total_episodes:
            # 检查是否需要回放
            is_replay = self.curriculum_manager.should_replay()

            # 训练一个回合
            metrics = self.train_one_episode()

            # 只有非回放回合才计入有效训练
            if not is_replay:
                trained_episodes += 1

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

            # 日志（非回放时才打印，避免刷屏）
            if not is_replay and trained_episodes % self.config.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-self.config.log_interval:])
                current_level = self.curriculum_manager.get_current_level()
                window_mean = self.curriculum_manager.reward_window.mean() if self.curriculum_manager.reward_window.count() > 0 else 0.0
                current_entropy_coef = self.get_entropy_coef()

                # 获取当前 log_std（直接从 actor 参数）
                log_std_mean = self.network.actor.log_std.mean().item()

                pbar.write(f"[Episode {trained_episodes}]")
                pbar.write(f"  课程等级: {current_level}")
                pbar.write(f"  熵系数: {current_entropy_coef:.4f}, log_std: {log_std_mean:.4f}")
                pbar.write(f"  平均最终奖励 (最近{self.config.log_interval}): {avg_reward:.2f}")
                pbar.write(f"  滑动缓冲区均值: {window_mean:.2f}")
                pbar.write(f"  Actor损失: {metrics.policy_loss:.4f}")
                pbar.write(f"  Critic损失: {metrics.value_loss:.4f}")
                pbar.write(f"  分配网络损失: {metrics.alloc_total_loss:.4f}")

            # 课程升级时保存
            if metrics.curriculum_changed == "promote" and not is_replay:
                save_path = f"{self.save_dir}/level_up_{trained_episodes}.pt"
                self.save(save_path)
                pbar.write(f"*** 课程升级! 模型已保存: {save_path} ***")

            # 定时保存（非回放回合）
            if not is_replay and trained_episodes % self.config.save_interval == 0:
                save_path = f"{self.save_dir}/checkpoint_{trained_episodes}.pt"
                self.save(save_path)
                pbar.write(f"[Episode {trained_episodes}] 模型已保存: {save_path}")

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
# 2. 随机数种子设置
# ============================================================================

def set_seed(seed: int):
    """设置所有随机数种子"""
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# 2. 主函数
# ============================================================================

def main():
    """主函数"""
    # 创建配置
    config = TrainingConfig()

    # 设置随机数种子
    set_seed(config.seed)

    # 创建训练器
    trainer = PPOTrainerWithAllocator(config)

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
