"""
PPO + 奖励分配网络 + 自适应课程学习 完整训练脚本
"""

import os
import time
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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

    # PPO actor网络
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])  # 策略网络隐藏层维度
    log_std_min: float = -2.0#策略网络log标准差最小值
    log_std_max: float = 0.5#策略网络log标准差最大值
    log_std_init: float = 0.0#策略网络log标准差初始值，控制策略的初始探索程度。std=exp(log_std)实际标准差

    # PPO 训练
    '''
    gamma 控制的是“未来奖励的折现速度”。
    调小 gamma（比如 0.97）会让未来奖励衰减更快，
    实际上也是在“缩短”价值估计的依赖长度，也能起到一定的稳定作用。
    公式：G_t = r_t + gamma * G_{t+1} + gamma^2 * G_{t+2} + ... + gamma^n * G_{t+n}
    '''
    gamma: float = 0.99#折扣因子,控制未来奖励的权重

    '''
    决定了你信任多少步的 TD 误差。但是分配网络不稳定，也就导致TD误差不准确，
    所以lam越小，策略更新越稳定。
    公式：A_t = delta_t + gamma * lam * A_{t+1} + gamma^2 * lam^2 * A_{t+2} + ... + gamma^n * lam^n * A_{t+n}
    '''
    lam: float = 0.9#GAE参数

    clip_eps: float = 0.05#PPO裁剪参数,控制策略更新的稳定性.

    '''
    鼓励策略不要过早变“确定”。也就是说，
    即使网络想减小 log_std 来“利用”，entropy_coef 也会“拉它一把”，让它保留一定的随机性。
    log_std 是 “策略自己决定的探索力度”。
    entropy_coef 是 “外部强推的探索压力”。
    '''
    entropy_coef: float = 0.001  # 熵正则化系数

    actor_lr: float = 3e-4  # 策略网络学习率
    critic_lr: float = 1e-4  # 价值网络学习率
    actor_lr_level2: float = 3e-5#课程学习等级2的策略网络学习率
    critic_lr_level2: float = 1e-4#课程学习等级2的价值网络学习率

    '''
    PPO训练epochs,越大样本利用率越高，但训练时间越长容易过拟合，
    #越小训练时间越短，但样本利用率越低
    '''
    ppo_epochs: int = 4
    ppo_update_interval: int = 5 #收集多少回合再进行训练
    '''
    如果发现 Critic Loss 一直很高，说明价值网络学得不够快，
    可以把 critic_updates_per_epoch 调到 6-8。
    '''
    critic_updates_per_epoch: int = 2#价值网络更新次数

    max_grad_norm: float = 0.5#最大梯度范数，防止梯度爆炸

    '''
    价值估计震荡严重 降低 value_clip_eps（如 0.1），降低 tau（如 0.005）
    '''
    value_clip_eps: float = 0.2#价值裁剪 epsilon，防止价值估计过拟合

    '''
    tau=0.01 是 DQN/PPO 的标准值，稳定可靠。如果训练震荡，可以降到 0.005。
    '''
    tau: float = 0.01#目标网络更新参数

    # 奖励分配网络
    alloc_d_model: int = 256#奖励分配网络隐藏层维度
    alloc_num_heads: int = 8#奖励分配网络头数
    alloc_num_layers: int = 4#奖励分配网络层数
    alloc_d_ff: int = 512#奖励分配网络前馈网络隐藏层维度，如果分配网络学得慢，可以调大
    alloc_dropout: float = 0.15#奖励分配网络dropout率
    alloc_entropy_coef: float = 0.05#奖励分配网络熵正则化系数
    alloc_lr: float = 1e-4#奖励分配网络学习率，如果分配网络的损失不降或震荡，可以降到 3e-4 或 1e-4。
    max_seq_len: int = 100#奖励分配网络最大序列长度

    # 模仿训练
    temperature: float = 0.3#模仿训练温度，决定密集奖励的分布是更尖锐（集中）还是更平坦（均匀）。
    # 0.1 会 softmax 塌缩成 one-hot（权重全给最高奖励步，推进步信号消失）；
    # 0.5 时 target 为"保持>推进>后退"（约 148:20:2.7），可学且梯度足够强（比 1.0 强约 10 倍）；
    # 1.0 时 target 太平坦（7.4:2.8:1），梯度被 80 维归一化稀释到 1e-5 量级，500 集也推不动。
    imitation_lr: float = 1e-3#模仿训练学习率
    imitation_batch_episodes: int = 30#模仿训练每批回合数（batch 维=回合，一次梯度更新）
    imitation_epochs: int = 20#模仿训练 epoch 数（每 epoch 打乱 500 回合重新分批；50×25批=1250 梯度步达标）
    imitation_dataset_size: int = 500#离线训练数据集大小（每 100 次训练时用当前 PPO 现采 N 回合组成固定数据集）

    # 课程学习
    curriculum_window_size: int = 50#课程学习窗口大小
    curriculum_promote_threshold: float = 500.0#课程学习提升阈值
    curriculum_demote_threshold: float = 10.0#课程学习降级阈值
    curriculum_promote_cooldown: int = 1000#课程学习提升冷却期
    curriculum_demote_cooldown: int = 200#课程学习降级冷却期
    curriculum_level: int = 0#课程学习初始等级

    # 训练流程
    total_episodes: int = 10000
    log_interval: int = 10#日志间隔
    save_interval: int = 1000#保存间隔
    save_dir: str = "./checkpoints/seed{seed}"#保存目录，只是普通的字符串
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 设备（自动检测）


@dataclass
class TrainingMetrics:
    """训练指标：记录每个 episode 的各项损失和奖励，用于日志和绘图"""
    # 基础信息
    episode: int = 0#回合数
    terminal_reward: float = 0.0#回合结束奖励
    episode_length: int = 0#回合长度

    # PPO 损失
    policy_loss: float = 0.0#策略网络损失
    value_loss: float = 0.0#价值网络损失
    entropy_loss: float = 0.0#熵正则化损失

    # 分配网络损失
    alloc_td_loss: float = 0.0#分配网络TD损失
    alloc_entropy_loss: float = 0.0#分配网络熵正则化损失
    alloc_total_loss: float = 0.0#分配网络总损失
    alloc_kl_loss: float = 0.0#分配网络KL损失
    alloc_mse_loss: float = 0.0#分配网络MSE损失
    alloc_kl_total_loss: float = 0.0  # 分配网络总损失
    alloc_w_rew_corr: float = 0.0  # 分配网络输出权重与密集奖励的相关性（诊断：是否学会"高奖励步权重高"）


    # 课程学习状态
    curriculum_level: int = 0#课程学习等级
    curriculum_changed: str = ""#课程学习变化
    log_std: float = 0.0#策略网络的log标准差

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
            f.write("episode,terminal_reward,curriculum_level,policy_loss,value_loss,entropy_loss,alloc_kl_total_loss,alloc_kl_loss,alloc_mse_loss,alloc_td_loss,alloc_entropy_loss,alloc_total_loss,alloc_w_rew_corr\n")

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
            learning_rate=config.imitation_lr,#模仿训练学习率
            max_seq_len=config.max_seq_len,
        )
        self.allocator_agent = RewardAllocatorAgent(alloc_config, str(self.device))
        self.raw_buffer_list = []  # 存放分配网络的训练数据

        print("[3/5] 初始化环境...")
        self.env = make_fetch_env(
            env_id=config.env_id,
            curriculum_level=config.curriculum_level, 
        )
        self.dense_reward = self.env._dense_reward#奖励模式
        self.curriculum_level = self.env._curriculum_level#课程学习等级

        print("[4/5] 初始化课程学习管理器...")
        self.curriculum_manager = AdaptiveCurriculumManager(
            window_size=config.curriculum_window_size,
            promote_threshold=config.curriculum_promote_threshold,
            demote_threshold=config.curriculum_demote_threshold,
            promote_cooldown=config.curriculum_promote_cooldown,
            demote_cooldown=config.curriculum_demote_cooldown,
            current_level=self.curriculum_level,
        )
        
        print("[5/5] 准备完成!")
        # 全局训练计数器
        self.episode_count = 0#回合数计数器
        # 分配网络模仿训练：累积的原始轨迹（Level 0 每集 append，每 100 次批量训练一次后清空）
        self.raw_buffer_list: List[RawTrajectoryBuffer] = []

    def _policy_fn(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,float, float]:
        """策略函数，用于采集数据"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value, log_std, mean = self.network.get_action(obs_t)

        return (
            action.squeeze(0).cpu().numpy(),  # 动作（tanh 后）
            log_std.squeeze(0).cpu().numpy(),  # log_std
            mean.squeeze(0).cpu().numpy(),  # mean
            log_prob.item(),  # log_prob
            value.item(),  # value
        )

    def get_entropy_coef(self) -> float:
        """获取熵系数（固定值）"""
        return self.config.entropy_coef

    def train_merge_buffers(self, buffers: List[TrainingBuffer]) -> TrainingBuffer:

        merged = TrainingBuffer()

        for buf in buffers:

            merged.states.extend(buf.states)
            merged.actions.extend(buf.actions)
            merged.advantages.extend(buf.advantages)
            merged.returns.extend(buf.returns)
            merged.old_log_probs.extend(buf.old_log_probs)

        merged.next_states = []  # 训练不需要
        merged.rewards = []  # 训练不需要

        merged.weights = None  # 训练不需要
        merged.weights_entropy = None  # 训练不需要
        merged.terminal_reward = 0.0  # 训练不需要

        return merged

    # def raw_merge_buffers(self, buffers: List[RawTrajectoryBuffer]) -> RawTrajectoryBuffer:
    #
    #     merged = RawTrajectoryBuffer()
    #
    #     for buf in buffers:

    def _build_state_pairs(self, observations: List[np.ndarray]) -> np.ndarray:
        """把 N+1 个观测序列构造成 N 个状态对 concat[s_i, s_{i+1}]（分配网络 Q 输入）"""
        obs_arr = np.array(observations, dtype=np.float32)          # [N+1, obs_dim]
        pairs = np.concatenate([obs_arr[:-1], obs_arr[1:]], axis=-1)  # [N, 2*obs_dim]
        return pairs

    def collect_trajectory(self) -> RawTrajectoryBuffer:
        """采集一个回合的轨迹"""
        buffer = RawTrajectoryBuffer()

        obs, info = self.env.reset()
        obs = unpack_obs(obs)

        done = False
        while not done:
            # 获取动作
            action, log_std, mean, log_prob, value = self._policy_fn(obs)

            # 与环境交互
            next_obs, terminated, truncated, info = self.env.step(action)
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

        # 存储最终奖励和每步的奖励的列表
        terminal_reward = info.get("terminal_reward", 0.0)
        step_rewards = info.get("step_rewards", [])
        buffer.finalize(terminal_reward, step_rewards)

        return buffer

    def allocate_rewards(self, buffer: RawTrajectoryBuffer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用奖励分配网络分配奖励"""

        # 状态对序列（N 个有效 transition）：concat[s_i, s_{i+1}]
        obs_padded = self._build_state_pairs(buffer.observations)
        # actions[1:]     → a_0..a_{N-1}（实际动作）
        # action_dist[1:] → dist_0..dist_{N-1}（动作分布）
        act_padded = np.array(buffer.actions[1:], dtype=np.float32)
        action_dist = buffer.get_action_dist()[1:]

        # 分配奖励（用分配网络产出权重）
        weights, entropy = self.allocator_agent.allocate(
            states=obs_padded,
            actions=act_padded,
            actions_dist=action_dist,
        )

        # 计算分配后的奖励
        terminal_reward = buffer.terminal_reward
        allocated_rewards = weights * terminal_reward

        return allocated_rewards, weights, entropy

    def compute_gae_and_train_buffer(
        self,
        buffer: RawTrajectoryBuffer,#长度为N+1
        allocated_rewards: torch.Tensor,#长度为N
        weights,
        entropy,
    ) -> TrainingBuffer:
        """计算 GAE 并构建训练缓冲区"""
        # 实际 transition 数 N (observations 有 N+1 个，包含 terminal state)
        N = len(buffer.observations) - 1#长度为N

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
        # weights 保持为 tensor（保留计算图，供 compute_td_loss 反向传播到分配网络）
        if isinstance(weights, torch.Tensor):
            train_buffer.weights = weights.cpu()
        else:
            train_buffer.weights = torch.tensor(weights, dtype=torch.float32)

        # 权重分布的熵（保留计算图，供 compute_td_loss 使用，避免重复计算）
        if entropy is not None:
            train_buffer.weights_entropy = entropy.cpu()

        for i in range(N):
            train_buffer.states.append(observations[i])  # s_i
            train_buffer.actions.append(buffer.actions[i + 1])  # a_i (shifted)
            train_buffer.rewards.append(allocated_rewards[i])  # r_i
            train_buffer.advantages.append(advantages[i])
            train_buffer.returns.append(returns[i])
            train_buffer.old_log_probs.append(buffer.log_probs[i])  # log π(a_i|s_i)
            train_buffer.next_states.append(observations[i + 1])  # s_{i+1}

        return train_buffer

    def train_ppo_step(self, train_buffer: TrainingBuffer) -> Dict[str, float]:
        """训练 PPO 网络：策略网络更新 ppo_epochs 次，价值网络更新 critic_updates_per_epoch * ppo_epochs 次"""

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
        for epoch in range(self.config.ppo_epochs):
            # ========== Actor 更新 ==========
            self.optimizer_actor.zero_grad()
            # 使用新的 get_log_prob 方法（包含 tanh 修正）
            new_log_probs = self.network.actor.get_log_prob(states, actions)

            # 计算对角高斯策略的熵（基于 log_std，不依赖具体采样动作）
            # 公式: H = 0.5 * Σ(ln(2πσ²) + 1) = Σ(0.5*(ln(2π)+1) + ln(σ))
            log_std = torch.clamp(self.network.actor.log_std, self.config.log_std_min, self.config.log_std_max)
            entropy = (0.5 * (1.0 + np.log(2.0 * np.pi)) + log_std).sum()

            ratio = torch.exp(new_log_probs - old_log_probs)
            # 限制极端比值，防止单次更新破坏策略
            ratio = torch.clamp(ratio, 0.01, 5.0)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -current_entropy_coef * entropy  # 熵损失
            actor_loss = policy_loss + entropy_loss

            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.config.max_grad_norm)
            self.optimizer_actor.step()

            # ========== Critic 更新（多次） ==========
            # 价值裁剪：只对第一个 epoch 保存 old_values（用于裁剪参考）
            if epoch == 0:
                with torch.no_grad():
                    old_values = self.network.get_value(states).squeeze(-1).detach()

            for _ in range(self.config.critic_updates_per_epoch):
                self.optimizer_critic.zero_grad()
                values = self.network.get_value(states).squeeze(-1)
                # 价值裁剪：限制 value 不能偏离 old_value 太远
                values_clipped = torch.clamp(
                    values,
                    old_values - self.config.value_clip_eps,
                    old_values + self.config.value_clip_eps,
                )
                # 使用裁剪后的价值计算损失（取较大的损失）
                value_loss_unclipped = F.smooth_l1_loss(values, returns)
                value_loss_clipped = F.smooth_l1_loss(values_clipped, returns)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
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
        }

    def train_allocator_step(
        self,
        train_buffer: TrainingBuffer,
        critic_network,
    ) -> Dict[str, float]:
        """训练奖励分配网络"""

        # 构造 N+1 个 states: [s_0, ..., s_{N-1}] + [s_N]
        states_list = list(train_buffer.states) + [train_buffer.next_states[-1]]
        states_padded = np.array(states_list, dtype=np.float32)

        # 获取价值估计（长度为 N+1）
        obs_t = torch.FloatTensor(states_padded).to(self.device)
        # online value 需要梯度，但 target value 不需要
        v_all = critic_network.get_value(obs_t).squeeze(-1)  # [N+1]: V(s_0)...V(s_N)
        with torch.no_grad():
            v_target_all = critic_network.get_target_value(obs_t).squeeze(-1)  # [N+1]

        # allocator 前向输出 weights 为 N（position 1...N 对应 transition 0...N-1）
        # TD target 使用对应位置的 value: V(s_1)...V(s_N)
        # 保持为 tensor 以支持梯度计算
        v_online = v_all[:-1].detach()  # [N], 断开 Critic 计算图
        v_target_next = v_target_all[1:]  # [N]，最后一步 V(s_N) 用于 bootstrap

        # # gamma 转为 tensor 以便与 tensor 相乘
        # gamma_tensor = torch.tensor(self.config.gamma, dtype=torch.float32, device=self.device)

        # 训练分配网络
        metrics = self.allocator_agent.TD_train_step(
            train_buffer=train_buffer,
            gamma=self.config.gamma,
            v_online=v_online,
            v_target_next=v_target_next,
        )

        return {
            "alloc_td_loss": metrics.get("td_loss", 0),
            "alloc_entropy_loss": metrics.get("entropy_loss", 0),
            "alloc_total_loss": metrics.get("total_loss", 0),
            "alloc_kl_loss": metrics.get("kl_loss", 0),
            "alloc_mse_loss": metrics.get("mse_loss", 0),
            "alloc_kl_total_loss": metrics.get("kl_total_loss", 0),
        }

    def collect_allocator_dataset(self, num_episodes: int) -> Dict[str, np.ndarray]:
        """用当前 PPO 策略采集 num_episodes 个回合，**采集时就完成预处理**，
        直接产出分配网络的离线训练数据集（开箱即用，训练时不再预处理）。

        数据集每个样本 = 一个完整回合（分配网络的输入+监督，softmax 回合内归一化）:
            states:   [N, 80, 2*obs]   状态对 concat[s_i, s_{i+1}]（前后两个时间步）
            actions:  [N, 80, act]     实际动作
            dists:    [N, 80, 2*act]   动作分布 (mean, log_std)
            targets:  [N, 80]          softmax(exp(r/T)) 模仿目标（Σ=1）
            rewards:  [N, 80]          密集奖励（诊断 corr 用）

        采集期间不更新任何网络（PPO 停在当前状态），保证数据集是同一时刻策略的固定快照。
        """
        self.network.eval()
        states_arr, actions_arr, dists_arr, targets_arr, rewards_arr = [], [], [], [], []
        for _ in range(num_episodes):
            rb = self.collect_trajectory()
            dense = np.array(rb.step_rewards, dtype=np.float32)                        # [80]
            imp = np.exp(dense / self.config.temperature)
            target = imp / imp.sum()                                                   # [80] Σ=1
            states_arr.append(self._build_state_pairs(rb.observations))                # [80, 2*obs]
            actions_arr.append(np.array(rb.actions[1:], dtype=np.float32))             # [80, act]
            dists_arr.append(np.array(rb.get_action_dist()[1:], dtype=np.float32))     # [80, 2*act]
            targets_arr.append(target)
            rewards_arr.append(dense)
        self.network.train()
        dataset = {
            "states": np.array(states_arr),
            "actions": np.array(actions_arr),
            "dists": np.array(dists_arr),
            "targets": np.array(targets_arr),
            "rewards": np.array(rewards_arr),
        }
        return dataset

    def train_allocator_step_imitation(self, dataset) -> Dict[str, float]:
        """训练奖励分配网络（模仿训练）：批量 mini-batch 流程（离线）

        输入: collect_allocator_dataset 的产物 dict（states/actions/dists/targets/rewards，
        已预处理，直接送分配网络，无需再预处理）。

        流程:
          1. 打乱回合顺序，按 imitation_batch_episodes(20) 回合/批划分
          2. 每批: 20 回合拼成 [B, 80, ...] 一起 forward，
             loss = MSE(logits_c, log_target_c) + 0.1*KL，一次梯度更新
          3. 重复 imitation_epochs(50) 个 epoch，每个 epoch 重新打乱、重新分批
          4. 训练后对全体回合计算权重-奖励相关性 corr（诊断是否学会"高奖励步权重高"）

        返回各指标均值（数据集为空时返回全 0）。
        """
        S = dataset["states"]      # [N, 80, 2*obs]
        A = dataset["actions"]     # [N, 80, act]
        D = dataset["dists"]       # [N, 80, 2*act]
        T = dataset["targets"]     # [N, 80]
        R = dataset["rewards"]     # [N, 80]
        n_eps = len(S)

        device = self.device
        batch = self.config.imitation_batch_episodes
        n_batch = (n_eps + batch - 1) // batch
        print(f"[模仿训练] {n_eps} 回合, {batch}/批, {n_batch} 批/epoch, "
              f"{self.config.imitation_epochs} epoch = {n_batch * self.config.imitation_epochs} 梯度步")

        # ---------- 2~4. 多 epoch mini-batch 训练 ----------
        total_kl, total_mse, total_steps = 0.0, 0.0, 0
        for epoch in range(self.config.imitation_epochs):
            order = np.random.permutation(n_eps)  # 每 epoch 重新打乱
            for start in range(0, n_eps, batch):
                idx = order[start:start + batch]
                s = torch.FloatTensor(S[idx]).to(device)      # [B, 80, 2*obs]
                a = torch.FloatTensor(A[idx]).to(device)      # [B, 80, act]
                d = torch.FloatTensor(D[idx]).to(device)      # [B, 80, 2*act]
                t = torch.FloatTensor(T[idx]).to(device)      # [B, 80]

                logits, w, _ = self.allocator_agent.network.forward_logits(s, a, d)  # [B, 80]

                # logits 回归 log(target)（去均值，只约束形状）
                logits_c = logits - logits.mean(dim=-1, keepdim=True)
                log_t = torch.log(t + 1e-8)
                log_t_c = log_t - log_t.mean(dim=-1, keepdim=True)
                mse_loss = F.mse_loss(logits_c, log_t_c)  # 主损失
                mse_loss = ((logits_c - log_t_c) ** 2).mean(dim=-1).mean()
                kl_loss = F.kl_div(torch.log(w + 1e-8), t, reduction='batchmean')  # 辅助，[B,80]→/B
                loss = mse_loss + 0.1 * kl_loss

                self.allocator_agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.allocator_agent.network.parameters(), max_norm=0.5)
                self.allocator_agent.optimizer.step()

                total_kl += kl_loss.item()
                total_mse += mse_loss.item()
                total_steps += 1

        # ---------- 5. 诊断 corr：训练后全体回合 权重 vs 奖励 ----------
        corrs = []
        with torch.no_grad():
            for i in range(n_eps):
                s = torch.FloatTensor(S[i:i+1]).to(device)
                a = torch.FloatTensor(A[i:i+1]).to(device)
                d = torch.FloatTensor(D[i:i+1]).to(device)
                _, w, _ = self.allocator_agent.network.forward_logits(s, a, d)
                wn = w.cpu().numpy()[0]
                if R[i].std() > 1e-9:
                    corrs.append(float(np.corrcoef(R[i], wn)[0, 1]))
        w_rew_corr = float(np.mean(corrs)) if corrs else 0.0

        return {
            "alloc_kl_loss": total_kl / max(total_steps, 1),
            "alloc_mse_loss": total_mse / max(total_steps, 1),
            "alloc_kl_total_loss": (total_kl + total_mse) / max(total_steps, 1),
            "alloc_td_loss": 0.0,
            "alloc_entropy_loss": 0.0,
            "alloc_total_loss": 0.0,
            "alloc_w_rew_corr": w_rew_corr,
        }

    def train_one_episode(self) -> TrainingMetrics:
        """训练一个回合"""
        if self.dense_reward:
            '''level 0 使用密集奖励模式，ppo用密集奖励信号训练，分配网络用密集奖励做模仿训练'''

            # 离线训练分配网络（模仿训练）：每 100 次训练时，
            #    先用当前 PPO 采集 imitation_dataset_size(500) 回合组成固定数据集，
            #    再打乱分批(20/批) 多 epoch(50) 训练，训练完清空
            if self.episode_count % 50 == 0:
                self.raw_buffer_list = self.collect_allocator_dataset(self.config.imitation_dataset_size)
                alloc_metrics = self.train_allocator_step_imitation(self.raw_buffer_list)
                # alloc_metrics = {"alloc_kl_loss" "alloc_mse_loss","alloc_kl_total_loss"}
                self.raw_buffer_list = []
            else:
                # 未到离线训练时，alloc_metrics 用默认值（metrics 构造需要）
                alloc_metrics = {"alloc_kl_loss": 0.0, "alloc_mse_loss": 0.0, "alloc_kl_total_loss": 0.0,
                                 "alloc_td_loss": 0.0, "alloc_entropy_loss": 0.0, "alloc_total_loss": 0.0,
                                 "alloc_w_rew_corr": 0.0}

            #ppo的训练
            train_buffers_list = []  # 存放多个 TrainingBuffer
            for _ in range(self.config.ppo_update_interval):
                # 1. 采集轨迹
                raw_buffer = self.collect_trajectory()  # 里面都是List[Dict]
                # 采集到了observation（N+1，最后一个位置为最终状态,第一个位置为初始状态）,
                # action（N+1,第一个位置为0）,step_reward（N,密集奖励）,
                # mean（N+1,第一个位置为0）,log_stds（N+1,第一个位置为0）,
                # terminal_reward（标量），log_prob (N个)，value (N个)，

                # 2. 奖励分配
                allocated_rewards = np.array(raw_buffer.step_rewards)  # 转为 numpy array
                '''
                state(N)、action(N)、reward(N)真实的密集奖励信号、advantage(N)、return(N)、old_log_prob(N)、
                next_state(N)、weights=[]、terminal_reward(标量)
                '''
                # 3. 计算 GAE 并构建训练缓冲区
                train_buffer = self.compute_gae_and_train_buffer(raw_buffer, allocated_rewards, [],None)

                train_buffers_list.append(train_buffer)

            train_buffers = self.train_merge_buffers(train_buffers_list)
            # 4. 训练 PPO,用密集奖励信号训练
            ppo_metrics = self.train_ppo_step(train_buffers)

        else:

            # 1. 采集轨迹
            raw_buffer = self.collect_trajectory()  # 里面都是List[Dict]
            # 采集到了observation（N+1，最后一个位置为最终状态,第一个位置为初始状态）,
            # action（N+1,第一个位置为0）,step_reward（N,密集奖励）,
            # mean（N+1,第一个位置为0）,log_stds（N+1,第一个位置为0）,
            # terminal_reward（标量），log_prob (N个)，value (N个)，

            '''level 1.2.3 使用稀疏奖励模式'''
            # 2. 奖励分配
            allocated_rewards1, weights, entropy = self.allocate_rewards(raw_buffer)#长度为N,level1分配网络训练用
            allocated_rewards2 = np.array(raw_buffer.step_rewards)#长度为N，level1 ppo训练用

            # 3. 计算 GAE 并构建训练缓冲区
            '''
            state(N)、action(N)、reward(N)分配网络给的密集奖励、advantage(N)、return(N)、old_log_prob(N)、
            next_state(N)、weights=分配网络给的奖励权重、terminal_reward(标量)
            '''
            train_buffer1 = self.compute_gae_and_train_buffer(raw_buffer, allocated_rewards1, weights, entropy)
            '''
            state(N)、action(N)、reward(N)真实的密集奖励信号、advantage(N)、return(N)、old_log_prob(N)、
            next_state(N)、weights=[]、terminal_reward(标量)
            '''
            train_buffer2 = self.compute_gae_and_train_buffer(raw_buffer, allocated_rewards2, [],None)

            # 4. PPO 只在 level 2+ 用分配网络给的奖励权重训练，level 1 用ppo的数据训练分配网络，但是ppo任然用密集奖励信号训练
            if self.curriculum_level == 1:
                ppo_metrics = self.train_ppo_step(train_buffer2)
                #ppo_metrics = {"policy_loss" "value_loss" "entropy_loss"}

            else:
                ppo_metrics = self.train_ppo_step(train_buffer1)
                #ppo_metrics = {"policy_loss" "value_loss" "entropy_loss" "ppo_loss"}

            # 5. 训练分配网络
            alloc_metrics = self.train_allocator_step(train_buffer1, self.network)
            #alloc_metrics = {"alloc_td_loss" "alloc_entropy_loss" "alloc_kl_total_loss"}

        # 更新课程学习
        terminal_reward = raw_buffer.terminal_reward
        curriculum_changed = self.curriculum_manager.on_episode_end(terminal_reward)
        self.curriculum_level = self.curriculum_manager.get_current_level()

        if curriculum_changed:
            self.env.set_curriculum_level(self.curriculum_level)
            # 获取当前奖励模式
            self.dense_reward = self.env._dense_reward
            
            if self.curriculum_level == 0:
                self.allocator_agent.update_learning_rate(self.config.imitation_lr)
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.config.actor_lr
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.config.critic_lr
            
            elif self.curriculum_level == 1:
                self.allocator_agent.update_learning_rate(self.config.alloc_lr)
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.config.actor_lr
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.config.critic_lr

            # 根据课程等级调整学习率
            elif self.curriculum_level == 2:
                # Level 2: PPO 使用较小的学习率
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.config.actor_lr_level2
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.config.critic_lr_level2
                print(f"[课程升级] Level 2: PPO lr -> actor={self.config.actor_lr_level2}, critic={self.config.critic_lr_level2}")
                
            elif self.curriculum_level == 3:
                # Level 3: PPO 恢复初始学习率
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.config.actor_lr
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.config.critic_lr
                print(f"[课程升级] Level 3: PPO lr -> actor={self.config.actor_lr}, critic={self.config.critic_lr}")

        # 构建指标
        metrics = TrainingMetrics(
            episode=self.episode_count,
            terminal_reward=terminal_reward,
            episode_length=len(raw_buffer) - 1,  # observations 有 N+1 个，transition 数为 N
            curriculum_level=self.curriculum_level,
            curriculum_changed=str(curriculum_changed),
            **{**ppo_metrics, **alloc_metrics},
        )

        # 获取当前 log_std
        log_std_mean = self.network.actor.log_std.mean().item()
        metrics.log_std = log_std_mean

        self.episode_count += 1

        # 每次回合结束都追加写入文件（课程切换时实时保存）
        with open(self.rewards_file, 'a') as f:
            f.write(f"{self.episode_count},{terminal_reward},{self.curriculum_level},"
                    f"{metrics.policy_loss:.6f},{metrics.value_loss:.6f},"
                    f"{metrics.entropy_loss:.6f},{metrics.alloc_kl_total_loss:.6f},"
                    f"{metrics.alloc_kl_loss:.6f},{metrics.alloc_mse_loss:.6f},"
                    f"{metrics.alloc_td_loss:.6f},{metrics.alloc_entropy_loss:.6f},"
                    f"{metrics.alloc_total_loss:.6f},{metrics.alloc_w_rew_corr:.6f}\n")

        return metrics

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        # 统计
        terminal_rewards = []#最终奖励
        curriculum_levels = []#课程等级
        
        # 主循环，会在终端中显示一个动态进度条
        pbar = tqdm(total=self.config.total_episodes, desc="Training")
        start_time = time.time()

        while self.episode_count < self.config.total_episodes:

            # 训练一个回合
            metrics = self.train_one_episode()

            # 记录
            terminal_rewards.append(metrics.terminal_reward)
            curriculum_levels.append(metrics.curriculum_level)

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "Level": metrics.curriculum_level,
                "TR": f"{metrics.terminal_reward:.1f}",
            })

            # 日志
            if self.episode_count % self.config.log_interval == 1:
                avg_reward = np.mean(terminal_rewards[-self.config.log_interval:])
                current_level = self.curriculum_manager.get_current_level()
                window_mean = self.curriculum_manager.reward_window.mean() if self.curriculum_manager.reward_window.count() > 0 else 0.0
                current_entropy_coef = self.get_entropy_coef()

                log_std_mean = self.network.actor.log_std.mean().item()

                pbar.write(f"[Episode {self.episode_count}]")
                pbar.write(f"  课程等级: {current_level}")
                pbar.write(f"  熵系数: {current_entropy_coef:.4f}, log_std: {log_std_mean:.4f}")
                pbar.write(f"  平均最终奖励 (最近{self.config.log_interval}): {avg_reward:.2f}")
                pbar.write(f"  滑动缓冲区均值: {window_mean:.2f}")
                pbar.write(f"  Actor损失: {metrics.policy_loss:.4f}")
                pbar.write(f"  Critic损失: {metrics.value_loss:.4f}")
                pbar.write(f"  Actor熵损失: {metrics.entropy_loss:.4f}")
                pbar.write(f"  分配网络td损失: {metrics.alloc_td_loss:.4f}")
                pbar.write(f"  分配网络熵损失: {metrics.alloc_entropy_loss:.4f}")
                pbar.write(f"  分配网络总损失: {metrics.alloc_total_loss:.4f}")
                pbar.write(f"  分配网络kl损失: {metrics.alloc_kl_loss:.4f}")
                pbar.write(f"  分配网络mes损失: {metrics.alloc_mse_loss:.4f}")
                pbar.write(f"  分配网络模仿训练总损失: {metrics.alloc_kl_total_loss:.4f}")
                pbar.write(f"  分配网络权重-奖励相关: {metrics.alloc_w_rew_corr:.3f}")

            # 课程升级时保存
            if metrics.curriculum_changed == "promote":
                save_path = f"{self.save_dir}/level_up_{self.episode_count}.pt"
                self.save(save_path)
                pbar.write(f"*** 课程升级! 模型已保存: {save_path} ***")

            # 定时保存
            if self.episode_count % self.config.save_interval == 0:
                save_path = f"{self.save_dir}/checkpoint_{self.episode_count}.pt"
                self.save(save_path)
                pbar.write(f"[Episode {self.episode_count}] 模型已保存: {save_path}")

        pbar.close()

        # 训练结束
        elapsed = time.time() - start_time
        print(f"\n训练完成! 耗时: {elapsed:.1f}秒")
        print(f"平均奖励: {np.mean(terminal_rewards):.2f}")
        print(f"最高奖励: {np.max(terminal_rewards):.2f}")
        print(f"奖励数据已保存到: {self.rewards_file}")

        # 保存最终模型
        self.save(f"{self.save_dir}/final_model.pt")

    def save(self, path: str):
        """保存模型（含课程管理器的完整内部状态）"""
        torch.save({
            "ppo_network": self.network.state_dict(),
            "allocator": self.allocator_agent.network.state_dict(),
            "curriculum_state": {
                "current_level": self.curriculum_manager.current_level,
                "episode_count": self.curriculum_manager.episode_count,
                "cooldown_remaining": self.curriculum_manager.cooldown_remaining,
                "since_last_change": self.curriculum_manager.since_last_change,
                "reward_buffer": list(self.curriculum_manager.reward_window._buffer),
                "level_episode_counts": dict(self.curriculum_manager.level_episode_counts),
                "event_history": self.curriculum_manager.event_history,
            },
            "config": self.config,
            "episode_count": self.episode_count,
        }, path)

    def load(self, path: str):
        """加载模型（含课程管理器完整恢复）"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["ppo_network"])
        self.allocator_agent.network.load_state_dict(checkpoint["allocator"])
        self.episode_count = checkpoint["episode_count"]

        # 恢复课程管理器内部状态
        cs = checkpoint["curriculum_state"]
        self.curriculum_manager.current_level = cs["current_level"]
        self.curriculum_manager.episode_count = cs["episode_count"]
        self.curriculum_manager.cooldown_remaining = cs["cooldown_remaining"]
        self.curriculum_manager.since_last_change = cs["since_last_change"]
        # 恢复滑动窗口缓冲区
        from collections import deque
        self.curriculum_manager.reward_window._buffer = deque(
            cs["reward_buffer"],
            maxlen=self.curriculum_manager.reward_window.window_size,
        )
        self.curriculum_manager.level_episode_counts = cs["level_episode_counts"]
        self.curriculum_manager.event_history = list(cs["event_history"])

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
    # 解析命令行参数
    parser = argparse.ArgumentParser()#解析命令行参数，description=	设置帮助信息，运行 python script.py -h 时会显示这段文字
    parser.add_argument("--seed", type=int, default=42)#--seed 指定随机种子，default=None 表示没有指定时使用默认值 42
    parser.add_argument("--gpu", type=str, default="")
    args = parser.parse_args()#解析命令行参数，返回一个Namespace对象，包含所有参数的值

    # 创建配置
    config = TrainingConfig()

    # 如果命令行指定了 seed，覆盖默认值
    if args.seed is not None:
        config.seed = args.seed

    # 如果命令行指定了 GPU，覆盖设备（01/02/03 → cuda:0/1/2）
    if args.gpu:
        gpu_id = int(args.gpu)
        if torch.cuda.is_available():
            if gpu_id >= torch.cuda.device_count():
                print(f"警告: --gpu {args.gpu} 超出可用 GPU 数 {torch.cuda.device_count()}，回退到 cuda:0")
                gpu_id = 0
            config.device = f"cuda:{gpu_id}"

    # 设置随机数种子
    set_seed(config.seed)

    # 创建训练器
    trainer = PPOTrainerWithAllocator(config)

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
