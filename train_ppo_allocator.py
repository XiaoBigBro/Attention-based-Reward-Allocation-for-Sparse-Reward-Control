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
# 1. 统一配置 - 所有可调参数集中于此
# =============================================================================

@dataclass
class TrainingConfig:
    """训练配置：包含所有可调超参数，供 PPOTrainerWithAllocator 使用"""
    # 环境
    env_id: str = "FetchPush-v4"
    seed: int = 888

    # PPO actor网络参数
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])  # 策略网络隐藏层维度
    log_std_min: float = -2.0#策略网络log标准差最小值
    log_std_max: float = 1.0#策略网络log标准差最大值
    log_std_init: float = 0.5#策略网络log标准差初始值，控制策略的初始探索程度。std=exp(log_std)实际标准差

    # PPO 训练的超参数

    gamma: float = 0.99  # 折扣因子,控制未来奖励的权重
    lam: float = 0.9#GAE参数
    clip_eps: float = 0.2#PPO裁剪参数,控制策略更新的稳定性.
    entropy_coef: float = 0.05  # 熵正则化系数

    actor_lr0: float = 3e-4  # 策略网络学习率
    critic_lr0: float = 3e-4  # 价值网络学习率

    actor_lr1: float = 1e-4#课程学习等级1的策略网络学习率
    critic_lr1: float = 1e-4#课程学习等级1的价值网络学习率

    actor_lr2: float = 1e-4  # 课程学习等级2的策略网络学习率
    critic_lr2: float = 1e-4  # 课程学习等级2的价值网络学习率

    actor_lr3: float = 1e-4  # 课程学习等级3的策略网络学习率
    critic_lr3: float = 1e-4  # 课程学习等级3的价值网络学习率

    max_grad_norm: float = 0.5  # 最大梯度范数，防止梯度爆炸
    value_clip_eps: float = 0.2  # 价值裁剪 epsilon，防止价值估计过拟合
    tau: float = 0.01  # 目标网络更新参数

    # level0中PPO 训练框架的参数

    ppo_epochs: int = 4 #每批数据训练多少次
    ppo_update_interval: int = 10 #收集多少回合再进行训练
    critic_updates_per_epoch: int = 2 #level0里面，价值网络额外更新的次数
    mini_batch_size = 1024

    # 奖励分配网络初始化超参数
    alloc_d_model: int = 64#奖励分配网络隐藏层维度
    alloc_num_heads: int = 4#奖励分配网络头数
    alloc_num_layers: int = 2#奖励分配网络层数
    alloc_d_ff: int = 128#奖励分配网络前馈网络隐藏层维度，如果分配网络学得慢，可以调大
    alloc_dropout: float = 0.2#奖励分配网络dropout率
    max_seq_len: int = 110#奖励分配网络最大序列长度

    # 分配网络level0模仿训练参数
    temperature: float = 0.7#模仿训练温度，决定密集奖励的分布是更尖锐（集中）还是更平坦（均匀）。
    # 0.1 会 softmax 塌缩成 one-hot（权重全给最高奖励步，推进步信号消失）；
    # 0.5 时 target 为"保持>推进>后退"（约 148:20:2.7），可学且梯度足够强（比 1.0 强约 10 倍）；
    # 1.0 时 target 太平坦（7.4:2.8:1），梯度被 80 维归一化稀释到 1e-5 量级，500 集也推不动。
    imitation_lr: float = 1e-3#模仿训练学习率
    imitation_batch_episodes: int = 512#模仿训练每批回合数（batch 维=回合，一次梯度更新）
    imitation_epochs: int = 100#模仿训练 epoch 数（每 epoch 打乱 500 回合重新分批；50×25批=1250 梯度步达标）
    imitation_dataset_size: int = 1000#离线训练数据集大小（每 100 次训练时用当前 PPO 现采 N 回合组成固定数据集）

    #奖励分配网络level1训练参数
    alloc_lr1: float = 1e-4  # 奖励分配网络学习率，如果分配网络的损失不降或震荡，可以降到 3e-4 或 1e-4。

    # 课程学习
    curriculum_window_size: int = 50#课程学习窗口大小
    curriculum_promote_threshold: float = 500.0#课程学习提升阈值
    curriculum_demote_threshold: float = 50.0#课程学习降级阈值
    curriculum_promote_cooldown: int = 200#课程学习提升冷却期
    curriculum_demote_cooldown: int = 60#课程学习降级冷却期
    curriculum_level: int = 0#课程学习初始等级

    # 训练流程
    total_episodes: int = 50000
    log_interval: int = 10#日志间隔
    save_interval: int = 200#保存间隔
    save_dir: str = "./checkpoints/seed{seed}"#保存目录，只是普通的字符串
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 设备（自动检测）

@dataclass
class TrainingMetrics:
    """训练指标：记录每个 episode 的各项损失和奖励，用于日志和绘图"""
    # 基础信息
    episode: int = 0#回合数
    terminal_reward: float = 0.0#回合结束奖励

    # PPO 损失
    policy_loss: float = 0.0#策略网络损失
    value_loss: float = 0.0#价值网络损失
    entropy_loss: float = 0.0#熵正则化损失

    # 分配网络损失
    alloc_kl_loss: float = 0.0#分配网络KL损失
    alloc_mse_loss: float = 0.0#分配网络MSE损失
    alloc_w_rew_corr: float = 0.0  # 分配网络输出权重与密集奖励的相关性（诊断：是否学会"高奖励步权重高"）

    # 课程学习状态
    curriculum_level: int = 0#课程学习等级
    log_std: float = 0.0#策略网络的log标准差

# ============================================================================
# 2. 训练器
# ============================================================================

class PPOTrainerWithAllocator:
    """
    PPO 训练器（集成奖励分配网络和课程学习）
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
            f.write("episode,terminal_reward,curriculum_level,policy_loss,value_loss,entropy_loss,alloc_kl_loss,alloc_mse_loss,alloc_w_rew_corr\n")

        # 创建设备
        self.device = torch.device(config.device)
        print(f"设备: device={self.device}")
        
        # 自动查询环境维度（不创建渲染窗口）
        obs_dim = get_env_obs_dim(config.env_id)
        action_dim = get_env_action_dim(config.env_id)

        print("\n[1/5] 初始化 PPO 网络...")
        self.network = PPONetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
            log_std_init=config.log_std_init,
        ).to(self.device)

        # ppo优化器（策略和价值网络分开设置学习率）
        self.optimizer_actor = optim.Adam(
            self.network.actor.parameters(),
            lr=config.actor_lr0,
        )
        self.optimizer_critic = optim.Adam(
            self.network.critic.parameters(),
            lr=config.critic_lr0,
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
            learning_rate=config.imitation_lr,#模仿训练学习率
            max_seq_len=config.max_seq_len,
        )
        self.allocator_agent = RewardAllocatorAgent(alloc_config, str(self.device))

        print("[3/5] 初始化环境...")
        self.env = make_fetch_env(
            env_id=config.env_id,
            curriculum_level=config.curriculum_level, 
        )
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
        self.curriculum_changed = False
        
        print("[5/5] 准备完成!")

        # ===================== 各课程等级训练状态（统一归类） =====================
        # 全局训练计数器（train() 主循环每回合 +1）
        self.episode_count = 0#回合数计数器

        # --- Level 0 训练状态（密集奖励 PPO + 分配网络离线模仿训练） ---
        # 分配网络模仿训练数据集累积（原始轨迹快照列表）：
        #   Level 0 分支每 episode_count 检查一次，到离线训练时机（episode_count % 50 == 0）
        #   就现场采集 imitation_dataset_size 集组成固定数据集训练分配网络，训练完清空。
        self.raw_buffer_list: List[RawTrajectoryBuffer] = []
        self.states_arr, self.actions_arr, self.dists_arr, self.targets_arr, self.rewards_arr = [],[], [], [], []

        # --- Level 1 训练状态（混合奖励过渡 PPO + mini-batch 块式训练） ---
        # _l1_alpha: 当前块的 α。块首（采集前）按 level1 内真实集数算一次、整块固定；
        #             α = min(1, level1 内真实集数 / curriculum_promote_cooldown)。
        self._l1_alpha: float = 0.0

    def _policy_fn(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,float, float]:
        """策略函数，用于采集数据"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, log_std, mean = self.network.get_action(obs_t)
            value = self.network.get_value(obs_t)

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

        return merged

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
            action, log_std, mean, log_prob, value = self._policy_fn(obs)#在cpu上

            # 与环境交互
            next_obs, terminated, truncated, info = self.env.step(action)
            next_obs = unpack_obs(next_obs)

            # 存储
            buffer.add(obs, action, mean, log_std, log_prob, value)#在cpu上

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
        act_padded = np.array(buffer.actions[1:], dtype=np.float32)
        action_dist = np.array(buffer.get_action_dist()[1:],dtype=np.float32)

        # 分配奖励（用分配网络产出权重）
        weights = self.allocator_agent.allocate(
            states=obs_padded,
            actions=act_padded,
            actions_dist=action_dist,
        )

        # 计算分配后的奖励
        terminal_reward = buffer.terminal_reward
        allocated_rewards = weights * terminal_reward

        return allocated_rewards

    def compute_gae_and_train_buffer(
        self,
        buffer: RawTrajectoryBuffer,#长度为N+1
        allocated_rewards: np.array,#长度为N
    ) -> TrainingBuffer:
        """计算 GAE 并构建训练缓冲区"""
        # 实际 transition 数 N (observations 有 N+1 个，包含 terminal state)
        N = len(buffer.observations) - 1#长度为N
        allocated_rewards_list = allocated_rewards.tolist()

        observations = buffer.observations  # 长度 N+1: s_0, ..., s_N

        # 重新计算价值（使用 observations 序列）
        obs_tensor = torch.FloatTensor(np.array(observations,dtype=np.float32)).to(self.device)
        with torch.no_grad():
            critic_values = self.network.get_value(obs_tensor).squeeze(-1).cpu().numpy().tolist()
            target_values = self.network.get_target_value(obs_tensor).squeeze(-1).cpu().numpy().tolist()

        values = critic_values[:-1]  # 长度 N，取出V(s_0), V(s_1), ..., V(s_{N-1})
        next_values = target_values[1:]  # 长度 N，取出V(s_1), V(s_2), ..., V(s_{N})
        dones = [False] * (N - 1) + [True]  # 长度 N，最后一步是 terminal

        # 计算 GAE（所有输入长度必须一致为 N）
        advantages, returns = compute_gae(
            allocated_rewards_list,  # 长度 N
            values,  # 长度 N
            next_values,  # 长度 N
            dones,  # 长度 N
            self.config.gamma,
            self.config.lam,
        )

        # 构建训练缓冲区（长度为 N）
        train_buffer = TrainingBuffer()

        train_buffer.states = observations[:N]  # s_0...s_{N-1}
        train_buffer.actions = buffer.actions[1:N + 1]  # a_0...a_{N-1}（已偏移）
        train_buffer.advantages = advantages[:N]  # 长度 N
        train_buffer.returns = returns[:N]  # 长度 N
        train_buffer.old_log_probs = buffer.log_probs[:N]  # 长度 N

        return train_buffer

    # ============================================================================
    # PPO 训练（两种模式，互不影响）：
    #   - train_ppo_step:            全 batch 更新（Level 0 / Level >= 2 兜底用）
    #   - train_ppo_step_minibatch:  打乱 mini-batch 更新（Level 1 块式训练用）
    # ============================================================================

    def train_ppo_step(self, train_buffer: TrainingBuffer) -> Dict[str, float]:
        """训练 PPO 网络（全 batch 更新）：策略网络更新 ppo_epochs 次，
        价值网络每 epoch 更新 critic_updates_per_epoch 次。
        供 Level 0 / Level >= 2 兜底使用（每 ppo_update_interval 集的 batch 做一次更新）。
        Level 1 的 mini-batch 打乱训练见 train_ppo_step_minibatch。"""

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
        total_policy_updates = 0
        total_entropy_updates = 0

        n = states.size(0)

        # 价值裁剪参考值：本次 update 开始前用当前 critic 对全部状态算一次，整个 update 固定
        with torch.no_grad():
            old_values = self.network.get_value(states).squeeze(-1).detach()

        # 策略网络更新 ppo_epochs 次，价值网络更新 critic_updates_per_epoch * ppo_epochs 次
        for epoch in range(self.config.ppo_epochs):

            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.config.mini_batch_size):

                idx = perm[start:start + self.config.mini_batch_size]
                s_b, a_b = states[idx], actions[idx]
                old_lp_b = old_log_probs[idx]
                adv_b = advantages[idx]
                ret_b = returns[idx]

                # ========== Actor 更新 ==========
                self.optimizer_actor.zero_grad()
                # 使用新的 get_log_prob 方法（包含 tanh 修正）
                new_log_probs = self.network.actor.get_log_prob(s_b, a_b)

                # 计算对角高斯策略的熵（基于 log_std，不依赖具体采样动作）
                # 公式: H = 0.5 * Σ(ln(2πσ²) + 1) = Σ(0.5*(ln(2π)+1) + ln(σ))
                log_std = torch.clamp(self.network.actor.log_std, self.config.log_std_min, self.config.log_std_max)
                entropy = (0.5 * (1.0 + np.log(2.0 * np.pi)) + log_std).sum()

                ratio = torch.exp(new_log_probs - old_lp_b)
                # 限制极端比值，防止单次更新破坏策略
                ratio = torch.clamp(ratio, 1e-6, 1e6)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -current_entropy_coef * entropy  # 熵损失
                actor_loss = policy_loss + entropy_loss

                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.network.actor.parameters(), self.config.max_grad_norm)
                self.optimizer_actor.step()

                # ========== Critic 更新（多次） ==========
                # 价值裁剪：只对第一个 epoch 保存 old_values（用于裁剪参考）
                for _ in range(self.config.critic_updates_per_epoch):
                    self.optimizer_critic.zero_grad()
                    values = self.network.get_value(s_b).squeeze(-1)
                    # 价值裁剪：限制 value 不能偏离 old_value 太远
                    values_clipped = torch.clamp(
                        values,
                        old_values[idx] - self.config.value_clip_eps,
                        old_values[idx] + self.config.value_clip_eps
                    )
                    # 使用裁剪后的价值计算损失（取较大的损失）
                    value_loss_unclipped = F.smooth_l1_loss(values, ret_b)
                    value_loss_clipped = F.smooth_l1_loss(values_clipped, ret_b)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.config.max_grad_norm)
                    self.optimizer_critic.step()

                    total_value_loss += value_loss.item()
                    total_critic_updates += 1

                total_policy_loss += policy_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_policy_updates += 1
                total_entropy_updates += 1

        # 软更新 target_critic
        self.network._soft_update_target(self.config.tau)

        return {
            "policy_loss": total_policy_loss / total_policy_updates,
            "value_loss": total_value_loss / total_critic_updates,
            "entropy_loss": total_entropy_loss / total_entropy_updates,
        }

    def clear_allocator_buffer(self):

        self.raw_buffer_list = []
        self.states_arr = []
        self.actions_arr = []
        self.dists_arr = []
        self.targets_arr = []
        self.rewards_arr = []


    def train_allocator_step_imitation(self, dataset):
        """
        训练奖励分配网络（模仿训练）：批量 mini-batch 流程（离线）
        """
        S = dataset["states"]      # [N, 80, 2*obs]，拼接的状态
        A = dataset["actions"]     # [N, 80, act]
        D = dataset["dists"]       # [N, 80, 2*act]
        T = dataset["targets"]     # [N, 80]
        R = dataset["rewards"]     # [N, 80]，真实密集奖励信号
        n_eps = len(S)

        # ========== 先打乱，再划分 ==========
        indices = np.random.permutation(n_eps)

        n_train = int(n_eps * 0.9)
        n_test = n_eps - n_train

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        device = self.device
        batch = self.config.imitation_batch_episodes#每次批采集多少个回合

        # ---------- 一次性把训练数据搬到 GPU ----------
        # 使用 from_numpy 避免多余拷贝，再转 float32 并送到 device
        S_gpu = torch.from_numpy(S).float().to(device)  # [N, 80, 2*obs]
        A_gpu = torch.from_numpy(A).float().to(device)  # [N, 80, act]
        D_gpu = torch.from_numpy(D).float().to(device)  # [N, 80, 2*act]
        T_gpu = torch.from_numpy(T).float().to(device)  # [N, 80]

        S_gpu_train = S_gpu[train_idx]
        A_gpu_train = A_gpu[train_idx]
        D_gpu_train = D_gpu[train_idx]
        T_gpu_train = T_gpu[train_idx]

        S_gpu_test = S_gpu[test_idx]
        A_gpu_test = A_gpu[test_idx]
        D_gpu_test = D_gpu[test_idx]
        T_gpu_test = T_gpu[test_idx]
        R_test = R[test_idx]
        w_rew_corr =0.0
        epoch = 0

        # ---------- 2~4. 多 epoch mini-batch 训练 ----------
        total_kl, total_mse, total_steps = 0.0, 0.0, 0

        while epoch < self.config.imitation_epochs*3 :

            self.allocator_agent.network.train()
            order = np.random.permutation(n_train)  # 每 epoch 重新打乱

            for start in range(0, n_train, batch):

                idx = order[start:start + batch]
                idx_gpu = torch.from_numpy(idx).long().to(device)

                s = S_gpu_train[idx_gpu]  # [B, 80, 2*obs]
                a = A_gpu_train[idx_gpu]  # [B, 80, act]
                d = D_gpu_train[idx_gpu]  # [B, 80, 2*act]
                t = T_gpu_train[idx_gpu]  # [B, 80]

                logits, w = self.allocator_agent.network(s, a, d)  # [B, 80]

                # logits 回归 log(target)（去均值，只约束形状）
                logits_c = logits - logits.mean(dim=-1, keepdim=True)
                log_t = torch.log(t + 1e-8)
                log_t_c = log_t - log_t.mean(dim=-1, keepdim=True)
                mse_loss = F.mse_loss(logits_c, log_t_c)  # 主损失
                w_clamped = torch.clamp(w, min=1e-8, max=1.0)
                kl_loss = F.kl_div(torch.log(w_clamped), t, reduction='batchmean')  # 辅助，[B,80]→/B
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
            self.allocator_agent.network.eval()
            with torch.no_grad():
                diag_batch = 128
                for i in range(0,n_test,diag_batch):

                    end = min(i + diag_batch, n_test)
                    s = S_gpu_test[i:end]  # [B, 80, 2*obs]
                    a = A_gpu_test[i:end]  # [B, 80, act]
                    d = D_gpu_test[i:end]  # [B, 80, 2*act]

                    _, w = self.allocator_agent.network(s, a, d)
                    wn = w.cpu().numpy()
                    for j in range(end - i):
                        if R_test[i+j].std() > 1e-9:
                            corrs.append(float(np.corrcoef(R_test[i+j], wn[j])[0, 1]))
            w_rew_corr = float(np.mean(corrs)) if corrs else 0.0
            epoch += 1
            print(f"Epoch {epoch}/{self.config.imitation_epochs}: "
                  f"mse={total_mse/total_steps:.4f}, corr={w_rew_corr:.4f}")

            if epoch >= self.config.imitation_epochs and w_rew_corr > 0.85:
                print(f"✅ 训练完成：epoch={epoch}, corr={w_rew_corr:.4f}")
                break

    def allocatorw_rew_corr(self, dataset) -> Dict[str, float]:

        S = dataset["states"]  # [N, 80, 2*obs]，拼接的状态
        A = dataset["actions"]  # [N, 80, act]
        D = dataset["dists"]  # [N, 80, 2*act]
        R = dataset["rewards"]  # [N, 80]，真实密集奖励信号
        device = self.device
        n_eps = len(S)

        S_gpu = torch.from_numpy(S).float().to(device)  # [N, 80, 2*obs]
        A_gpu = torch.from_numpy(A).float().to(device)  # [N, 80, act]
        D_gpu = torch.from_numpy(D).float().to(device)  # [N, 80, 2*act]

        corrs = []
        with torch.no_grad():
            for i in range(n_eps):

                s = S_gpu[i:i + 1]  # [B, 80, 2*obs]
                a = A_gpu[i:i + 1]  # [B, 80, act]
                d = D_gpu[i:i + 1]  # [B, 80, 2*act]

                _, w = self.allocator_agent.network(s, a, d)
                wn = w.cpu().numpy()[0]
                if R[i].std() > 1e-9:
                    corrs.append(float(np.corrcoef(R[i], wn)[0, 1]))
        w_rew_corr = float(np.mean(corrs)) if corrs else 0.0

        return {
            "alloc_kl_loss": 0.0,
            "alloc_mse_loss": 0.0,
            "alloc_w_rew_corr": w_rew_corr,
        }

    def train_one_episode(self) -> TrainingMetrics:

        if self.curriculum_level == 0:

            train_buffers_list = []  # 存放多个 TrainingBuffer
            total = 0.0
            self.network.eval()

            for i in range(self.config.ppo_update_interval):#采集ppo_update_interval个回合，目前是50

                # 1. 采集轨迹
                raw_buffer = self.collect_trajectory()
                total += raw_buffer.terminal_reward

                # 2. 奖励 = 真实密集奖励
                dense_rewards = np.array(raw_buffer.step_rewards,dtype=np.float32)  # 转为 numpy array

                #收集分配网络的数据
                imp = np.exp(dense_rewards / self.config.temperature)
                target = imp / imp.sum()
                self.states_arr.append(self._build_state_pairs(raw_buffer.observations))  # [80, 2*obs]，拼接的
                self.actions_arr.append(np.array(raw_buffer.actions[1:], dtype=np.float32))  # [80, act]
                self.dists_arr.append(np.array(raw_buffer.get_action_dist()[1:], dtype=np.float32))  # [80, 2*act]
                self.targets_arr.append(target)
                self.rewards_arr.append(dense_rewards)#真实的密集奖励

                # 3. 计算 GAE 并构建训练缓冲区
                train_buffer = self.compute_gae_and_train_buffer(raw_buffer, dense_rewards)
                train_buffers_list.append(train_buffer)

            self.network.train()
            train_buffers = self.train_merge_buffers(train_buffers_list)
            # 4. batch 训练 PPO，用密集奖励信号训练
            ppo_metrics = self.train_ppo_step(train_buffers)
            alloc_metrics = {"alloc_kl_loss": 0.0, "alloc_mse_loss": 0.0, "alloc_w_rew_corr": 0.0}

        elif self.curriculum_level == 1:

            self._l1_alpha = self.curriculum_manager.level_episode_counts[self.curriculum_level] / self.config.curriculum_promote_cooldown
            self._l1_alpha = min(1.0, 4 * self._l1_alpha)

            train_buffers_list = []  # 存放多个 TrainingBuffer
            total = 0.0
            self.network.eval()

            for i in range(self.config.ppo_update_interval):
                # 1.1 采集轨迹
                raw_buffer = self.collect_trajectory()
                total += raw_buffer.terminal_reward

                #真实的密集奖励
                dense_rewards = np.array(raw_buffer.step_rewards, dtype=np.float32)

                # 1.2 分配网络对当前轨迹事后分配：allocated = R_term·w（[N]，Σ=R_term）
                allocated = self.allocate_rewards(raw_buffer)

                # 1.3 混合奖励（Σ 恒 = R_term），用块首固定的 α
                allocated_np = allocated.detach().cpu().numpy()
                r_mix = (1.0 - self._l1_alpha) * dense_rewards + self._l1_alpha * allocated_np

                # # 收集分配网络的数据
                self.states_arr.append(self._build_state_pairs(raw_buffer.observations))  # [80, 2*obs]
                self.actions_arr.append(np.array(raw_buffer.actions[1:], dtype=np.float32))  # [80, act]
                self.dists_arr.append(np.array(raw_buffer.get_action_dist()[1:], dtype=np.float32))  # [80, 2*act]
                self.rewards_arr.append(dense_rewards)

                # 1.4 计算 GAE 并构建训练缓冲区
                train_buffer = self.compute_gae_and_train_buffer(raw_buffer, r_mix)
                train_buffers_list.append(train_buffer)

            self.network.train()
            train_buffers = self.train_merge_buffers(train_buffers_list)
            ppo_metrics = self.train_ppo_step(train_buffers)

            # 计算分配网络与真实信号的相关性
            self.raw_buffer_list = {"states": np.array(self.states_arr), "actions": np.array(self.actions_arr),
                                    "dists": np.array(self.dists_arr), "rewards": np.array(self.rewards_arr)}

            alloc_metrics = self.allocatorw_rew_corr(self.raw_buffer_list)
            self.clear_allocator_buffer()

        else:
            train_buffers_list = []  # 存放多个 TrainingBuffer
            total = 0.0
            self.network.eval()

            for i in range(self.config.ppo_update_interval):
                # 1. 采集轨迹
                raw_buffer = self.collect_trajectory()
                total += raw_buffer.terminal_reward

                # 真实的密集奖励
                dense_rewards = np.array(raw_buffer.step_rewards, dtype=np.float32)

                # 2 分配网络对当前轨迹事后分配
                allocated = self.allocate_rewards(raw_buffer)
                allocated_np = allocated.detach().cpu().numpy()

                #收集信息，用于计算分配网络输出与真实值的相关性
                self.states_arr.append(self._build_state_pairs(raw_buffer.observations))  # [80, 2*obs]
                self.actions_arr.append(np.array(raw_buffer.actions[1:], dtype=np.float32))  # [80, act]
                self.dists_arr.append(np.array(raw_buffer.get_action_dist()[1:], dtype=np.float32))  # [80, 2*act]
                self.rewards_arr.append(dense_rewards)

                # 3. 计算 GAE 并构建训练缓冲区
                train_buffer = self.compute_gae_and_train_buffer(raw_buffer, allocated_np)
                train_buffers_list.append(train_buffer)

            self.network.train()
            train_buffers = self.train_merge_buffers(train_buffers_list)
            ppo_metrics = self.train_ppo_step(train_buffers)

            #计算分配网络与真实信号的相关性
            self.raw_buffer_list = {"states": np.array(self.states_arr), "actions": np.array(self.actions_arr),
                                    "dists": np.array(self.dists_arr),"rewards": np.array(self.rewards_arr)}

            alloc_metrics = self.allocatorw_rew_corr(self.raw_buffer_list)
            self.clear_allocator_buffer()

        # 更新课程学习
        terminal_reward = total / (i+1)
        self.curriculum_changed = self.curriculum_manager.on_episode_end(terminal_reward)
        self.curriculum_level = self.curriculum_manager.get_current_level()

        if self.curriculum_changed:

            self.env.set_curriculum_level(self.curriculum_level)
            # 获取当前奖励模式
            
            if self.curriculum_level == 0:
                self.allocator_agent.update_learning_rate(self.config.imitation_lr)
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.config.actor_lr0
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.config.critic_lr0
            
            elif self.curriculum_level == 1:

                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.config.actor_lr1
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.config.critic_lr1

            # 根据课程等级调整学习率
            elif self.curriculum_level == 2:

                self.allocator_agent.network.eval()
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.config.actor_lr2
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.config.critic_lr2
                
            elif self.curriculum_level == 3:

                self.allocator_agent.network.eval()
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.config.actor_lr3
                for param_group in self.optimizer_critic.param_groups:
                    param_group['lr'] = self.config.critic_lr3

        # 构建指标
        metrics = TrainingMetrics(
            episode=self.episode_count,
            terminal_reward=terminal_reward,
            curriculum_level=self.curriculum_level,
            **{**ppo_metrics, **alloc_metrics},
        )

        # 获取当前 log_std
        log_std_mean = self.network.actor.log_std.mean().item()
        metrics.log_std = log_std_mean

        # 每次回合结束都追加写入文件（课程切换时实时保存）
        with open(self.rewards_file, 'a') as f:
            f.write(f"{self.episode_count},{terminal_reward},{self.curriculum_level},"
                    f"{metrics.policy_loss:.6f},{metrics.value_loss:.6f},"
                    f"{metrics.entropy_loss:.6f},{metrics.alloc_kl_loss:.6f},"
                    f"{metrics.alloc_mse_loss:.6f},{metrics.alloc_w_rew_corr:.6f}\n")

        return metrics

    def train(self):

        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        # 统计
        terminal_rewards = []#最终奖励
        
        # 主循环，会在终端中显示一个动态进度条
        pbar = tqdm(total=self.config.total_episodes, desc="Training")
        start_time = time.time()

        while self.episode_count < self.config.total_episodes:

            # 训练一个回合
            metrics = self.train_one_episode()

            # 记录
            terminal_rewards.append(metrics.terminal_reward)

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "level": self.curriculum_manager.get_current_level(),
                "TR": f"{metrics.terminal_reward:.1f}",
            })

            # 日志
            if self.episode_count % self.config.log_interval == 0 and self.episode_count != 0:
                avg_reward = np.mean(terminal_rewards[-self.config.log_interval:])
                current_level = self.curriculum_manager.get_current_level()
                window_mean = self.curriculum_manager.reward_window.mean()
                current_entropy_coef = self.get_entropy_coef()

                log_std_mean = metrics.log_std

                pbar.write(f"[Episode {self.episode_count}]")
                pbar.write(f"  课程等级: {current_level}")
                pbar.write(f"  熵系数: {current_entropy_coef:.4f}, log_std: {log_std_mean:.4f}")
                pbar.write(f"  平均最终奖励 (最近{self.config.log_interval}): {avg_reward:.2f}")
                pbar.write(f"  滑动缓冲区均值: {window_mean:.2f}")
                pbar.write(f"  Actor损失: {metrics.policy_loss:.4f}")
                pbar.write(f"  Critic损失: {metrics.value_loss:.4f}")
                pbar.write(f"  Actor熵损失: {metrics.entropy_loss:.4f}")
                pbar.write(f"  分配网络kl损失: {metrics.alloc_kl_loss:.4f}")
                pbar.write(f"  分配网络mes损失: {metrics.alloc_mse_loss:.4f}")
                pbar.write(f"  分配网络权重-奖励相关: {metrics.alloc_w_rew_corr:.3f}")

            self.episode_count += 1

            if self.curriculum_changed and self.curriculum_level == 1:
                print("分配网络训练")
                # 分配网络的训练
                self.raw_buffer_list = {"states": np.array(self.states_arr), "actions": np.array(self.actions_arr),
                                        "dists": np.array(self.dists_arr),
                                        "targets": np.array(self.targets_arr), "rewards": np.array(self.rewards_arr)}
                self.train_allocator_step_imitation(self.raw_buffer_list)
                self.allocator_agent.update_learning_rate(self.config.alloc_lr1)
                self.clear_allocator_buffer()
                self.allocator_agent.network.eval()

            # 定时保存
            if self.episode_count % self.config.save_interval == 0:
                save_path = f"{self.save_dir}/checkpoint_{self.episode_count}.pt"
                self.save(save_path)
                pbar.write(f"[Episode {self.episode_count}] 模型已保存: {save_path}")

        pbar.close()

        # 训练结束
        elapsed = time.time() - start_time
        self.env.close()
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

# ============================================================================
# 3. 随机数种子设置
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
# 4. 命令行解析
# ============================================================================

def parse_and_set_config(config):
    """解析命令行参数并设置配置"""
    # 解析命令行参数
    parser = argparse.ArgumentParser()#解析命令行参数
    parser.add_argument("--seed", type=int, default=42)#--seed 指定随机种子，default=None 表示没有指定时使用默认值 42
    parser.add_argument("--gpu", type=str, default=0)
    args = parser.parse_args()#解析命令行参数，返回一个Namespace对象，包含所有参数的值

    # 如果命令行指定了 seed，覆盖默认值
    if args.seed is not None:
        config.seed = args.seed

    # 如果命令行指定了 GPU，覆盖设备（01/02/03 → cuda:0/1/2）
    if args.gpu is not None:

        gpu_id = int(args.gpu)

        if torch.cuda.is_available():

            if gpu_id >= torch.cuda.device_count():#gpu的索引从0开始，但是device_count返回的是个数
                gpu_id = 0
            config.device = f"cuda:{gpu_id}"
    
    return config

# ============================================================================
# 5. 主函数
# ============================================================================

def main():
    """主函数"""

    # 创建配置参数
    config = TrainingConfig()

    # 解析命令行参数并设置配置
    config = parse_and_set_config(config)
    
    # 设置随机数种子
    set_seed(config.seed)

    # 创建训练器
    trainer = PPOTrainerWithAllocator(config)

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()


