"""
奖励分配网络 (Reward Allocator)
- 基于注意力机制将 terminal_reward 分配到序列每步
- 损失: TD误差 + 负熵正则化（鼓励权重集中）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ppo import TrainingBuffer


# --- Positional Encoding ---
#位置编码
#1
class PositionalEncoding(nn.Module):
    """
    Sinusoidal 位置编码

    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer (不参与梯度计算，但会随模型保存)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x [batch, seq_len, d_model]
        输出: x + positional_encoding
        """
        return x + self.pe[:, :x.size(1)]

# --- FeedForward ---
#前馈网络
class FeedForward(nn.Module):
    """
    标准 FFN: Linear → ReLU → Linear

    公式: FFN(x) = W2 · ReLU(W1 · x + b1) + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --- MultiHeadAttention ---
#多头注意力机制
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    公式:
        MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W^O
        head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        Attention(Q,K,V) = softmax(QK^T / √d_k) · V
    """

    def __init__(
        self,
        d_model: int,#输入维度
        num_heads: int,#头数
        dropout: float,
        bias: bool,#是否使用偏置
        use_causal_mask: bool ,#防止看到未来位置
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_causal_mask = use_causal_mask

        # 投影矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)#
        self.W_o = nn.Linear(d_model, d_model, bias=bias)#输出投影矩阵
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        #将Q,K,V投影到多头注意力机制
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Causal Mask: 防止看到未来位置（本任务为事后分配，统一不使用）
        if self.use_causal_mask:
            causal = torch.tril(
                torch.ones(seq_len_q, seq_len_k, device=query.device, dtype=torch.bool),
                diagonal=0
            )
            scores = scores.masked_fill(~causal, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len_q, self.d_model)
        return self.W_o(context)

#1
# --- TransformerLayer ---
#Transformer层
class TransformerLayer(nn.Module):
    """
    单层 Transformer Block

    结构: Self-Attention → Add & LayerNorm → FFN → Add & LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        use_causal_mask: bool,
    ):
        super().__init__()

        # 自注意力层
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout,bias=True,
                                           use_causal_mask=use_causal_mask)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        #dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x [batch, seq_len, d_model]
        输出: [batch, seq_len, d_model]
        """
        # 自注意力 + 残差连接
        attn_out = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + 残差连接
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
#1
#交叉注意力层
class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层

    结构: Cross-Attention → Add & LayerNorm → FFN → Add & LayerNorm

    注意: 本任务为事后分配（episode 已结束、最终奖励已知），不使用因果掩蔽
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        use_causal_mask: bool,
    ):
        super().__init__()

        # 交叉注意力: Query 来自 state, Key/Value 来自 action
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, bias=True, use_causal_mask=use_causal_mask)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.use_causal_mask = use_causal_mask

    def forward(
        self,
        query: torch.Tensor,  # 来自状态
        key: torch.Tensor,     # K 来自动作分布
        value: torch.Tensor,   # V 来自动作
    ) -> torch.Tensor:
        """
        输入:
            query: [batch, q_len, d_model] (状态)
            key: [batch, kv_len, d_model] (动作分布 K)
            value: [batch, kv_len, d_model] (动作 V)
        输出: [batch, q_len, d_model]
        """
        # 交叉注意力 + 残差连接
        attn_out = self.cross_attn(query, key, value)
        x = self.norm1(query + self.dropout(attn_out))

        # FFN + 残差连接
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

#1
class RewardAllocator(nn.Module):
    """基于注意力机制的奖励分配网络。

    注意力设计:
        Q = 状态对 concat[s_i, s_{i+1}]（增量信息载体：贡献 Δd = d(s_i) − d(s_{i+1})）
        K = action_dist (动作分布: mean + log_std)
        V = actions (实际采取的动作 a_0..a_{N-1})

    对齐约定（输入 N 个有效 transition，无 dummy）:
        position i (0-based) 对应 transition i:
            Q = concat[s_i, s_{i+1}]（执行 a_i 前后的状态对）
            K = dist_i（动作 a_i 的分布）
            V = a_i（实际动作）
        输出权重 w_i 分配给 transition i

    特点:
        - Q/K/V 三路均注入位置编码（保持时间语义一致）
        - 状态自注意力与交叉注意力均无因果掩码（事后分配允许看全局）
        - 输出两层 MLP → softmax，Σw = 1
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int ,
        num_heads: int,
        num_layers: int,#Transformer层数
        d_ff: int,#前馈网络隐藏层维度
        dropout: float,
        entropy_coef: float,#熵正则化系数
        max_seq_len: int ,#最大序列长度
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.entropy_coef = entropy_coef

        # Q 编码器: 状态对 (concat[s_i, s_{i+1}]) 编码 + 位置编码
        # 输入维度 = 2 * obs_dim（相邻两状态拼接，携带增量信息）
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        #位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # K 编码器: 动作分布 (mean + log_std) -> 简单 MLP
        # 输入维度: action_dim (mean) + action_dim (log_std) = action_dim * 2
        self.dist_encoder = nn.Sequential(
            nn.Linear(action_dim * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # V 编码器: 动作编码
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Q 处理: Transformer 层 (用于状态对序列，无掩码：事后分配允许看全局)
        self.state_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout, use_causal_mask=False)
            for _ in range(num_layers)
        ])

        # 交叉注意力层 (Q attends to K, V，无因果掩码：事后分配)
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, d_ff, dropout, use_causal_mask=False)
            for _ in range(num_layers)
        ])

        # 输出层：两层 MLP（增强权重预测表达力）
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        actions_dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        输入（N 个有效 transition，无 dummy，长度必须一致）:
            states:       [N, 2*obs_dim]    状态对 concat[s_i, s_{i+1}]
            actions:      [N, action_dim]   实际动作 a_0..a_{N-1}
            actions_dist: [N, action_dim*2] 动作分布 (mean, log_std)

        输出:
            weights: [N] 每个 transition 的权重，Σ=1
            entropy: [B] 权重分布的熵，保留计算图
        """
        # 处理单序列输入
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            actions_dist = actions_dist.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        _, weights, entropy = self._forward_no_squeeze(states, actions, actions_dist)

        # 如果输入是单序列，压缩输出
        if squeeze_output:
            weights = weights.squeeze(0)
            entropy = entropy.squeeze(0)

        return weights, entropy

    def forward_logits(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        actions_dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播（返回 logits，供模仿训练直接回归 log(target)）。

        输入同 forward；输出: (logits, weights, entropy)。
        logits 是 softmax 之前的原始分数（去均值后形状与 log(target) 对应）。
        """
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            actions_dist = actions_dist.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        logits, weights, entropy = self._forward_no_squeeze(states, actions, actions_dist)

        if squeeze_output:
            logits = logits.squeeze(0)
            weights = weights.squeeze(0)
            entropy = entropy.squeeze(0)

        return logits, weights, entropy

    def compute_td_loss(
        self,
        train_buffer: TrainingBuffer,
        gamma: float,
        v_online: torch.Tensor,
        v_target_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 TD 误差损失 + 熵正则化。

        输入:
            train_buffer: TrainingBuffer
            gamma: float
            v_online: [N] V(s_1)...V(s_N)，已断开 Critic 计算图
            v_target_next: [N] V(s_2)...V(s_{N+1})

        weights 和 entropy 来自 train_buffer，保留着分配网络的计算图。
        """
        # weights 是 tensor，直接使用（保留计算图供反向传播）
        weights_td = train_buffer.weights.to(v_online.device)
        terminal_reward = train_buffer.terminal_reward

        td_target = terminal_reward * weights_td + gamma * v_target_next
        td_error = td_target - v_online
        td_loss = (td_error ** 2).mean()

        # 熵正则化：使用 _forward_no_squeeze 已算好的熵，避免重复计算
        entropy_val = train_buffer.weights_entropy.to(v_online.device).mean()
        entropy_loss = self.entropy_coef * entropy_val  # 正号：最小化熵
        total_loss = td_loss + entropy_loss

        # 构建诊断指标
        metrics = {
            'td_loss': td_loss.item() if isinstance(td_loss, torch.Tensor) else td_loss,
            'entropy_loss': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        }

        return total_loss, metrics

    def _forward_no_squeeze(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        actions_dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = states.shape

        # Q = 状态对（concat[s_i, s_{i+1}]，携带增量信息）
        q_emb = self.state_encoder(states)
        q_emb = self.pos_encoder(q_emb)

        # V = 动作（实际动作 a_0..a_{N-1}）
        v_emb = self.action_encoder(actions)
        v_emb = self.pos_encoder(v_emb)

        # K = 动作分布（dist_0..dist_{N-1}），与 Q/V 一致注入位置编码
        k_emb = self.dist_encoder(actions_dist)
        k_emb = self.pos_encoder(k_emb)

        # Q 处理（状态对自注意力，无掩码：事后分配允许看全局状态）
        q_hidden = q_emb
        for layer in self.state_layers:
            q_hidden = layer(q_hidden)

        # 交叉注意力 (Q attends to K, V)，无因果掩码（事后分配允许看全局动作）
        fused = q_hidden
        for layer in self.cross_layers:
            fused = layer(fused, k_emb, v_emb)

        # 每个位置一个分数 → softmax 归一化为权重
        logits = self.output_proj(fused).squeeze(-1)  # [B, N]
        weights = F.softmax(logits, dim=-1)  # [B, N]

        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)

        return logits, weights, entropy#logits、权重、熵


# --- RewardAllocatorAgent ---
#1
@dataclass
class RewardAllocatorConfig:
    """奖励分配器配置（所有参数由 TrainingConfig 集中传入）"""
    obs_dim: int
    action_dim: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    dropout: float
    entropy_coef: float
    learning_rate: float
    max_seq_len: int

class RewardAllocatorAgent:
    """
    奖励分配器封装类

    整合网络、优化器，提供简洁的训练接口。
    """

    def __init__(self, config: RewardAllocatorConfig, device: str ):
        self.config = config
        self.device = device

        # 创建网络
        self.network = RewardAllocator(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            entropy_coef=config.entropy_coef,
            max_seq_len=config.max_seq_len,#最大序列长度
        ).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

        # 统计信息,记录训练步数
        self.step_count = 0

    def allocate(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        actions_dist: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分配奖励

        输入（N 个有效 transition，无 dummy，长度必须一致）:
            states:       [N, 2*obs_dim]    状态对 concat[s_i, s_{i+1}]
            actions:      [N, action_dim]   实际动作 a_0..a_{N-1}
            actions_dist: [N, action_dim*2] 动作分布 (mean, log_std)

        输出:
            weights: [N] 每个 transition 的权重，保留计算图供训练
            entropy: [B] 权重分布的熵，保留计算图供训练
        """
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.FloatTensor(actions).to(self.device)
            actions_dist_t = torch.FloatTensor(actions_dist).to(self.device)

        weights, entropy = self.network(states_t, actions_t, actions_dist_t)

        return weights, entropy

    def allocate_logits(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        actions_dist: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        分配奖励（返回 logits，供模仿训练直接回归 log(target)）。

        输入同 allocate；输出: (logits, weights, entropy)，均保留计算图。
        """
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.FloatTensor(actions).to(self.device)
            actions_dist_t = torch.FloatTensor(actions_dist).to(self.device)

        logits, weights, entropy = self.network.forward_logits(states_t, actions_t, actions_dist_t)

        return logits, weights, entropy

    # 模仿训练
    def imitation_train_step(
        self,
        target_weights: torch.Tensor,
        logits: torch.Tensor,
    ) -> dict:
        """模仿训练：让 logits 逼近 log(target_weights)（去均值后回归）。

        对比 softmax 交叉熵（KL）：交叉熵梯度 = (w − t)，在 80 维归一化分布下
        被 softmax 耦合 + batchmean(/80) 稀释到 ~1e-4，方向噪声大、学不动；
        logits 回归的梯度 = (logits − log(target))，直接由 target 给定方向，
        无 softmax 耦合、尺度大（log 域 ~1），训练快且稳定。

        主损失：去均值后的 logits 与 log(target) 的 MSE（只约束形状，不约束 softmax 常数自由度）。
        辅助损失：0.1 × softmax KL（数值尺度校准）。
        """
        self.network.train()

        self.optimizer.zero_grad()

        # 去均值：softmax 对 logits 整体平移不变，只约束形状
        logits_c = logits - logits.mean(dim=-1, keepdim=True)
        log_target = torch.log(target_weights + 1e-8)
        log_target_c = log_target - log_target.mean(dim=-1, keepdim=True)

        mse_loss = F.mse_loss(logits_c, log_target_c)  # 主损失：形状对齐

        # 辅助：softmax 交叉熵（数值尺度校准，权重 0.1）
        weights = F.softmax(logits, dim=-1)
        kl_loss = F.kl_div(torch.log(weights + 1e-8), target_weights, reduction='batchmean')
        total_loss = mse_loss + 0.1 * kl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.step_count += 1
        return {"kl_loss": kl_loss.item(), "mse_loss": mse_loss.item(), "kl_total_loss": total_loss.item()}



    # TD 训练
    def TD_train_step(
        self,
        train_buffer: TrainingBuffer ,
        gamma: float ,
        v_online = None,
        v_target_next = None,
    ) -> dict:
        """
        训练一步

        输入:
            train_buffer: TrainingBuffer，含 weights 和 weights_entropy（保留计算图）
            gamma: float
            v_online: [N] V(s_1)...V(s_N)，应已断开 Critic 计算图
            v_target_next: [N] V(s_2)...V(s_{N+1})
        """
        self.network.train()

        # 转为 tensor
        if v_online is not None:
            if not isinstance(v_online, torch.Tensor):
                v_online_t = torch.FloatTensor(v_online).to(self.device)
            else:
                v_online_t = v_online.to(self.device)
        else:
            v_online_t = None
        if v_target_next is not None:
            if not isinstance(v_target_next, torch.Tensor):
                v_target_next_t = torch.FloatTensor(v_target_next).to(self.device)
            else:
                v_target_next_t = v_target_next.to(self.device)
        else:
            v_target_next_t = None
        # gamma 转为 tensor

        # 前向传播
        self.optimizer.zero_grad()
        loss, metrics = self.network.compute_td_loss(
            train_buffer, gamma, v_online_t, v_target_next_t,
        )

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.step_count += 1

        return metrics

    def update_learning_rate(self, lr: float):
        """更新优化器学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step_count': self.step_count,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.step_count = checkpoint['step_count']

    def get_parameters(self) -> dict:
        """获取网络参数数量"""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
        }


