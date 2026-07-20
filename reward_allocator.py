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
        dropout: float = 0.1,
        bias: bool = True,#是否使用偏置
        use_causal_mask: bool = False,#防止看到未来位置
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

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        #将Q,K,V投影到多头注意力机制
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Causal Mask: 防止看到未来位置
        if self.use_causal_mask:
            # q_len = seq_len_q, k_len = seq_len_k
            # 创建下三角掩码 (q_idx >= k_idx 时允许，即 s_t 只能看到 a_0..a_t)
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
        d_ff: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = False,
    ):
        super().__init__()

        # 自注意力层
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout,
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

    注意: 默认使用因果掩蔽，防止状态看到未来的动作
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
    ):
        super().__init__()

        # 交叉注意力: Query 来自 state, Key/Value 来自 action
        # use_causal_mask=True: Q 在 t 时刻只能看到 V 的 0..t 位置（不能看未来动作）
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_causal_mask=use_causal_mask)
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
        Q = states (状态序列)
        K = action_dist (动作分布: mean + log_std)
        V = actions (实际采取的动作)

    特点:
        - K 和 V 使用简单的 MLP 编码（不需要自注意力）
        - Q 使用完整的 Transformer 块 + 交叉注意力
        - 交叉注意力添加因果掩蔽（防止看到未来动作）
        - 输出权重的第一个元素归零（无意义）
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
        max_seq_len: int = 200,#最大序列长度
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.entropy_coef = entropy_coef

        # Q 编码器: 状态编码 + 位置编码
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # K 编码器: 动作分布 (mean + log_std) -> 简单 MLP
        # 输入维度: action_dim (mean) + action_dim (log_std) = action_dim * 2
        self.dist_encoder = nn.Sequential(
            nn.Linear(action_dim * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # V 编码器: 动作编码 + 位置编码 (不使用自注意力)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Q 处理: Transformer 层 (用于状态序列)
        self.state_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout, use_causal_mask=False)
            for _ in range(num_layers)
        ])

        # 交叉注意力层 (Q attends to K, V)
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, d_ff, dropout, use_causal_mask=True)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_dist: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播

        输入:
            states: [T, obs_dim] 或 [batch, T, obs_dim]
            actions: [T, action_dim] 或 [batch, T, action_dim]
            action_dist: [T, action_dim * 2] 或 [batch, T, action_dim * 2]
                         包含 [mean, log_std] 拼接

        输出:
            weights: [T] 或 [batch, T]  奖励权重
            metrics: dict  包含熵等指标
        """
        # 处理单序列输入
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            if action_dist is not None:
                action_dist = action_dist.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = states.shape

        # 编码
        # Q = 状态
        q_emb = self.state_encoder(states)
        q_emb = self.pos_encoder(q_emb)

        # V = 动作 (简单 MLP + 位置编码)
        v_emb = self.action_encoder(actions)
        v_emb = self.pos_encoder(v_emb)

        # K = 动作分布 (简单 MLP)
        if action_dist is not None:
            k_emb = self.dist_encoder(action_dist)
        else:
            # 如果没有提供 action_dist，用零填充
            k_emb = torch.zeros(batch_size, seq_len, self.d_model, device=states.device)

        # Q 处理 (状态自注意力)
        q_hidden = q_emb
        for layer in self.state_layers:
            q_hidden = layer(q_hidden)

        # 交叉注意力融合 (Q attends to K, V)
        fused = q_hidden
        for layer in self.cross_layers:
            fused = layer(fused, k_emb, v_emb)

        # 输出权重：先丢弃第一个（对应 s_0，无意义），再 softmax
        logits = self.output_proj(fused).squeeze(-1)
        logits = logits[..., 1:]  # [T+1] → [T]
        weights = F.softmax(logits, dim=-1)

        # 计算熵
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)

        if squeeze_output:
            weights = weights.squeeze(0)
            entropy = entropy.squeeze(0)

        return weights, {'entropy': entropy.mean().item() if isinstance(entropy, torch.Tensor) else entropy, 'seq_len': seq_len - 1}

    def compute_td_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_dist: torch.Tensor = None,
        terminal_reward: float = 0.0,
        gamma: float = 0.99,
        v_online: torch.Tensor = None,
        v_target_next: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 TD 误差损失 + 熵正则化。

        输入:
            states: [T+1, obs_dim]
            actions: [T, action_dim]
            action_dist: [T, action_dim * 2] (可选)
            terminal_reward: float
            gamma: float
            v_online: [T+1] (可选)
            v_target_next: [T+1] (可选)
        """
        # 确保 states 是 batch 格式
        squeeze_input = False
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            if action_dist is not None:
                action_dist = action_dist.unsqueeze(0)
            if v_online is not None:
                v_online = v_online.unsqueeze(0)
            if v_target_next is not None:
                v_target_next = v_target_next.unsqueeze(0)
            squeeze_input = True

        # 前向传播
        weights, metrics = self._forward_no_squeeze(states, actions, action_dist)

        # TD error 计算 (如果提供了价值估计)
        if v_online is not None and v_target_next is not None:
            # weights shape: [1, T+1], terminal_reward 是标量
            r_allocated = weights * terminal_reward
            td_target = r_allocated + gamma * v_target_next
            td_error = td_target - v_online
            td_loss = (td_error ** 2).mean()
        else:
            # 如果没有价值估计，只用奖励分配
            td_loss = (weights ** 2).mean()  # 简单代理损失

        # 熵正则化
        entropy = metrics['entropy']
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.mean()
        entropy_loss = -self.entropy_coef * entropy
        total_loss = td_loss + entropy_loss

        metrics.update({
            'td_loss': td_loss.item() if isinstance(td_loss, torch.Tensor) else td_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'weights_mean': weights.mean().item(),
            'weights_max': weights.max().item(),
            'weights_min': weights.min().item(),
        })

        # 如果输入是单序列，压缩输出
        if squeeze_input:
            for k, v in metrics.items():
                if isinstance(v, float):
                    metrics[k] = v

        return total_loss, metrics

    def _forward_no_squeeze(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_dist: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len, _ = states.shape

        # Q = 状态
        q_emb = self.state_encoder(states)
        q_emb = self.pos_encoder(q_emb)

        # V = 动作
        v_emb = self.action_encoder(actions)
        v_emb = self.pos_encoder(v_emb)

        # K = 动作分布
        if action_dist is not None:
            k_emb = self.dist_encoder(action_dist)
        else:
            k_emb = torch.zeros(batch_size, actions.size(1), self.d_model, device=states.device)

        # Q 处理
        q_hidden = q_emb
        for layer in self.state_layers:
            q_hidden = layer(q_hidden)

        # 交叉注意力 (Q attends to K, V)
        # 注意: q_hidden 长度是 seq_len+1 (包含 terminal state)
        # k_emb, v_emb 长度是 seq_len
        # 需要对齐长度
        fused = q_hidden
        for layer in self.cross_layers:
            fused = layer(fused[:, :actions.size(1)], k_emb, v_emb)

        logits = self.output_proj(fused).squeeze(-1)
        logits = logits[..., 1:]  # 先丢弃再 softmax
        weights = F.softmax(logits, dim=-1)

        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)

        return weights, {'entropy': entropy.mean().item() if isinstance(entropy, torch.Tensor) else entropy, 'seq_len': seq_len - 1}


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

#1
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
        ).to(device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

        # 统计信息
        self.step_count = 0

    def allocate(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        action_dist: np.ndarray = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        分配奖励

        输入:
            states: [T+1, obs_dim]
            actions: [T, action_dim]
            action_dist: [T, action_dim * 2] (可选, 包含 mean + log_std)

        输出:
            weights: [T]
            metrics: dict
        """
        self.network.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.FloatTensor(actions).to(self.device)
            if action_dist is not None:
                action_dist_t = torch.FloatTensor(action_dist).to(self.device)
            else:
                action_dist_t = None

            weights, metrics = self.network(states_t, actions_t, action_dist_t)

            # 转为 numpy
            weights = weights.cpu().numpy()

        return weights, metrics

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        action_dist: np.ndarray = None,
        terminal_reward: float = 0.0,
        gamma: float = 0.99,
        v_online: np.ndarray = None,
        v_target_next: np.ndarray = None,
    ) -> dict:
        """
        训练一步

        输入:
            states: [T+1, obs_dim]
            actions: [T, action_dim]
            action_dist: [T, action_dim * 2] (可选)
            terminal_reward: float
            gamma: float
            v_online: [T+1] (可选)
            v_target_next: [T+1] (可选)
        """
        self.network.train()

        # 转为 tensor
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        if action_dist is not None:
            action_dist_t = torch.FloatTensor(action_dist).to(self.device)
        else:
            action_dist_t = None
        if v_online is not None:
            v_online_t = torch.FloatTensor(v_online).to(self.device)
        else:
            v_online_t = None
        if v_target_next is not None:
            v_target_next_t = torch.FloatTensor(v_target_next).to(self.device)
        else:
            v_target_next_t = None

        # 前向传播
        self.optimizer.zero_grad()
        loss, metrics = self.network.compute_td_loss(
            states_t, actions_t, action_dist_t,
            terminal_reward, gamma,
            v_online_t, v_target_next_t,
        )

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.step_count += 1

        return metrics

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


