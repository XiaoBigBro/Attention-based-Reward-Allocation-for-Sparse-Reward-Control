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
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_causal_mask: bool = False,
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
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len = query.size(1)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Causal Mask: 防止看到未来位置
        if self.use_causal_mask:
            causal = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        return self.W_o(context)


# --- TransformerLayer ---

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

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        输入: x [batch, seq_len, d_model]
        输出: [batch, seq_len, d_model]
        """
        # 自注意力 + 残差连接
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + 残差连接
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层

    结构: Cross-Attention → Add & LayerNorm → FFN → Add & LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 交叉注意力: Query 来自 state, Key/Value 来自 action
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # 来自状态
        kv: torch.Tensor,     # 来自动作
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        输入:
            query: [batch, seq_len, d_model] (状态)
            kv: [batch, seq_len, d_model] (动作)
        输出: [batch, seq_len, d_model]
        """
        # 交叉注意力 + 残差连接
        attn_out = self.cross_attn(query, kv, kv, mask)
        x = self.norm1(query + self.dropout(attn_out))

        # FFN + 残差连接
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


# --- RewardAllocator ---

class RewardAllocator(nn.Module):
    """基于注意力机制的奖励分配网络。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        entropy_coef: float = 0.01,
        max_seq_len: int = 500,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.entropy_coef = entropy_coef

        # 编码层
        self.state_encoder = nn.Linear(obs_dim, d_model)
        self.action_encoder = nn.Linear(action_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # 状态处理分支 (仅 Padding Mask)
        self.state_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout, use_causal_mask=False)
            for _ in range(num_layers)
        ])

        # 动作处理分支 (Padding Mask + Causal Mask)
        self.action_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout, use_causal_mask=True)
            for _ in range(num_layers)
        ])

        # 交叉注意力层
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播

        输入:
            states: [T, obs_dim] 或 [batch, T, obs_dim]
            actions: [T, action_dim] 或 [batch, T, action_dim]
            mask: [T] 或 [batch, T]

        输出:
            weights: [T] 或 [batch, T]  奖励权重
            metrics: dict  包含熵等指标
        """
        # 处理单序列输入
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = states.shape

        # 编码
        state_emb = self.state_encoder(states)
        state_emb = self.pos_encoder(state_emb)
        action_emb = self.action_encoder(actions)

        # 状态/动作自注意力处理
        state_hidden = state_emb
        for layer in self.state_layers:
            state_hidden = layer(state_hidden, mask)

        action_hidden = action_emb
        for layer in self.action_layers:
            action_hidden = layer(action_hidden, mask)

        # 交叉注意力融合
        fused = state_hidden
        for layer in self.cross_layers:
            fused = layer(fused, action_hidden, mask)

        # 输出权重 (softmax)
        logits = self.output_proj(fused).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(logits, dim=-1)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)

        if squeeze_output:
            weights = weights.squeeze(0)
            entropy = entropy.squeeze(0)

        return weights, {'entropy': entropy.mean().item() if isinstance(entropy, torch.Tensor) else entropy, 'seq_len': seq_len}

    def compute_td_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        terminal_reward: float,
        gamma: float,
        v_online: torch.Tensor,
        v_target_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """计算 TD 误差损失 + 熵正则化。"""
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            if v_online.dim() == 1:
                v_online = v_online.unsqueeze(0)
            if v_target_next.dim() == 1:
                v_target_next = v_target_next.unsqueeze(0)

        weights, metrics = self._forward_no_squeeze(states, actions, mask)

        # TD error
        r_allocated = weights * terminal_reward
        td_target = r_allocated + gamma * v_target_next[:, :-1]
        td_error = td_target - v_online
        if mask is not None:
            td_error = td_error * mask

        # TD 损失 + 熵正则化
        td_loss = (td_error ** 2).sum() / mask.sum() if mask is not None else (td_error ** 2).mean()
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
        return total_loss, metrics

    def _forward_no_squeeze(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len, _ = states.shape

        state_emb = self.state_encoder(states)
        state_emb = self.pos_encoder(state_emb)
        action_emb = self.action_encoder(actions)

        state_hidden = state_emb
        for layer in self.state_layers:
            state_hidden = layer(state_hidden, mask)

        action_hidden = action_emb
        for layer in self.action_layers:
            action_hidden = layer(action_hidden, mask)

        fused = state_hidden
        for layer in self.cross_layers:
            fused = layer(fused, action_hidden, mask)

        logits = self.output_proj(fused).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(logits, dim=-1)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)

        return weights, {'entropy': entropy.mean().item() if isinstance(entropy, torch.Tensor) else entropy, 'seq_len': seq_len}


# --- RewardAllocatorAgent ---

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
        mask: np.ndarray = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        分配奖励

        输入:
            states: [T, obs_dim]
            actions: [T, action_dim]
            mask: [T] (可选)

        输出:
            weights: [T]
            metrics: dict
        """
        self.network.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.FloatTensor(actions).to(self.device)
            mask_t = torch.FloatTensor(mask).to(self.device) if mask is not None else None

            weights, metrics = self.network(states_t, actions_t, mask_t)

            # 转为 numpy
            weights = weights.cpu().numpy()

        return weights, metrics

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        mask: np.ndarray,
        terminal_reward: float,
        gamma: float,
        v_online: np.ndarray,
        v_target_next: np.ndarray,
    ) -> dict:
        """
        训练一步

        输入:
            states: [T, obs_dim]
            actions: [T, action_dim]
            mask: [T]
            terminal_reward: float
            gamma: float
            v_online: [T]
            v_target_next: [T+1]
        """
        self.network.train()

        # 转为 tensor
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        mask_t = torch.FloatTensor(mask).to(self.device)
        v_online_t = torch.FloatTensor(v_online).to(self.device)
        v_target_next_t = torch.FloatTensor(v_target_next).to(self.device)

        # 前向传播 (直接调用 compute_td_loss)
        self.optimizer.zero_grad()
        loss, metrics = self.network.compute_td_loss(
            states_t, actions_t, mask_t,
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


# --- Utilities ---

def pad_sequence(
    observations: List[np.ndarray],
    actions: List[np.ndarray],
    max_len: int = None,
    padding_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    填充序列到统一长度

    输入:
        observations: [s1, s2, ..., sT]  每个是 [obs_dim]
        actions: [a1, a2, ..., aT]  每个是 [action_dim]
        max_len: 最大长度 (默认取实际长度)

    输出:
        obs_padded: [max_len, obs_dim]
        act_padded: [max_len, action_dim]
        mask: [max_len]  (1=有效, 0=填充)
    """
    seq_len = len(observations)
    if max_len is None:
        max_len = seq_len

    obs_dim = observations[0].shape[-1] if observations[0].ndim > 0 else 1
    act_dim = actions[0].shape[-1] if actions[0].ndim > 0 else 1

    # 初始化填充数组
    obs_padded = np.full((max_len, obs_dim), padding_value, dtype=np.float32)
    act_padded = np.full((max_len, act_dim), padding_value, dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.float32)

    # 复制有效数据
    actual_len = min(seq_len, max_len)
    for i in range(actual_len):
        obs = observations[i]
        act = actions[i]
        if obs.ndim == 0:
            obs = np.array([obs])
        if act.ndim == 0:
            act = np.array([act])
        obs_padded[i] = obs
        act_padded[i] = act
        mask[i] = 1.0

    return obs_padded, act_padded, mask
