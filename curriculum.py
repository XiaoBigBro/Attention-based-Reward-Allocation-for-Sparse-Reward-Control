"""
自适应课程学习模块
- 根据奖励滑动窗口均值自动升降级
- 定期回放低等级防止遗忘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque
import numpy as np

from fetch_env import (
    EnvCurriculumParams,
    CURRICULUM_PRESETS,
    NUM_CURRICULUM_LEVELS,
)


# --- SlidingWindowStats ---

class SlidingWindowStats:
    """固定大小的滑动窗口，自动计算统计量。"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)

    def push(self, value: float) -> None:
        """压入新值"""
        self._buffer.append(value)

    def clear(self) -> None:
        """清空窗口"""
        self._buffer.clear()

    def mean(self) -> float:
        """平均值"""
        if len(self._buffer) == 0:
            return 0.0
        return float(np.mean(self._buffer))

    def std(self) -> float:
        """标准差"""
        if len(self._buffer) <= 1:
            return 0.0
        return float(np.std(self._buffer))

    def min(self) -> float:
        """最小值"""
        if len(self._buffer) == 0:
            return 0.0
        return float(np.min(self._buffer))

    def max(self) -> float:
        """最大值"""
        if len(self._buffer) == 0:
            return 0.0
        return float(np.max(self._buffer))

    def median(self) -> float:
        """中位数"""
        if len(self._buffer) == 0:
            return 0.0
        return float(np.median(self._buffer))

    def is_full(self) -> bool:
        """窗口是否已填满"""
        return len(self._buffer) >= self.window_size

    def count(self) -> int:
        """当前数据量"""
        return len(self._buffer)

    def get_stats(self) -> Dict[str, float]:
        """返回所有统计量"""
        return {
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min(),
            "max": self.max(),
            "median": self.median(),
            "count": self.count(),
            "is_full": self.is_full(),
        }

# --- CurriculumEvent ---

@dataclass
class CurriculumEvent:
    """课程等级变化事件"""
    episode: int
    old_level: int
    new_level: int
    reason: str  # "promote", "demote"
    reward_mean: float
    reward_threshold: float

# --- AdaptiveCurriculumManager ---

class AdaptiveCurriculumManager:
    """基于滑动窗口奖励的自动课程学习（升降级 + 回放）。"""

    def __init__(
        self,
        window_size: int,
        promote_threshold: float,
        demote_threshold: float,
        promote_cooldown: int,
        demote_cooldown: int,
        replay_interval: int,
        auto_adjust_thresholds: bool = True,
    ):
        # 配置
        self.window_size = window_size
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.promote_cooldown = promote_cooldown
        self.demote_cooldown = demote_cooldown
        self.replay_interval = replay_interval
        self.auto_adjust_thresholds = auto_adjust_thresholds

        # 状态
        self.current_level: int = 0
        self.episode_count: int = 0
        self.level_episode_counts: Dict[int, int] = {i: 0 for i in range(NUM_CURRICULUM_LEVELS)}#记录每个等级训练的回合数

        # 冷却（初始按升级冷却算）
        self.cooldown_remaining: int = self.promote_cooldown
        self.since_last_change: int = 0

        # 滑动窗口
        self.reward_window = SlidingWindowStats(window_size)

        # 事件历史
        self.event_history: List[CurriculumEvent] = []

        # 回放状态
        self._replay_flag: bool = False
        self._replay_levels: List[int] = []  # 保存所有需要回放的等级（轮流回放）
        self._replay_level_index: int = 0     # 当前回放的等级索引
        self._replay_episode_count: int = 0   # 当前等级回放的回合数

    @property
    def progress(self) -> float:
        """当前等级的训练进度 [0, 1]，基于冷却进度。"""
        if self.promote_cooldown == 0:
            return 1.0
        return 1.0 - (self.cooldown_remaining / self.promote_cooldown)

    def on_episode_end(
        self,
        terminal_reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self.episode_count += 1
        self.level_episode_counts[self.current_level] += 1
        self.since_last_change += 1

        # 冷却递减
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        # 判断是否在回放模式
        in_replay = self._replay_flag and len(self._replay_levels) > 0

        # 只有不在回放模式时，才将奖励压入滑动窗口
        # 因为回放的奖励是其他等级的，不应该影响当前等级的判断
        if not in_replay:
            self.reward_window.push(terminal_reward)

        # 推进回放
        if in_replay:
            self._replay_episode_count += 1

            # 每个等级回放5回合后切换到下一个等级
            if self._replay_episode_count >= 5:
                self._replay_episode_count = 0
                self._replay_level_index += 1

                # 如果已经轮流完所有等级，结束回放
                if self._replay_level_index >= len(self._replay_levels):
                    self._replay_flag = False
                    self._replay_levels = []
                    self._replay_level_index = 0

        # 检查是否触发新一回放（不在回放时才检查）
        if not in_replay:
            self._check_replay()

        # 检查是否触发课程切换（不在回放时才检查）
        if not in_replay:
            return self._check_curriculum_change(terminal_reward)
        return False

    def _check_replay(self) -> None:
        """
        检查是否应该触发回放

        工作流程：
        1. 每隔 replay_interval 回合触发一次回放
        2. 收集所有比当前等级低的已训练等级
        3. 设置回放标志，开始轮流回放这些等级
        """
        if self.episode_count > 0 and self.episode_count % self.replay_interval == 0:
            # 回放到之前训练过的所有较低等级（轮流回放）
            replay_levels = [l for l in range(self.current_level) if self.level_episode_counts.get(l, 0) > 0]
            if replay_levels:
                self._replay_flag = True
                self._replay_levels = replay_levels  # 保存所有需要回放的等级
                self._replay_level_index = 0  # 从第一个等级开始回放
                print(f"\n*** 进入课程回放模式! 将依次回放等级: {replay_levels} ***\n")

    def _check_curriculum_change(self, current_reward: float) -> bool:
        """检查是否触发课程切换（窗口均值 + 冷却 + 阈值判断）。"""
        if not self.reward_window.is_full():
            return False

        # 冷却中不切换（冷却结束后才能再次切换）
        if self.cooldown_remaining > 0:
            return False

        mean_reward = self.reward_window.mean()

        # 获取根据等级调整后的阈值
        promote_th = self._get_adjusted_threshold(self.promote_threshold, direction="promote")
        demote_th = self._get_adjusted_threshold(self.demote_threshold, direction="demote")

        # 尝试升级
        if mean_reward >= promote_th and self.current_level < NUM_CURRICULUM_LEVELS - 1:
            return self._promote(mean_reward, promote_th)

        # 尝试降级
        if mean_reward <= demote_th and self.current_level > 0:
            return self._demote(mean_reward, demote_th)

        return False

    def _get_adjusted_threshold(self, base_threshold: float, direction: str) -> float:
        """
        根据当前等级调整阈值

        原理：不同等级奖励上限不同，需要按比例调整阈值
        - 等级越高，奖励上限越大，阈值也应该越大
        - 调整比例 = 当前等级T / 基准T（等级0的T）

        示例：
        - 等级0: T=30, 阈值 = 100
        - 等级1: T=70, 阈值 = 100 * (70/30) = 233
        - 等级2: T=150, 阈值 = 100 * (150/30) = 500
        """
        if not self.auto_adjust_thresholds:
            return base_threshold

        # 基础奖励范围（根据 CURRICULUM_PRESETS 估算）
        # Level 0: T=30, 可能的最大连续段 ≈ 30, 奖励 ≈ 30^2 = 900
        # Level 1: T=70, 可能 ≈ 70^2 = 4900
        # Level 2: T=150, 可能 ≈ 150^2 = 22500

        # 简化：用 T 的平方作为奖励上限的比例
        level_T = CURRICULUM_PRESETS[self.current_level].max_episode_steps
        base_T = CURRICULUM_PRESETS[0].max_episode_steps

        # 按 T 比例调整
        adjustment = (level_T / base_T) if direction == "promote" else (base_T / level_T)

        return base_threshold * adjustment

    def _promote(self, mean_reward: float, threshold: float) -> bool:
        """执行升级操作。"""
        old_level = self.current_level
        self.current_level += 1

        # 重置冷却（升级后冷却较长）
        self.cooldown_remaining = self.promote_cooldown
        self.level_episode_counts[self.current_level] = 0

        # 清空窗口，准备收集新等级的数据
        self.reward_window.clear()

        # 记录事件
        event = CurriculumEvent(
            episode=self.episode_count,
            old_level=old_level,
            new_level=self.current_level,
            reason="promote",
            reward_mean=mean_reward,
            reward_threshold=threshold,
        )
        self.event_history.append(event)

        return True

    def _demote(self, mean_reward: float, threshold: float) -> bool:
        """执行降级操作。"""
        old_level = self.current_level
        self.current_level -= 1

        # 重置冷却（降级后冷却较短）
        self.cooldown_remaining = self.demote_cooldown
        self.level_episode_counts[self.current_level] = 0

        # 清空窗口
        self.reward_window.clear()

        # 记录事件
        event = CurriculumEvent(
            episode=self.episode_count,
            old_level=old_level,
            new_level=self.current_level,
            reason="demote",
            reward_mean=mean_reward,
            reward_threshold=threshold,
        )
        self.event_history.append(event)

        return True

    def get_current_level(self) -> int:
        """获取当前课程等级"""
        return self.current_level

    def get_params(self) -> EnvCurriculumParams:
        """获取当前等级对应的参数"""
        return CURRICULUM_PRESETS[self.current_level]

    def should_replay(self) -> bool:
        """是否应该进行回放"""
        return self._replay_flag

    def get_replay_level(self) -> Optional[int]:
        """获取当前要回放的等级，轮流回放。"""
        if not self._replay_flag:
            return None
        # 轮流返回每个需要回放的等级
        return self._replay_levels[self._replay_level_index % len(self._replay_levels)]

    def reset_replay_flag(self) -> None:
        """重置回放标志，准备结束回放。"""
        self._replay_flag = False
        self._replay_levels = []
        self._replay_level_index = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        return {
            "current_level": self.current_level,
            "episode_count": self.episode_count,
            "reward_stats": self.reward_window.get_stats(),
            "cooldown_remaining": self.cooldown_remaining,
            "since_last_change": self.since_last_change,
            "level_episode_counts": dict(self.level_episode_counts),
            "progress": self.progress,
            "should_replay": self.should_replay(),
            "replay_target_level": self.get_replay_level(),
        }

    def get_event_history(self) -> List[CurriculumEvent]:
        """获取课程变化历史"""
        return list(self.event_history)

# --- CurriculumTrainerWrapper ---

class CurriculumTrainerWrapper:

    def __init__(
        self,
        trainer,
        curriculum_manager: AdaptiveCurriculumManager,
    ):
        self.trainer = trainer
        self.curriculum_manager = curriculum_manager
        self.episode = 0

    def update(self) -> Dict[str, Any]:
        """执行一次完整更新（采集 + 训练 + 课程检查）。"""
        self.episode += 1

        # 执行训练更新
        metrics = self.trainer.update()

        # 课程学习：每回合结束检查
        # - 记录奖励到滑动窗口
        # - 推进回放索引
        # - 检查是否需要切换难度
        changed = self.curriculum_manager.on_episode_end(
            metrics.get("terminal_reward", 0.0),
            {},
        )

        # 处理课程等级和回放
        # 优先级：回放 > 等级变化 > 保持原等级
        if self.curriculum_manager.should_replay():
            # 正在回放，切换到回放等级
            replay_level = self.curriculum_manager.get_replay_level()
            self.trainer.env.set_curriculum_level(replay_level)
            metrics["replay"] = True
            metrics["replay_level"] = replay_level
            metrics["curriculum_changed"] = False
        elif changed:
            # 不回放但等级变化了，切换到新等级
            new_level = self.curriculum_manager.get_current_level()
            self.trainer.env.set_curriculum_level(new_level)
            metrics["replay"] = False
            metrics["curriculum_changed"] = True
            metrics["new_level"] = new_level
        else:
            # 不回放且等级没变化，回到当前课程等级（current_level）
            current = self.curriculum_manager.get_current_level()
            self.trainer.env.set_curriculum_level(current)
            metrics["replay"] = False
            metrics["curriculum_changed"] = False

        # 合并课程统计
        metrics["curriculum_stats"] = self.curriculum_manager.get_stats()

        return metrics

    def save(self, path: str) -> None:
        """保存模型（包含课程状态）"""
        self.trainer.save(path)

    def load(self, path: str) -> None:
        """加载模型"""
        self.trainer.load(path)


# --- wrap_trainer_with_curriculum ---

def wrap_trainer_with_curriculum(trainer, **curriculum_kwargs) -> CurriculumTrainerWrapper:
    """
    将 PPOTrainer 包装为支持课程学习的版本

    示例:
        from ppo import PPOTrainer, DEFAULT_CONFIG
        from curriculum import AdaptiveCurriculumManager, wrap_trainer_with_curriculum

        env = make_fetch_env("FetchReach-v3", curriculum_level=0)
        trainer = PPOTrainer(env, DEFAULT_CONFIG)
        wrapped = wrap_trainer_with_curriculum(trainer)

        for _ in range(1000):
            metrics = wrapped.update()
            if metrics.get("curriculum_changed"):
                print(f"Level changed to {metrics['new_level']}")
    """
    curriculum_manager = AdaptiveCurriculumManager(**curriculum_kwargs)
    return CurriculumTrainerWrapper(trainer, curriculum_manager)
