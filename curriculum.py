"""
自适应课程学习模块
根据奖励滑动窗口均值自动升降级
"""

from __future__ import annotations

from dataclasses import dataclass
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

    def __init__(self, window_size: int):
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


@dataclass
class CurriculumEvent:
    """课程等级变化事件"""
    episode: int
    old_level: int
    new_level: int
    reason: str  # "promote", "demote"
    reward_mean: float
    reward_threshold: float


class AdaptiveCurriculumManager:
    """基于滑动窗口奖励的自动课程学习（升降级）。"""

    def __init__(
        self,
        window_size: int,
        promote_threshold: float,
        demote_threshold: float,
        promote_cooldown: int,
        demote_cooldown: int,
        current_level: int,
    ):
        self.window_size = window_size
        self.promote_threshold = promote_threshold#升级阈值
        self.demote_threshold = demote_threshold#降级阈值
        self.promote_cooldown = promote_cooldown
        self.demote_cooldown = demote_cooldown

        self.current_level: int = current_level
        self.episode_count: int = 0
        self.level_episode_counts: Dict[int, int] = {i: 0 for i in range(NUM_CURRICULUM_LEVELS)}

        self.cooldown_remaining: int = self.promote_cooldown
        self.since_last_change: int = 0

        self.reward_window = SlidingWindowStats(window_size)

        self.event_history: List[CurriculumEvent] = []

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

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        self.reward_window.push(terminal_reward)

        return self._check_curriculum_change()

    def _check_curriculum_change(self) -> bool:
        """检查是否触发课程切换（窗口均值 + 冷却 + 阈值判断）。"""
        if not self.reward_window.is_full():
            return False

        if self.cooldown_remaining > 0:
            return False

        mean_reward = self.reward_window.mean()

        if mean_reward >= self.promote_threshold and self.current_level < NUM_CURRICULUM_LEVELS - 1:
            return self._promote(mean_reward, self.promote_threshold)

        if mean_reward <= self.demote_threshold and self.current_level > 0:
            return self._demote(mean_reward, self.demote_threshold)

        return False

    def _promote(self, mean_reward: float, threshold: float) -> bool:
        """执行升级操作。"""
        old_level = self.current_level
        self.current_level += 1

        self.cooldown_remaining = self.promote_cooldown
        self.level_episode_counts[self.current_level] = 0

        self.reward_window.clear()

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

        self.cooldown_remaining = self.demote_cooldown
        self.level_episode_counts[self.current_level] = 0

        self.reward_window.clear()

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
        }

    def get_event_history(self) -> List[CurriculumEvent]:
        """获取课程变化历史"""
        return list(self.event_history)
