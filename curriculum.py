"""
自适应课程学习模块
根据奖励滑动窗口均值自动升降级
"""

from __future__ import annotations

from typing import Dict
from collections import deque
import numpy as np

from fetchreach_env import NUM_CURRICULUM_LEVELS


# --- SlidingWindowStats ---
class SlidingWindowStats:

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


    def is_full(self) -> bool:
        """窗口是否已填满"""
        return len(self._buffer) >= self.window_size

    def count(self) -> int:
        """当前数据量"""
        return len(self._buffer)

"""基于滑动窗口奖励的自动课程学习（升降级）。"""
class AdaptiveCurriculumManager:

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
        self.reward_window = SlidingWindowStats(window_size)

    def on_episode_end(
        self,
        terminal_reward: float,
    ) -> bool:

        self.episode_count += 1
        self.level_episode_counts[self.current_level] += 1

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        self.reward_window.push(terminal_reward)

        return self._check_curriculum_change()

    """检查是否触发课程切换（窗口均值 + 冷却 + 阈值判断）。"""
    def _check_curriculum_change(self) -> bool:

        if not self.reward_window.is_full():
            return False

        if self.cooldown_remaining > 0:
            return False

        mean_reward = self.reward_window.mean()

        if mean_reward >= (self.promote_threshold - self.current_level * 20 ) and self.current_level < NUM_CURRICULUM_LEVELS - 1:
            return self._promote(mean_reward, self.promote_threshold)

        if mean_reward <= self.demote_threshold and self.current_level > 0:
            return self._demote(mean_reward, self.demote_threshold)

        return False

    def _promote(self, mean_reward: float, threshold: float) -> bool:
        """执行升级操作。"""

        self.current_level += 1
        self.cooldown_remaining = self.promote_cooldown
        self.level_episode_counts[self.current_level] = 0
        self.reward_window.clear()

        return True

    def _demote(self, mean_reward: float, threshold: float) -> bool:
        """执行降级操作。"""

        self.current_level -= 1
        self.cooldown_remaining = self.demote_cooldown
        self.level_episode_counts[self.current_level] = 0
        self.reward_window.clear()

        return True

    def get_current_level(self) -> int:
        """获取当前课程等级"""
        return self.current_level
