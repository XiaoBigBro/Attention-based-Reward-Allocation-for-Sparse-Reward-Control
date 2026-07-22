"""
FETCH 环境改造 + 课程学习接口
- 回合级稀疏奖励（中间步 reward=0，回合末 reward=ΣL^p）
- 4 档预设难度（0~3），可动态切换
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — register FetchReach/Push envs
import numpy as np


# ---------------------------------------------------------------------------
# 1. 课程难度参数
# ---------------------------------------------------------------------------

NUM_CURRICULUM_LEVELS = 4  # 等级 0.1.2.3


# 是一个代码生成器，它让你专注于定义“有哪些数据”（属性），
# 而自动帮你生成“如何处理这些数据”（初始化、打印、比较等方法）
@dataclass
class EnvCurriculumParams:
    """控制难度的参数（单位：米）。"""
    success_threshold: float = 0.20   #成功判定距离 d
    min_segment_length: int = 1#奖励连续段长度小于min_segment_length的段不计算奖励
    max_episode_steps: int = 100#最大回合步数
    target_range: float = 0.10#目标区所在的位置范围大小
    segment_scale: float = 1.0#奖励系数 K
    reward_power: float = 1.3#指数奖励因子 p

CURRICULUM_PRESETS = {
    0: EnvCurriculumParams(target_range=0.10, success_threshold=0.20, max_episode_steps=100, min_segment_length=1),
    1: EnvCurriculumParams(target_range=0.15, success_threshold=0.15, max_episode_steps=100, min_segment_length=1),
    2: EnvCurriculumParams(target_range=0.20, success_threshold=0.10, max_episode_steps=100, min_segment_length=1),
    3: EnvCurriculumParams(target_range=0.25, success_threshold=0.05, max_episode_steps=100, min_segment_length=1),
}


# ---------------------------------------------------------------------------
# 2. 奖励计算
# ---------------------------------------------------------------------------

def compute_terminal_reward(
    dwell_segments: list[int],
    *,
    params: EnvCurriculumParams,
) -> float:
    """回合结束时计算最终奖励。"""
    return params.segment_scale * sum(
        L ** params.reward_power
        for L in dwell_segments
        if L >= params.min_segment_length
    )


# ---------------------------------------------------------------------------
# 3. 核心 Wrapper
# ---------------------------------------------------------------------------
#1
class EpisodicDwellRewardWrapper(gym.Wrapper):
    """
    将 FetchReach/Push 改造为严格回合级稀疏奖励。

    行为：
    - 每步 reward = 0（中间无反馈）
    - 回合固定跑满 T 步（不因提前到达目标而提前终止）
    - 每步判定是否在目标区（||achieved_goal - desired_goal|| < ε）
    - 离开目标区则连续计数清零，每段长度记录到 dwell_segments
    - 末步按 R_final = sum(L^p) 返回累计奖励
    """

    def __init__(
        self,
        env: gym.Env,
        curriculum_level: int = 0,
    ):#初始化EpisodicDwellRewardWrapper这个类的时候，得传入env、curriculum_level这两个参数
        super().__init__(env)#这里的env就是__init__方法中的env参数，调用父类gym.Wrapper的__init__方法
        self._curriculum_level = curriculum_level
        self._params = CURRICULUM_PRESETS[curriculum_level]

        # 回合状态
        self._episode_step: int = 0#当前回合的步数
        self._current_segment_len: int = 0#当前连续段的长度
        self._dwell_segments: list[int] = []#连续段列表

        # 同步 target_range 到底层 env（供 _sample_goal 使用）
        self._sync_target_range()#同步 target_range 到底层 env（供 _sample_goal 使用）

    # ------------------------------------------------------------------
    # 课程接口（Step 2 直接调用）
    # ------------------------------------------------------------------

    @property
    def curriculum_level(self) -> int:
        return self._curriculum_level

    def set_curriculum_level(self, level: int) -> None:
        """切换到指定难度等级（0=易, 1=中, 2=难,3=最难）。"""
        if level not in CURRICULUM_PRESETS:
            raise ValueError(f"level must be 0..{NUM_CURRICULUM_LEVELS - 1}, got {level}")
        self._curriculum_level = level
        self._params = CURRICULUM_PRESETS[level]
        self._sync_target_range()

    def get_curriculum_params(self) -> EnvCurriculumParams:
        return self._params#返回当前难度等级对应的参数

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _sync_target_range(self) -> None:
        #self 必须定义在括号中，才能在类的方法中使用
        """
        将 target_range 写入底层 Fetch env 的 unwrapped，
        使 _sample_goal() 采样到对应范围的目标。
        """
        try:
            unwrapped = self.unwrapped
            if hasattr(unwrapped, "target_range"):
                unwrapped.target_range = self._params.target_range
        except Exception:
            # FetchPush 等可能属性不同，静默忽略
            pass

    def _is_at_goal(self, obs: dict) -> bool:
        """判定当前末端/物体是否在目标区内。"""
        achieved = obs.get("achieved_goal")
        desired = obs.get("desired_goal")
        if achieved is None or desired is None:
            return False
        dist = float(np.linalg.norm(achieved - desired))
        return dist < self._params.success_threshold

    def _flush_current_segment(self) -> None:
        """收尾当前连续段（回合结束时调用）。"""
        if self._current_segment_len > 0:
            self._dwell_segments.append(self._current_segment_len)

    def _reset_episode_state(self) -> None:
        """reset() / 新回合开始时重置计数。"""
        self._episode_step = 0
        self._current_segment_len = 0
        self._dwell_segments = []

    # ------------------------------------------------------------------
    # gym.Wrapper 覆写
    # ------------------------------------------------------------------

    def reset(self, **kwargs):#**kwargs是关键字参数，可以传入任意数量的关键字参数
        self._reset_episode_state()
        # 先同步 target_range，再 reset（确保本回合目标按新参数采样）
        self._sync_target_range()
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # 丢弃底层 reward，用我们自己的
        obs, _, terminated, truncated, info = self.env.step(action)
        raw_reward = 0.0  # 屏蔽原生逐步奖励

        # 判定是否在目标区
        at_goal = self._is_at_goal(obs)

        # 更新连续段计数
        if at_goal:
            self._current_segment_len += 1
        else:
            if self._current_segment_len > 0:
                self._dwell_segments.append(self._current_segment_len)
            self._current_segment_len = 0#如果不在目标区，则当前连续段长度清零

        self._episode_step += 1

        # 屏蔽底层提前终止（强制跑满 T 步）
        terminated = False

        # 回合截断判断
        truncated = self._episode_step >= self._params.max_episode_steps

        # 末步结算
        if truncated:
            self._flush_current_segment()
            raw_reward = compute_terminal_reward(
                self._dwell_segments,
                params=self._params,
            )

        # info 扩展
        info["terminal_reward"] = raw_reward if truncated else 0.0
        info["dwell_segments"] = self._dwell_segments if truncated else list(self._dwell_segments)
        info["curriculum_level"] = self._curriculum_level
        info["curriculum_params"] = self._params
        info["is_at_goal"] = at_goal
        info["episode_step"] = self._episode_step

        return obs, raw_reward, terminated, truncated, info
        #返回观测、奖励（为0，最后一步有奖励值）、终止(为False)、截断(如果回合步数达到最大步数，则截断为True，其余为False)、
        # 信息（包含当前难度等级、当前难度等级对应的参数、当前是否在目标区、当前回合步数、连续段列、最终奖励）


# ---------------------------------------------------------------------------
# 5. 环境工厂
# ---------------------------------------------------------------------------
#1
def make_fetch_env(
    env_id: str = "FetchReach-v4",
    curriculum_level: int = 0,
    render_mode: Optional[str] = None,
) -> EpisodicDwellRewardWrapper:
    """
    创建 FetchReach-v3 / FetchPush-v3 环境，并包上 EpisodicDwellRewardWrapper。

    Parameters
    ----------
    env_id : str
        环境 ID，如 "FetchReach-v3" / "FetchPush-v3"
    curriculum_level : int
        难度等级 0（易）/ 1（中）/ 2（难）/3（最难）
    render_mode : str, optional
        渲染模式，如 "human" / "rgb_array"

    Returns
    -------
    EpisodicDwellRewardWrapper
    """
    base_env = gym.make(env_id, render_mode=render_mode, max_episode_steps=10000)
    wrapped = EpisodicDwellRewardWrapper(base_env, curriculum_level)
    return wrapped
#1
def get_env_obs_dim(env_id: str) -> int:
    """查询环境的观测维度（不创建渲染窗口）。"""
    dummy = gym.make(env_id, render_mode=None)
    try:
        obs, _ = dummy.reset()
        dim = obs["observation"].shape[0] + obs["achieved_goal"].shape[0] + obs["desired_goal"].shape[0]
    finally:
        dummy.close()
    return dim
#1
def get_env_action_dim(env_id: str) -> int:
    """查询环境的动作维度（不创建渲染窗口）。"""
    dummy = gym.make(env_id, render_mode=None)
    try:
        dummy.reset()
        dim = dummy.action_space.shape[0]
    finally:
        dummy.close()
    return dim
