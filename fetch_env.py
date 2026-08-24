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

NUM_CURRICULUM_LEVELS = 4  # 等级 0.1.2.3，其中0为密集奖励模式，1.2.3为稀疏奖励模式

# 是一个代码生成器，它让你专注于定义“有哪些数据”（属性），
# 而自动帮你生成“如何处理这些数据”（初始化、打印、比较等方法）
@dataclass
class EnvCurriculumParams:
    """控制难度的参数（单位：米）。"""
    success_threshold: float = 0.15   #成功判定距离 d
    target_range: float = 0.15#目标区所在的位置范围大小
    max_episode_steps: int = 80#最大回合步数
    dense_reward: bool = True#是否使用密集奖励模式

CURRICULUM_PRESETS = {
    0: EnvCurriculumParams(target_range=0.20, success_threshold=0.10, max_episode_steps=80,dense_reward=True),
    1: EnvCurriculumParams(target_range=0.20, success_threshold=0.10, max_episode_steps=80,dense_reward=False),
    2: EnvCurriculumParams(target_range=0.20, success_threshold=0.10, max_episode_steps=80,dense_reward=False),
    3: EnvCurriculumParams(target_range=0.25, success_threshold=0.05, max_episode_steps=80,dense_reward=False),
}


# ---------------------------------------------------------------------------
# 2. 奖励计算
# ---------------------------------------------------------------------------
#密集奖励函数
def compute_step_reward(
    obs: dict,
    threshold: float,
) -> float:
    """每步的即时奖励，基于末端与目标的距离计算。

    公式: reward = exp(-dist * 5) + (1.0 if reached else 0.0) - 0.5
    - dist < threshold 时视为到达目标区域（获得额外到达奖励）
    - 远离目标时 reward < 0，接近/到达时 reward > 0
    """
    achieved = obs["achieved_goal"]
    desired = obs["desired_goal"]
    dist = float(np.linalg.norm(achieved - desired))
    reached = dist < threshold
    reward = np.exp(-dist * 5) + (1.0 if reached else 0.0)
    return reward

#最终奖励
def compute_terminal_reward(
    step_rewards: list[float],
) -> float:
    """回合结束时计算最终奖励，等于所有即时奖励之和。"""
    return sum(step_rewards)


# ---------------------------------------------------------------------------
# 3. 核心 Wrapper
# ---------------------------------------------------------------------------

class EpisodicDwellRewardWrapper(gym.Wrapper):
    """
    将 FetchReach/Push 改造为可切换的稠密/稀疏奖励环境。

    行为：
    - Level 0 (dense_reward=True): 每步返回基于距离的即时奖励
    - Level 1+ (dense_reward=False): 中间步 reward=0，回合末返回累积奖励
    - 回合固定跑满 T 步（不因提前到达目标而提前终止）
    """

    def __init__(
        self,
        env: gym.Env,
        curriculum_level: int = 0,
    ):#初始化EpisodicDwellRewardWrapper这个类的时候，得传入env、curriculum_level这两个参数
        super().__init__(env)#这里的env就是__init__方法中的env参数，调用父类gym.Wrapper的__init__方法
        self._curriculum_level = curriculum_level
        self._params = CURRICULUM_PRESETS[curriculum_level]
        self._dense_reward = self._params.dense_reward#True时为密集奖励模式，False时为稀疏奖励模式

        # 回合状态
        self._episode_step: int = 0#当前回合的步数
        self._step_rewards: list[float] = []  # 存储每步的即时奖励

        # 同步 target_range 到底层 env（供 _sample_goal 使用）
        self._sync_target_range()#同步 target_range 到底层 env（供 _sample_goal 使用）

    # ------------------------------------------------------------------
    # 课程接口（Step 2 直接调用）
    # ------------------------------------------------------------------

    @property
    def curriculum_level(self) -> int:
        """返回当前难度等级。"""
        return self._curriculum_level

    @property
    def use_dense_reward(self) -> bool:
        """返回是否使用稠密奖励模式。"""
        return self._dense_reward

    def set_curriculum_level(self, level: int, ) -> None:
        """切换到指定难度等级（0=易, 1=中, 2=难,3=最难）。"""
        if level not in CURRICULUM_PRESETS:
            raise ValueError(f"level must be 0..{NUM_CURRICULUM_LEVELS - 1}, got {level}")
        self._curriculum_level = level
        self._params = CURRICULUM_PRESETS[level]
        self._dense_reward = self._params.dense_reward  # Level 0 用稠密，Level 1+ 用稀疏
        self._sync_target_range()

    def get_curriculum_params(self) -> EnvCurriculumParams:
        return self._params#返回当前难度等级对应的参数

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _sync_target_range(self) -> None:
        """
        将 target_range 和 distance_threshold 写入底层 Fetch env 的 unwrapped。

        必须成功写入，否则立即报错，避免静默失败导致课程难度未生效。
        """
        unwrapped = self.unwrapped
        if not hasattr(unwrapped, "target_range"):
            raise AttributeError(
                f"底层环境 {type(unwrapped).__name__} 缺少 target_range 属性"
            )
        unwrapped.target_range = self._params.target_range

        if not hasattr(unwrapped, "distance_threshold"):
            raise AttributeError(
                f"底层环境 {type(unwrapped).__name__} 缺少 distance_threshold 属性"
            )
        unwrapped.distance_threshold = self._params.success_threshold

    def _reset_episode_state(self) -> None:
        """reset() / 新回合开始时重置计数。"""
        self._episode_step = 0
        self._step_rewards = []

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

        # 计算每步的即时奖励（基于距离和当前等级的 success_threshold）
        step_reward = compute_step_reward(obs, self._params.success_threshold)

        # 存储即时奖励
        self._step_rewards.append(step_reward)

        self._episode_step += 1

        # 屏蔽底层提前终止（强制跑满 T 步）
        terminated = False

        # 回合截断判断
        truncated = self._episode_step >= self._params.max_episode_steps

        # 是否到达目标区域
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        reached = float(np.linalg.norm(achieved - desired)) < self._params.success_threshold

        # info 扩展
        info["step_reward"] = step_reward
        info["step_rewards"] = list(self._step_rewards)  # 所有即时奖励
        info["reached"] = reached  # 当前步是否到达目标
        info["terminal_reward"] = compute_terminal_reward(self._step_rewards) if truncated else 0.0
        info["curriculum_level"] = self._curriculum_level
        info["use_dense_reward"] = self._dense_reward
        info["episode_step"] = self._episode_step

        return obs, terminated, truncated, info
        #返回观测、终止(为False)、截断(如果回合步数达到最大步数，则截断为True，其余为False)、
        # 信息（包含当前难度等级、当前难度等级对应的参数、当前是否在目标区、当前回合步数、连续段列、最终奖励）


# ---------------------------------------------------------------------------
# 5. 环境工厂
# ---------------------------------------------------------------------------

def make_fetch_env(
    env_id: str ,
    curriculum_level: int ,
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
    dense_reward : bool
        是否使用稠密奖励模式。True 时每步返回即时奖励，False 时返回稀疏奖励。

    Returns
    -------
    EpisodicDwellRewardWrapper
    """
    base_env = gym.make(env_id, render_mode=render_mode, max_episode_steps=10000)
    wrapped = EpisodicDwellRewardWrapper(base_env, curriculum_level)
    return wrapped

def get_env_obs_dim(env_id: str) -> int:
    """查询环境的观测维度（不创建渲染窗口）。"""
    dummy = gym.make(env_id, render_mode=None)
    try:
        obs, _ = dummy.reset()
        dim = obs["observation"].shape[0] + obs["achieved_goal"].shape[0] + obs["desired_goal"].shape[0]
    finally:
        dummy.close()
    return dim

def get_env_action_dim(env_id: str) -> int:
    """查询环境的动作维度（不创建渲染窗口）。"""
    dummy = gym.make(env_id, render_mode=None)
    try:
        dummy.reset()
        dim = dummy.action_space.shape[0]
    finally:
        dummy.close()
    return dim
