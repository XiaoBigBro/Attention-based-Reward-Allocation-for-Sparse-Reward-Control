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
    success_threshold: float = 0.08  #成功判定距离 d
    target_range: float = 0.28#目标区所在的位置范围大小
    max_episode_steps: int = 80#最大回合步数
    dense_reward: bool = True#是否使用密集奖励模式

CURRICULUM_PRESETS = {
    0: EnvCurriculumParams(target_range=0.28, success_threshold=0.08, max_episode_steps=80,dense_reward=True),
    1: EnvCurriculumParams(target_range=0.30, success_threshold=0.07, max_episode_steps=80,dense_reward=False),
    2: EnvCurriculumParams(target_range=0.32, success_threshold=0.06, max_episode_steps=80,dense_reward=False),
    3: EnvCurriculumParams(target_range=0.34, success_threshold=0.05, max_episode_steps=80,dense_reward=False),
}

# ---------------------------------------------------------------------------
# 2. 奖励计算
# ---------------------------------------------------------------------------
#密集奖励函数
def compute_step_reward(
    prev_dist: float,
    cur_dist: float,
    threshold: float,
) -> float:
    """增量式密集奖励：推进奖励 + 到达保持奖励，恒正。

    分段公式:
    - 区外 (cur_dist >= threshold): r = 1 + tanh(k * (prev_dist - cur_dist))
      - 推进 Δd>0（靠近目标）: r ∈ (1, 2)
      - 后退 Δd<0（远离目标）: r ∈ (0, 1)
    - 区内 (cur_dist < threshold): r = 1 + reached_bonus（恒定）
      - 到达目标区域后任务已完成，区内微小漂移不再按 Δd 计分（避免抖动噪声）
      - 只有滑出目标区才会跌回区外公式，承受"丢保持奖励 + 后退惩罚"双重损失
    """
    if cur_dist < threshold:
        # 已到达目标区域：恒定保持奖励（区内微小位移不计 Δd）
        return 1.0 + 1.5
    delta = prev_dist - cur_dist
    k = 20.0 if delta >= 0 else 50.0
    return 1.0 + float(np.tanh(k * delta))

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
    - step() 返回 (obs, terminated, truncated, info) 四元组（不含 reward）；
      每步即时奖励写入 info["step_rewards"]，回合末写入 info["terminal_reward"]。
    - dense_reward 仅标识档位（Level 0 密集档 / Level 1+ 稀疏档），
      奖励计算与档位无关，各档都按 CURRICULUM_PRESETS 的 success_threshold 计算增量奖励。
    - 回合固定跑满 max_episode_steps 步（不因提前到达目标而提前终止）。
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
        self._prev_dist: Optional[float] = None  # 上一步末端到目标的距离（增量式奖励用）

        # 同步 target_range 到底层 env（供 _sample_goal 使用）
        self._sync_target_range()#同步 target_range 到底层 env（供 _sample_goal 使用）

    # ------------------------------------------------------------------
    # 课程接口
    # ------------------------------------------------------------------

    def set_curriculum_level(self, level: int, ) -> None:
        """切换到指定难度等级（0..5；0 为密集档，1..5 为稀疏档）。"""
        if level not in CURRICULUM_PRESETS:
            raise ValueError(f"level must be 0..{NUM_CURRICULUM_LEVELS - 1}, got {level}")
        self._curriculum_level = level
        self._params = CURRICULUM_PRESETS[level]
        self._dense_reward = self._params.dense_reward  # Level 0 用稠密，Level 1+ 用稀疏
        self._sync_target_range()

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
        self._prev_dist = None

    # ------------------------------------------------------------------
    # gym.Wrapper 覆写
    # ------------------------------------------------------------------

    def reset(self, **kwargs):#**kwargs是关键字参数，可以传入任意数量的关键字参数
        self._reset_episode_state()
        # 先同步 target_range，再 reset（确保本回合目标按新参数采样）
        self._sync_target_range()
        obs, info = self.env.reset(**kwargs)
        # 增量式奖励需要初始距离 d(s_0)（回合起点）
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        self._prev_dist = float(np.linalg.norm(achieved - desired))
        return obs, info

    def step(self, action):
        # 丢弃底层 reward，用我们自己的
        obs, _, terminated, truncated, info = self.env.step(action)

        # 计算当前末端到目标的距离
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        cur_dist = float(np.linalg.norm(achieved - desired))

        # 是否到达目标区域（增量式奖励的到达项判定）
        reached = cur_dist < self._params.success_threshold

        # 增量式密集奖励：区内恒定 / 区外增量推进
        step_reward = compute_step_reward(
            self._prev_dist, cur_dist,
            self._params.success_threshold,
        )
        self._prev_dist = cur_dist

        # 存储即时奖励
        self._step_rewards.append(step_reward)

        self._episode_step += 1

        # 屏蔽底层提前终止（强制跑满 T 步）
        terminated = False

        # 回合截断判断
        truncated = self._episode_step >= self._params.max_episode_steps

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

    base_env = gym.make(env_id, render_mode=render_mode, max_episode_steps=1000)
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