"""
META-WORLD window-open-v3 环境改造 + 课程学习接口
- 4 档预设难度（0~3），只变窗体的初始位置与推开深度阈值，可动态切换
- 单回合固定 200 步（不因推到位而提前终止）

底层事实（metaworld 3.1.1 源码）：
  * 入口  gym.make("Meta-World/MT1", env_name="window-open-v3", max_episode_steps=T)
  * 观测  39 维扁平：obs[0:3] 末端、obs[3] 夹爪开合、obs[4:7] 把手(handle)位置、
          obs[7:11] 把手四元数、obs[18:36] 前一帧、obs[36:39] 目标位置
  * 关节  window_slide（滑动关节，纯平移，非铰链），沿 +x 推开
  * 官方成功判据：|目标x - 把手x| <= 0.05（只看 x）
  * 窗体随机范围：x∈[-0.1,0.1], y∈[0.7,0.9], z≡0.16
  * 目标点 = 窗体初始位置 + (+0.2, 0, 0)，maxPullDist = 0.2
  * grasp_success 恒为 1.0（不要求夹爪真抓，推动即可）
  * 初始剩余距离恒为 0.2（目标点随窗体平移）
"""
#windowopen

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import metaworld  # noqa: F401 — register Meta-World envs
import numpy as np

# ---------------------------------------------------------------------------
# 1. 课程难度参数
# ---------------------------------------------------------------------------

NUM_CURRICULUM_LEVELS = 4  # 等级 0.1.2.3，难度递增（窗体散布范围变大、推开深度阈值变严）

# 官方全范围（sawyer_window_open_v3.py:40-41）
WINDOW_X_CENTER = 0.0    # 窗体 x 中心（官方采样 x∈[-0.1,0.1] 的中心）
WINDOW_Y_CENTER = 0.8    # 窗体 y 中心（官方采样 y∈[0.7,0.9] 的中心）
WINDOW_Z = 0.16          # 窗体高度（固定，与 XML 一致）

# 就位点相对把手的偏移：窗沿 +x 推开，夹爪须站在把手 −x 侧才能施加推力
INPLACE_OFFSET = np.array([-0.04, 0.0, 0.0])#原本是np.array([-0.04, 0.0, 0.0])

@dataclass
class EnvCurriculumParams:
    """控制难度的参数（单位：米）。"""
    window_x_center: float = WINDOW_X_CENTER
    window_x_range: float = 0.01
    window_y_center: float = WINDOW_Y_CENTER
    window_y_range: float = 0.01
    window_z: float = WINDOW_Z
    success_threshold: float = 0.12   # 把手与目标的剩余 x 距离阈值：<= 该值视为推到位
    max_episode_steps: int = 150

CURRICULUM_PRESETS = {
    0: EnvCurriculumParams(window_x_range=0.010, window_y_range=0.010, success_threshold=0.15, max_episode_steps=150),
    1: EnvCurriculumParams(window_x_range=0.030, window_y_range=0.030, success_threshold=0.11, max_episode_steps=150),
    2: EnvCurriculumParams(window_x_range=0.060, window_y_range=0.060, success_threshold=0.08, max_episode_steps=150),
    3: EnvCurriculumParams(window_x_range=0.100, window_y_range=0.100, success_threshold=0.05, max_episode_steps=150),
}

# ---------------------------------------------------------------------------
# 2. 奖励计算
# ---------------------------------------------------------------------------

def compute_step_reward(
    prev_dist: float,
    cur_dist: float,
    prev_dist_girpper2obj: float,
    gripper_to_obj: float,
    threshold: float,
) -> float:

    # 已推到位：推得越到位（剩余距离越小），奖励越高
    if cur_dist < threshold:
        depth_ratio = 1.0 - cur_dist / threshold  # 0~1
        return 2.5 + depth_ratio  # 2.5 ~ 3.5

    # 推动奖励（把手朝目标方向推进为正，被带回为负）
    delta_pull = prev_dist - cur_dist
    r_pull = np.tanh(150.0 * delta_pull if delta_pull >= 0 else 15.0 * delta_pull)

    # 远处靠近奖励（趋势引导：只在距离还远时计）
    if gripper_to_obj > 0.09:
        delta_approach = prev_dist_girpper2obj - gripper_to_obj
        r_approach = 0.3 * np.tanh(50.0 * delta_approach)
    else:
        r_approach = 0

    # 就位奖励：越靠近"把手 −x 侧的就位点"越高
    r_inplace = 0.7 * max(0.0, 1.0 - gripper_to_obj / 0.20)
    # # 接触奖励
    # r_contact = 0.2 if gripper_to_obj < 0.08 else 0.0

    return 0.3 + r_pull + r_approach + r_inplace


def compute_terminal_reward(step_rewards: list[float]) -> float:
    """回合结束时计算最终奖励，等于所有即时奖励之和。"""
    return sum(step_rewards)

# ---------------------------------------------------------------------------
# 3. 核心 Wrapper
# ---------------------------------------------------------------------------

class EpisodicDwellRewardWrapper(gym.Wrapper):
    """
    将 Meta-World window-open-v3 改造为可切换难度的环境。

    行为：
    - step() 返回 (obs, terminated, truncated, info) 四元组（不含 reward）；
      每步即时奖励写入 info["step_rewards"]，回合末写入 info["terminal_reward"]。
    - 观测包成 {"observation": 39 维向量}，使 ppo.unpack_obs 不用改就能用。
    - 观测索引：obs[0:3] 末端位置、obs[3] 夹爪开合、obs[4:7] 把手位置、obs[36:39] 目标位置。
    - 难度只由 CURRICULUM_PRESETS 决定：窗体初始位置 + 推开深度阈值。
    - 回合固定跑满 max_episode_steps 步（推到位也不提前终止）。
    """

    def __init__(self, env: gym.Env, curriculum_level: int = 0):
        super().__init__(env)
        self._curriculum_level = curriculum_level
        self._params = CURRICULUM_PRESETS[curriculum_level]

        # 回合状态
        self._episode_step: int = 0
        self._step_rewards: list[float] = []
        self._prev_dist: Optional[float] = None
        self._prev_dist_girpper2obj: Optional[float] = None
        self._env_prepared: bool = False

        self._sync_env_params()

    # ------------------------------------------------------------------
    # 课程接口，切换难度
    # ------------------------------------------------------------------

    def set_curriculum_level(self, level: int) -> None:
        """切换到指定难度等级（0..3）。"""
        if level not in CURRICULUM_PRESETS:
            raise ValueError(f"level must be 0..{NUM_CURRICULUM_LEVELS - 1}, got {level}")
        self._curriculum_level = level
        self._params = CURRICULUM_PRESETS[level]
        self._sync_env_params()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _sync_env_params(self) -> None:
        """把底层环境准备成"可由本 wrapper 指定窗体位置"的状态。"""
        if not self._env_prepared:
            tasks = self.env.get_wrapper_attr("tasks")
            if not tasks:
                raise RuntimeError("底层 Meta-World 环境没有可用 task")
            self.env.get_wrapper_attr("set_task")(tasks[0])
            self.env.get_wrapper_attr("toggle_sample_tasks_on_reset")(False)
            self._unwrapped = self.env.unwrapped

            obs_shape = tuple(self.env.observation_space.shape)
            act_shape = tuple(self.env.action_space.shape)
            if obs_shape != (39,):
                raise ValueError(f"预期观测 39 维，实际 {obs_shape}")
            if act_shape != (4,):
                raise ValueError(f"预期动作 4 维，实际 {act_shape}")
            self._env_prepared = True

        spec = getattr(self.env, "spec", None)
        limit = getattr(spec, "max_episode_steps", None)
        if limit is not None and limit < self._params.max_episode_steps:
            raise ValueError(
                f"底层 max_episode_steps={limit} 小于本档 {self._params.max_episode_steps}，"
                f"请调大环境工厂里的 max_episode_steps"
            )

    def _sample_window_pos(self) -> np.ndarray:
        """按当前档位采样本回合的窗体位置（3 维向量，直接交给底层 reset_model 使用）。"""
        p = self._params
        x = float(p.window_x_center + np.random.uniform(-p.window_x_range, p.window_x_range))
        y = float(p.window_y_center + np.random.uniform(-p.window_y_range, p.window_y_range))
        return np.array([x, y, p.window_z], dtype=np.float64)

    def _read_dists(self, obs: np.ndarray) -> Tuple[float, float]:
        """从 39 维观测里读出本 wrapper 需要的两个距离。

        cur_dist       : 把手与目标在 x 方向的剩余距离（与官方 success 判据同量）
        gripper_to_obj : 夹爪末端到"就位点"的距离。
                         就位点 = 把手 + (-0.04, 0, 0)，即把手 −x 侧 4cm
                         （窗沿 +x 推开，夹爪须站在 −x 侧施力；
                          只算到把手中心的距离无法区分"贴着把手"和"就位"）
        """
        handle_x = float(obs[4:7][0])
        goal_x = float(obs[36:39][0])
        cur_dist = abs(goal_x - handle_x)
        inplace_point = obs[4:7] + INPLACE_OFFSET
        gripper_to_obj = float(np.linalg.norm(obs[0:3] - inplace_point))
        return cur_dist, gripper_to_obj

    def _reset_episode_state(self) -> None:
        """reset() / 新回合开始时重置计数。"""
        self._episode_step = 0
        self._step_rewards = []
        self._prev_dist = None
        self._prev_dist_girpper2obj = None

    # ------------------------------------------------------------------
    # gym.Wrapper 覆写
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._reset_episode_state()

        # 本回合的窗体初始位置：写进底层随机向量，底层 reset_model 会据此摆放窗体
        self._unwrapped._freeze_rand_vec = True
        self._unwrapped._last_rand_vec = self._sample_window_pos()

        obs, info = self.env.reset(**kwargs)

        # 增量式奖励需要初始距离（窗总是从关闭状态开始，故初始剩余距离恒为 0.2）
        self._prev_dist, self._prev_dist_girpper2obj = self._read_dists(obs)

        return {"observation": obs}, info

    def step(self, action):
        # 丢弃底层 reward，用我们自己的
        obs, _native_reward, _native_terminated, native_truncated, info = self.env.step(action)

        cur_dist, gripper_to_obj = self._read_dists(obs)

        step_reward = compute_step_reward(
            prev_dist=self._prev_dist,
            cur_dist=cur_dist,
            prev_dist_girpper2obj=self._prev_dist_girpper2obj,
            gripper_to_obj=gripper_to_obj,
            threshold=self._params.success_threshold,
        )
        self._prev_dist = cur_dist
        self._prev_dist_girpper2obj = gripper_to_obj

        self._step_rewards.append(step_reward)
        self._episode_step += 1

        # 屏蔽底层提前终止（强制跑满 T 步）
        terminated = False
        truncated = bool(native_truncated) or (self._episode_step >= self._params.max_episode_steps)

        info["step_reward"] = step_reward
        info["step_rewards"] = list(self._step_rewards)
        info["terminal_reward"] = compute_terminal_reward(self._step_rewards) if truncated else 0.0
        info["curriculum_level"] = self._curriculum_level
        info["episode_step"] = self._episode_step

        return {"observation": obs}, terminated, truncated, info

# ---------------------------------------------------------------------------
# 5. 环境工厂
# ---------------------------------------------------------------------------

def make_windowopen_env(
    env_id: str,
    curriculum_level: int,
    render_mode: Optional[str] = None,
) -> EpisodicDwellRewardWrapper:

    # 底层上限留余量（250 > 各档 horizon 200），保证截断始终由本 wrapper 决定
    base_env = gym.make("Meta-World/MT1", env_name=env_id, render_mode=render_mode, max_episode_steps=200)
    wrapped = EpisodicDwellRewardWrapper(base_env, curriculum_level)
    return wrapped

def get_env_obs_dim(env_id: str) -> int:
    """查询环境的观测维度（不创建渲染窗口）。"""
    dummy = make_windowopen_env(env_id=env_id, curriculum_level=0)
    try:
        obs, _ = dummy.reset()
        dim = obs["observation"].shape[0]
    finally:
        dummy.close()
    return dim

def get_env_action_dim(env_id: str) -> int:
    """查询环境的动作维度（不创建渲染窗口）。"""
    dummy = make_windowopen_env(env_id=env_id, curriculum_level=0)
    try:
        dummy.reset()
        dim = dummy.action_space.shape[0]
    finally:
        dummy.close()
    return dim
