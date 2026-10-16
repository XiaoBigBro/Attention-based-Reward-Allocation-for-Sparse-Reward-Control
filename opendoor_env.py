"""
META-WORLD door-open-v3 环境改造 + 课程学习接口
- 4 档预设难度（0~3），只变门的初始位置与拉动深度阈值，可动态切换
- 单回合固定 200 步（不因拉到位而提前终止）

底层事实（metaworld 3.1.1 源码 + 实测）：
  * 入口  gym.make("Meta-World/MT1", env_name="door-open-v3", max_episode_steps=T)
  * 观测  39 维扁平：obs[0:3] 末端、obs[3] 夹爪开合、obs[4:7] 把手(handle)位置、
          obs[7:11] 把手四元数、obs[18:36] 前一帧、obs[36:39] 目标位置
  * 官方成功判据：|目标x - 把手x| <= 0.08（只看 x，且是"双侧窗口"：
          推过头会重新判定失败；门铰链限位 -2 rad ≈ -114.6°）
  * 门体随机范围：x∈[0,0.1], y∈[0.85,0.95], z≡0.15；目标点 = 门体位置 + (-0.3,-0.45,0)
  * 初始剩余距离恒为 0.4907（目标点随门平移）
  * 官方专家（夹爪闭合）6 回合首次成功步数：74/77/81/81/98/117
"""
#opendoor

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import metaworld  # noqa: F401 — register Meta-World envs
import numpy as np

# ---------------------------------------------------------------------------
# 1. 课程难度参数
# ---------------------------------------------------------------------------

NUM_CURRICULUM_LEVELS = 4  # 等级 0.1.2.3，难度递增（门的散布范围变大、拉动深度阈值变严）

# 是一个代码生成器，它让你专注于定义“有哪些数据”（属性），
# 而自动帮你生成“如何处理这些数据”（初始化、打印、比较等方法）
@dataclass
class EnvCurriculumParams:
    """控制难度的参数（单位：米）。"""
    door_x_center: float = 0.05       #门体 x 方向中心（官方采样 x∈[0,0.1] 的中心）
    door_x_range: float = 0.01        #门体 x 方向半范围
    door_y_center: float = 0.90       #门体 y 方向中心（官方采样 y∈[0.85,0.95] 的中心）
    door_y_range: float = 0.01        #门体 y 方向半范围
    door_z: float = 0.15              #门体高度（固定，与 XML 中 door 的 z 一致）
    success_threshold: float = 0.15   #把手与目标的剩余 x 距离阈值：<= 该值视为拉到位
    max_episode_steps: int = 200      #最大回合步数

CURRICULUM_PRESETS = {
    0: EnvCurriculumParams(door_x_range=0.010, door_y_range=0.010, success_threshold=0.15, max_episode_steps=200),
    1: EnvCurriculumParams(door_x_range=0.025, door_y_range=0.025, success_threshold=0.12, max_episode_steps=200),
    2: EnvCurriculumParams(door_x_range=0.040, door_y_range=0.040, success_threshold=0.09, max_episode_steps=200),
    3: EnvCurriculumParams(door_x_range=0.050, door_y_range=0.050, success_threshold=0.06, max_episode_steps=200),
}

# ---------------------------------------------------------------------------
# 2. 奖励计算
# ---------------------------------------------------------------------------

#密集奖励函数
def compute_step_reward(
    prev_dist: float,
    cur_dist: float,
    prev_dist_girpper2obj:float,
    gripper_to_obj :float,
    threshold: float,
) -> float:

    # 已拉到位：拉得越到位（剩余距离越小），奖励越高
    if cur_dist < threshold:

        depth_ratio = 1.0 - cur_dist / threshold  # 0~1
        return 2.0 + depth_ratio  # 2.0 ~ 3.0

    #拉动奖励（把手朝目标方向推进为正，被门带回来为负）
    delta_pull = prev_dist - cur_dist
    r_pull = np.tanh( 150.0 * delta_pull if delta_pull >= 0 else 15.0 * delta_pull)

    #远处靠近奖励（趋势引导：只在距离还远时计；这里的距离是"到就位点"的距离）
    if gripper_to_obj > 0.10:
        delta_approach = prev_dist_girpper2obj - gripper_to_obj
        r_approach = 0.3 * np.tanh(50.0*delta_approach)
    else:
        r_approach = 0

    #就位奖励：越靠近"把手与门板之间"的就位点越高（0.06 以内给满；权重远小于 dwell，
    #避免"贴着把手不动"也能刷分）。实测：专家拉动时到就位点 0.046，天真直冲策略最小 0.082
    r_inplace = 0.4 * max(0.0, 1.0 - gripper_to_obj / 0.06)

    #（原"接触奖励"项已删除：实测 176/200 步都触发、退化成第二个常数奖励，作用已被靠近项覆盖）
    return 0.3 + r_pull + r_approach + r_inplace

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
    将 Meta-World door-open-v3 改造为可切换难度的环境。

    行为：
    - step() 返回 (obs, terminated, truncated, info) 四元组（不含 reward）；
      每步即时奖励写入 info["step_rewards"]，回合末写入 info["terminal_reward"]。
    - 观测包成 {"observation": 39 维向量}，使 ppo.unpack_obs 不用改就能用。
    - 观测索引：obs[0:3] 末端位置、obs[3] 夹爪开合、obs[4:7] 把手位置、obs[36:39] 目标位置。
    - 难度只由 CURRICULUM_PRESETS 决定：门的初始位置 + 拉动深度阈值。
    - 回合固定跑满 max_episode_steps 步（拉到位也不提前终止）。
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
        self._step_rewards: list[float] = []  # 存储每步的即时奖励
        self._prev_dist: Optional[float] = None  # 上一步"把手→目标"的剩余 x 距离（增量式奖励用）
        self._prev_dist_girpper2obj: Optional[float] = None  # 上一步"夹爪→把手"的距离
        self._env_prepared: bool = False#底层环境是否已完成一次性准备

        # 同步本档难度到底层 env（门体位置在每回合 reset 时采样）
        self._sync_env_params()#同步底层 env 的准备状态与 horizon 检查

    # ------------------------------------------------------------------
    # 课程接口，切换难度
    # ------------------------------------------------------------------

    def set_curriculum_level(self, level: int, ) -> None:
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
        """
        对应 Fetch 版的 _sync_target_range()：把底层环境准备成"可由本 wrapper 指定门体位置"的状态。
        必须成功，否则立即报错，避免静默失败导致课程难度未生效。
        首次调用做一次性准备；horizon 检查每次都做（各档可能不同）。
        """
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

    def _sample_door_pos(self) -> np.ndarray:
        """按当前档位采样本回合的门体位置（3 维向量，直接交给底层 reset_model 使用）。"""
        p = self._params
        x = float(p.door_x_center + np.random.uniform(-p.door_x_range, p.door_x_range))
        y = float(p.door_y_center + np.random.uniform(-p.door_y_range, p.door_y_range))
        return np.array([x, y, p.door_z], dtype=np.float64)

    def _read_dists(self, obs: np.ndarray) -> Tuple[float, float]:
        """从 39 维观测里读出本 wrapper 需要的两个距离。

        cur_dist       : 把手与目标在 x 方向的剩余距离（与官方 success 判据同量）
        gripper_to_obj : 夹爪末端到"就位点"的距离。
                         就位点 = 把手 + (0, +0.03, -0.01)，即把手朝门板一侧 3cm 处
                         （取自官方原生奖励的 in_place 项；开门必须把手伸进"把手与门板之间"，
                          只算到把手中心的距离无法区分"贴着把手"和"就位"）

        用 obs 自己算而不读 info：底层"仿真异常"分支会把 info 里的量全部写成 0，
        那种时刻读 info 会误判为"已拉到位"。
        """
        handle_x = float(obs[4:7][0])#obs[4:7]   = 把手位置
        goal_x = float(obs[36:39][0])#obs[36:39] = 目标位置
        cur_dist = abs(goal_x - handle_x)
        inplace_point = obs[4:7] + np.array([0.0, 0.03, -0.01])#就位点 = 把手 + (0,+0.03,-0.01)
        gripper_to_obj = float(
            np.linalg.norm(obs[0:3] - inplace_point)#obs[0:3] = 末端位置
        )
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

    def reset(self, **kwargs):#**kwargs是关键字参数，可以传入任意数量的关键字参数
        self._reset_episode_state()

        # 本回合的门体初始位置：写进底层随机向量，底层 reset_model 会据此摆放门
        self._unwrapped._freeze_rand_vec = True
        self._unwrapped._last_rand_vec = self._sample_door_pos()

        obs, info = self.env.reset(**kwargs)

        # 增量式奖励需要初始距离（门总是从关闭状态开始，故初始剩余距离恒为 0.4907）
        self._prev_dist, self._prev_dist_girpper2obj = self._read_dists(obs)

        return {"observation": obs}, info

    def step(self, action):
        # 丢弃底层 reward，用我们自己的

        obs, _native_reward, _native_terminated, native_truncated, info = self.env.step(action)

        # 计算当前"把手→目标"剩余 x 距离 与 "夹爪→就位点"距离
        cur_dist, gripper_to_obj = self._read_dists(obs)

        # 增量式密集奖励：拉到位恒定 / 未到位按拉动+靠近+就位
        step_reward = compute_step_reward(
            prev_dist=self._prev_dist,#上次把手与目标的剩余距离
            cur_dist=cur_dist,#这次把手与目标的剩余距离
            prev_dist_girpper2obj = self._prev_dist_girpper2obj,#上次夹爪到就位点的距离
            gripper_to_obj = gripper_to_obj,#这次夹爪到就位点的距离
            threshold = self._params.success_threshold,#判定拉到位的阈值
        )
        self._prev_dist = cur_dist
        self._prev_dist_girpper2obj = gripper_to_obj

        # 存储即时奖励
        self._step_rewards.append(step_reward)

        self._episode_step += 1

        # 屏蔽底层提前终止（强制跑满 T 步）
        terminated = False

        # 回合截断判断（同时纳入底层截断，避免底层先截断导致上层等不到 truncated）
        truncated = bool(native_truncated) or (self._episode_step >= self._params.max_episode_steps)

        # info 扩展（success / grasp_reward 等底层键保持原样，不覆盖）
        info["step_reward"] = step_reward
        info["step_rewards"] = list(self._step_rewards)  # 所有即时奖励
        info["terminal_reward"] = compute_terminal_reward(self._step_rewards) if truncated else 0.0
        info["curriculum_level"] = self._curriculum_level
        info["episode_step"] = self._episode_step

        return {"observation": obs}, terminated, truncated, info
        #返回观测、终止(为False)、截断(如果回合步数达到最大步数，则截断为True，其余为False)、
        # 信息（包含当前难度等级、当前回合步数、最终奖励、以及底层透传的 success 等）

# ---------------------------------------------------------------------------
# 5. 环境工厂
# ---------------------------------------------------------------------------

def make_opendoor_env(
    env_id: str,
    curriculum_level: int,
    render_mode: Optional[str] = None,
) -> EpisodicDwellRewardWrapper:

    # 底层上限留余量（250 > 各档 horizon 200），保证截断始终由本 wrapper 决定
    base_env = gym.make("Meta-World/MT1", env_name=env_id, render_mode=render_mode, max_episode_steps=250)
    wrapped = EpisodicDwellRewardWrapper(base_env, curriculum_level)
    return wrapped

def get_env_obs_dim(env_id: str) -> int:
    """查询环境的观测维度（不创建渲染窗口）。"""
    dummy = make_opendoor_env(env_id=env_id, curriculum_level=0)
    try:
        obs, _ = dummy.reset()
        dim = obs["observation"].shape[0]
    finally:
        dummy.close()
    return dim

def get_env_action_dim(env_id: str) -> int:
    """查询环境的动作维度（不创建渲染窗口）。"""
    dummy = make_opendoor_env(env_id=env_id, curriculum_level=0)
    try:
        dummy.reset()
        dim = dummy.action_space.shape[0]
    finally:
        dummy.close()
    return dim
