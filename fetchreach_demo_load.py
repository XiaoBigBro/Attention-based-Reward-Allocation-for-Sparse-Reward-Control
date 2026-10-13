# demo_load.py
import torch
import numpy as np
from fetchreach_env import make_fetch_env, get_env_obs_dim, get_env_action_dim
from ppo import PPONetwork
import time  # 添加这行导入

import __main__

# ========== 关键：把训练脚本里的类导入到 __main__ ==========
from train_ppo_allocator import (
    TrainingConfig,
    RewardAllocatorConfig,
    PPOTrainerWithAllocator,
    RawTrajectoryBuffer,
    TrainingBuffer,
)

# 把这些类“注册”到 __main__ 模块，让 pickle 能找到
for cls in [TrainingConfig, RewardAllocatorConfig,
            PPOTrainerWithAllocator, RawTrajectoryBuffer, TrainingBuffer]:
    setattr(__main__, cls.__name__, cls)


def unpack_obs(obs: dict) -> np.ndarray:
    """gymnasium dict obs → 1D numpy"""
    components = [
        obs.get("observation", []),
        obs.get("achieved_goal", []),
        obs.get("desired_goal", []),
    ]
    return np.concatenate([c.flatten() for c in components if len(c) > 0])

def load_and_demo():
    """加载训练好的模型并进行演示"""
    
    print("=== 模型演示加载 ===")
    
    # 1. 指定保存的检查点路径
    save_path = "checkpoints/seed57/checkpoint_4600.pt"  # 使用您项目中已保存的检查点

    # 2. 加载保存的检查点
    print(f"正在加载模型: {save_path}")
    checkpoint = torch.load(save_path, map_location='cpu')
    print("✓ 模型文件加载成功")

    # 3. 获取配置信息
    config = checkpoint["config"]
    print("✓ 配置信息加载成功")

    # 4. 创建环境（支持渲染）
    print("正在创建环境...")
    env = make_fetch_env("FetchPush-v4", curriculum_level=0, render_mode="human")
    print("✓ 环境创建成功")

    # 5. 获取环境维度
    obs_dim = get_env_obs_dim("FetchPush-v4")
    action_dim = get_env_action_dim("FetchPush-v4")
    print(f"环境维度 - 观测: {obs_dim}, 动作: {action_dim}")

    # 6. 创建网络结构（必须和训练时完全一致）
    print("正在创建网络结构...")
    network = PPONetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max,
        log_std_init=config.log_std_init,
    )
    print("✓ 网络结构创建成功")

    # 7. 加载PPO网络权重
    print("正在加载PPO网络权重...")
    network.load_state_dict(checkpoint["ppo_network"])
    network.eval()  # 设置为评估模式
    print("✓ PPO网络权重加载成功")


    # 8. 准备演示
    print("开始演示...")
    for i in range (20):

        obs, info = env.reset()
        obs = unpack_obs(obs)
        done = False

        step_count = 0
        max_steps = 1000  # 限制最大步数避免无限循环

        print("演示开始！按 Ctrl+C 可以中断演示")
        print("-" * 50)
        time.sleep(1)  # 添加延迟0.1秒

        while not done and step_count < max_steps:
            # 9. 使用加载的策略网络获取动作
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, log_prob,log_std, mean = network.get_action(obs_tensor)
                action = action.squeeze(0).numpy()
                # action_deterministic = torch.tanh(mean).squeeze(0).cpu().numpy()

            # 10. 执行动作
            next_obs, terminated, truncated, info = env.step(action)
            next_obs = unpack_obs(next_obs)

            # 检查结束
            done = truncated
            obs = next_obs

            step_count += 1

        terminal_reward = info.get("terminal_reward", 0.0)


        # 12. 演示结束
        print("-" * 50)
        print(f"演示结束！")
        print(f"总奖励: {terminal_reward:.2f}")

    # 13. 关闭环境
    env.close()
    print("✓ 环境已关闭")



if __name__ == "__main__":
    load_and_demo()