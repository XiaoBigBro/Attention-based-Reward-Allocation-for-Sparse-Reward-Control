"""
绘制训练曲线 - 终端奖励 + 课程等级
"""
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = np.loadtxt('checkpoints/training_rewards.txt', delimiter=',', skiprows=1)
episodes = data[:, 0]
terminal_rewards = data[:, 1]
curriculum_levels = data[:, 2]

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 子图1：终端奖励曲线
window = 50
smoothed = [np.mean(terminal_rewards[max(0, i-window):i+1]) for i in range(len(terminal_rewards))]
axes[0].plot(terminal_rewards, alpha=0.3, color='blue', label='Raw')
axes[0].plot(smoothed, linewidth=2, color='red', label=f'{window}-Episode Avg')
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Terminal Reward', fontsize=12)
axes[0].set_title('Terminal Reward over Training', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 子图2：课程等级变化
axes[1].plot(episodes, curriculum_levels, linewidth=1, color='green', step='post')
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('Curriculum Level', fontsize=12)
axes[1].set_title('Curriculum Level over Training', fontsize=14)
axes[1].set_yticks([0, 1, 2])
axes[1].set_yticklabels(['Level 0', 'Level 1', 'Level 2'])
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.5, 2.5)

plt.tight_layout()
plt.savefig('checkpoints/training_curve.png', dpi=150)
print('Curve saved to: checkpoints/training_curve.png')
plt.close()
