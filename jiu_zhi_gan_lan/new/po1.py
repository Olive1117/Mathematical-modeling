import matplotlib.pyplot as plt
import numpy as np
# 最简化的字体设置，确保兼容性
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 基础样式设置
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 80
plt.rcParams["font.size"] = 12
def plot10():
    """3D散点图（极简版）"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_proj_type('persp')
    # 生成数据

    # 导弹位置
    missiles = np.array([
        [20000, 0, 2000],
        [19000, 600, 2100],
        [18000, -600, 1900]
    ])

    # 无人机位置
    drones = np.array([
        [17800, 0, 1800],
        [12000, 1400, 1400],
        [6000, -3000, 700],
        [11000, 2000, 1800],
        [13000, -2000, 1300]
    ])

    # 假目标位置（原点）
    fake_target = np.array([0, 0, 0])

    # 真目标下底面圆心
    real_target = np.array([0, 200, 0])

    # 绘制导弹（红色三角）
    ax.scatter(missiles[:, 0], missiles[:, 1], missiles[:, 2],
               color='red', marker='^', s=50, label='导弹')

    # 绘制无人机（蓝色圆点）
    ax.scatter(drones[:, 0], drones[:, 1], drones[:, 2],
               color='blue', marker='o', s=50, label='无人机')

    # 绘制假目标（绿色星形）
    ax.scatter(*fake_target, color='red', marker='o', s=100, label='真目标')

    # 绘制真目标（黑色圆点）
    ax.scatter(*real_target, color='black', marker='o', s=100, label='假目标')

    ax.legend()
    ax.set_title('3D散点图')
    plt.tight_layout()
    plt.show()

plot10()