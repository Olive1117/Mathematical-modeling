import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


def plot_3d_smoke_deployment():
    """
    根据q5_b2结果绘制三维示意图
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建3D图形
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')

    # 导弹初始位置
    missile_positions = {
        'm1': np.array([20000, 0, 2000]),
        'm2': np.array([19000, 600, 2100]),
        'm3': np.array([18000, -600, 1900])
    }

    # 无人机初始位置
    drone_positions = {
        'fy1': np.array([17800, 0, 1800]),
        'fy2': np.array([12000, 1400, 1400]),
        'fy3': np.array([6000, -3000, 700]),
        'fy4': np.array([11000, 2000, 1800]),
        'fy5': np.array([13000, -2000, 1300])
    }

    # 烟雾弹投放数据（从您的结果中提取）
    smoke_data = {
        'fy1': [
            {'time': 5.0, 'drop_point': [17512.4, 0.0, 1795.3], 'blast_point': [17512.4, 0.0, 1751.2], 'missile': 'm1'}
        ],
        'fy2': [
            {'time': 15.0, 'drop_point': [13536.0, 427.5, 1540.2], 'blast_point': [13536.0, 427.5, 1496.1],
             'missile': 'm2'},
            {'time': 35.0, 'drop_point': [7575.2, 239.2, 881.4], 'blast_point': [7575.2, 239.2, 837.3], 'missile': 'm2'}
        ],
        'fy3': [
            {'time': 45.0, 'drop_point': [5572.0, 0.0, 601.3], 'blast_point': [5572.0, 0.0, 557.2], 'missile': 'm1'}
        ],
        'fy4': [
            {'time': 20.0, 'drop_point': [11042.5, -831.9, 1209.7], 'blast_point': [11042.5, -831.9, 1165.6],
             'missile': 'm3'},
            {'time': 25.0, 'drop_point': [9551.6, -881.6, 1052.3], 'blast_point': [9551.6, -881.6, 1008.2],
             'missile': 'm3'},
            {'time': 30.0, 'drop_point': [8060.7, -931.3, 895.0], 'blast_point': [8060.7, -931.3, 850.9],
             'missile': 'm3'}
        ],
        'fy5': [
            {'time': 25.0, 'drop_point': [10555.6, 333.3, 1210.8], 'blast_point': [10555.6, 333.3, 1166.7],
             'missile': 'm2'}
        ]
    }

    # 颜色定义
    missile_colors = {'m1': 'red', 'm2': 'blue', 'm3': 'green'}
    drone_colors = {'fy1': 'orange', 'fy2': 'purple', 'fy3': 'brown', 'fy4': 'pink', 'fy5': 'gray'}

    # 1. 绘制导弹初始位置和轨迹
    print("绘制导弹轨迹...")
    for missile_name, pos in missile_positions.items():
        color = missile_colors[missile_name]

        # 导弹初始位置
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=200, marker='^', label=f'导弹{missile_name}', alpha=0.8)

        # 导弹大致轨迹（假设向原点飞行）
        trajectory_x = np.linspace(pos[0], 0, 50)
        trajectory_y = np.linspace(pos[1], 0, 50)
        trajectory_z = np.linspace(pos[2], 0, 50)
        ax.plot(trajectory_x, trajectory_y, trajectory_z, color=color, linestyle='--', alpha=0.5)

    # 2. 绘制无人机初始位置
    print("绘制无人机位置...")
    for drone_name, pos in drone_positions.items():
        color = drone_colors[drone_name]
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=150, marker='o', label=f'无人机{drone_name}', alpha=0.8)

    # 3. 绘制烟雾弹投放点和爆点
    print("绘制烟雾弹投放...")
    for drone_name, smokes in smoke_data.items():
        drone_color = drone_colors[drone_name]

        for i, smoke in enumerate(smokes):
            missile_color = missile_colors[smoke['missile']]

            drop_point = smoke['drop_point']
            blast_point = smoke['blast_point']

            # 投放点（较大）
            ax.scatter(drop_point[0], drop_point[1], drop_point[2],
                       c=drone_color, s=100, marker='D', alpha=0.9)

            # 爆点（较小）
            ax.scatter(blast_point[0], blast_point[1], blast_point[2],
                       c=missile_color, s=80, marker='*', alpha=0.9)

            # 投放线（从无人机到投放点）
            drone_pos = drone_positions[drone_name]
            ax.plot([drone_pos[0], drop_point[0]],
                    [drone_pos[1], drop_point[1]],
                    [drone_pos[2], drop_point[2]],
                    color=drone_color, linestyle=':', alpha=0.6)

            # 下降线（从投放到爆点）
            ax.plot([drop_point[0], blast_point[0]],
                    [drop_point[1], blast_point[1]],
                    [drop_point[2], blast_point[2]],
                    color='black', linestyle='-', alpha=0.7)

            # 添加时间标签
            ax.text(drop_point[0], drop_point[1], drop_point[2] + 50,
                    f't={smoke["time"]}s', fontsize=8, color=drone_color)

    # 4. 绘制烟雾区域（简化表示）
    print("绘制烟雾区域...")
    for drone_name, smokes in smoke_data.items():
        for smoke in smokes:
            blast_point = smoke['blast_point']

            # 在爆点周围绘制一个简单的球体表示烟雾
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = blast_point[0] + 200 * np.outer(np.cos(u), np.sin(v))
            y = blast_point[1] + 200 * np.outer(np.sin(u), np.sin(v))
            z = blast_point[2] + 200 * np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

    # 设置坐标轴标签
    ax.set_xlabel('X坐标 (m)')
    ax.set_ylabel('Y坐标 (m)')
    ax.set_zlabel('Z坐标 (m)')
    ax.set_title('无人机烟雾弹投放三维示意图', fontsize=16, fontweight='bold')

    # 设置视角
    ax.view_init(elev=30, azim=40)

    # 创建图例
    legend_elements = []
    for missile, color in missile_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='w',
                                          markerfacecolor=color, markersize=10,
                                          label=f'导弹{missile}'))

    for drone, color in drone_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=10,
                                          label=f'无人机{drone}'))

    legend_elements.extend([
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black', markersize=10, label='投放点'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10, label='爆点'),
        plt.Line2D([0], [0], color='black', linestyle=':', label='无人机航线'),
        plt.Line2D([0], [0], color='black', linestyle='-', label='烟雾弹下降轨迹')
    ])

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

    # 设置坐标轴范围
    ax.set_xlim(0, 21000)
    ax.set_ylim(-3500, 2500)
    ax.set_zlim(0, 2500)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('smoke_deployment_3d.png', dpi=300, bbox_inches='tight')
    plt.savefig('smoke_deployment_3d.pdf', bbox_inches='tight')

    print("显示图像...")
    plt.show()

    return fig, ax


def create_2d_projection_plots():
    """
    创建2D投影图（XY平面和XZ平面）
    """
    # XY平面投影
    fig_xy, ax_xy = plt.subplots(figsize=(12, 8))

    # 导弹位置
    missile_positions = {
        'm1': [20000, 0], 'm2': [19000, 600], 'm3': [18000, -600]
    }

    # 无人机位置
    drone_positions = {
        'fy1': [17800, 0], 'fy2': [12000, 1400], 'fy3': [6000, -3000],
        'fy4': [11000, 2000], 'fy5': [13000, -2000]
    }

    # 绘制导弹和无人机
    for missile, pos in missile_positions.items():
        ax_xy.scatter(pos[0], pos[1], s=100, marker='^', label=f'导弹{missile}')

    for drone, pos in drone_positions.items():
        ax_xy.scatter(pos[0], pos[1], s=80, marker='o', label=f'无人机{drone}')

    ax_xy.set_xlabel('X坐标 (m)')
    ax_xy.set_ylabel('Y坐标 (m)')
    ax_xy.set_title('XY平面投影 - 导弹和无人机位置')
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect('equal')

    plt.savefig('xy_projection.png', dpi=300, bbox_inches='tight')
    plt.show()


# 运行绘图函数
if __name__ == "__main__":
    print("开始绘制三维示意图...")

    # 绘制主3D图
    fig_3d, ax_3d = plot_3d_smoke_deployment()

    print("\n开始绘制2D投影图...")
    # 绘制2D投影图
    create_2d_projection_plots()

    print("\n所有图像绘制完成！")
    print("已保存文件:")
    print("- smoke_deployment_3d.png")
    print("- smoke_deployment_3d.pdf")
    print("- xy_projection.png")