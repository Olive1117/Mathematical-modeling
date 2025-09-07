import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def plot_problem4_3d():
    """
    绘制问题4的三维示意图：FY1、FY2、FY3对M1的干扰
    """
    # 创建3D图形
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 导弹M1初始位置
    m1_pos = np.array([20000, 0, 2000])

    # 无人机初始位置
    fy1_pos = np.array([17800, 0, 1800])
    fy2_pos = np.array([12000, 1400, 1400])
    fy3_pos = np.array([6000, -3000, 700])

    # 目标位置（假设为原点）
    target_pos = np.array([0, 0, 0])

    # 绘制导弹M1轨迹
    def plot_missile_trajectory():
        trajectory_x = np.linspace(m1_pos[0], target_pos[0], 100)
        trajectory_y = np.linspace(m1_pos[1], target_pos[1], 100)
        trajectory_z = np.linspace(m1_pos[2], target_pos[2], 100)

        ax.plot(trajectory_x, trajectory_y, trajectory_z,
                color='red', linestyle='-', linewidth=3, alpha=0.8, label='导弹M1轨迹')

        # 导弹起始点
        ax.scatter(m1_pos[0], m1_pos[1], m1_pos[2],
                   color='red', s=300, marker='^', alpha=1.0, label='导弹M1')

    # 绘制无人机位置和航线
    def plot_drones():
        drones = [
            (fy1_pos, 'orange', 'FY1'),
            (fy2_pos, 'purple', 'FY2'),
            (fy3_pos, 'green', 'FY3')
        ]

        for pos, color, label in drones:
            ax.scatter(pos[0], pos[1], pos[2],
                       color=color, s=200, marker='o', alpha=1.0, label=label)



    # 绘制烟雾弹投放示例（基于优化结果）
    def plot_smoke_deployment():
        # 这里使用示例参数，您可以根据实际优化结果调整
        smoke_params = [
            # (无人机位置, 投放时间, 引爆时间, 角度, 速度, 颜色, 标签)
            (fy1_pos, 5.0, 3.0, np.deg2rad(0), 120, 'orange', 'FY1烟雾弹'),
            (fy2_pos, 8.0, 2.5, np.deg2rad(-30), 130, 'purple', 'FY2烟雾弹'),
            (fy3_pos, 12.0, 3.2, np.deg2rad(45), 110, 'green', 'FY3烟雾弹')
        ]

        for drone_pos, drop_t, detonate_t, angle, speed, color, label in smoke_params:
            # 计算投放点
            d_release = speed * drop_t
            delta_xy = np.array([np.cos(angle), np.sin(angle)]) * d_release
            drop_point = drone_pos.copy()
            drop_point[0] += delta_xy[0]
            drop_point[1] += delta_xy[1]

            # 计算爆点
            g = 9.8
            total_time = drop_t + detonate_t
            h = 0.5 * g * detonate_t ** 2
            blast_point = drop_point.copy()
            blast_point[2] -= h
            blast_point[0] += np.cos(angle) * speed * detonate_t
            blast_point[1] += np.sin(angle) * speed * detonate_t



            # 标记点
            ax.scatter(drop_point[0], drop_point[1], drop_point[2],
                       color=color, s=150, marker='D', alpha=1.0, label=f'{label}投放点')
            ax.scatter(blast_point[0], blast_point[1], blast_point[2],
                       color='red', s=120, marker='*', alpha=1.0, label=f'{label}爆点')

            # 绘制烟雾云团（简化表示）
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = blast_point[0] + 100 * np.outer(np.cos(u), np.sin(v))
            y = blast_point[1] + 100 * np.outer(np.sin(u), np.sin(v))
            z = blast_point[2] + 100 * np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_wireframe(x, y, z, color=color, alpha=0.3)

    # 绘制目标区域
    def plot_target_area():
        # 绘制目标点
        ax.scatter(target_pos[0], target_pos[1], target_pos[2],
                   color='black', s=400, marker='X', alpha=1.0, label='目标点')



    # 执行绘图
    plot_missile_trajectory()
    plot_drones()
    plot_smoke_deployment()
    plot_target_area()

    # 设置坐标轴标签
    ax.set_xlabel('X坐标 (m)', fontsize=14, labelpad=15)
    ax.set_ylabel('Y坐标 (m)', fontsize=14, labelpad=15)
    ax.set_zlabel('Z坐标 (m)', fontsize=14, labelpad=15)
    ax.set_title('问题4：FY1、FY2、FY3对M1的烟幕干扰三维示意图', fontsize=16, fontweight='bold', pad=20)

    # 设置坐标轴范围
    ax.set_xlim(0, 21000)
    ax.set_ylim(-3500, 2500)
    ax.set_zlim(0, 2500)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 添加图例（避免重复标签）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    # 设置视角
    ax.view_init(elev=25, azim=45)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('problem4_3d_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('problem4_3d_diagram.pdf', bbox_inches='tight')

    plt.show()

    return fig, ax








    plt.show()


# 运行绘图
if __name__ == "__main__":
    print("开始绘制问题4的三维示意图...")
    fig_3d, ax_3d = plot_problem4_3d()


    print("绘图完成！图像已保存为：")
    print("- problem4_3d_diagram.png")
    print("- problem4_3d_diagram.pdf")
    print("- problem4_2d_views.png")