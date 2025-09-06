import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# 使用您第一问中的物理模型
from core import Scene
from drones import Drone
from box_targets import BoxTarget
from cloud import Cloud
from missiles import *


class SmokeOptimizerV2:
    def __init__(self):
        self.scene = Scene()
        self.setup_scene()

    def setup_scene(self):
        """设置场景"""
        self.scene.targets.append(BoxTarget(0, self.scene))

        # 导弹M1
        self.m1 = Missile(0, np.array([20000, 0, 2000]), self.scene)
        self.scene.missile.append(self.m1)

        # 无人机FY1初始位置
        self.fy1_start = np.array([17800.0, 0.0, 1800.0])
        self.fake_pos = np.array([0.0, 0.0, 0.0])

    def calculate_coverage(self, params):
        """
        计算遮蔽时间
        params: [theta, v, t_drop, t_burst]
        theta: 飞行方向角(弧度)
        v: 飞行速度(m/s)
        t_drop: 投放时间(s)
        t_burst: 起爆时间(s)
        """
        theta, v, t_drop, t_burst = params

        # 重置场景
        self.setup_scene()

        # 计算飞行方向向量
        direction = np.array([np.cos(theta), np.sin(theta), 0])

        # 计算投放点位置
        drop_pos = self.fy1_start.copy()
        drop_pos[:2] += v * t_drop * direction[:2]

        # 计算起爆点位置
        v_hor = v * direction[:2]
        delta_xy = v_hor * t_burst
        delta_z = -0.5 * 9.8 * t_burst ** 2
        bang_pos = np.array([
            drop_pos[0] + delta_xy[0],
            drop_pos[1] + delta_xy[1],
            drop_pos[2] + delta_z
        ])

        # 创建烟幕云团
        cloud = Cloud(1, bang_pos, self.scene)
        self.scene.cloud.append(cloud)

        # 模拟整个过程
        total_time = t_drop + t_burst + 25  # 投放+起爆+20秒遮蔽
        dt = 0.01
        t_current = 0.0

        # 第一阶段：投放前
        for _ in range(int(t_drop / dt)):
            self.scene.step(t_current, dt)
            t_current += dt

        # 第二阶段：投放后到起爆前
        for _ in range(int(t_burst / dt)):
            self.scene.step(t_current, dt)
            t_current += dt

        # 第三阶段：起爆后20秒
        for _ in range(int(20 / dt)):
            self.scene.step(t_current, dt)
            t_current += dt

        # 获取遮蔽时间
        coverage_time = self.m1.get_blocked_time()
        return -coverage_time  # 返回负值用于最小化


def optimize_with_de():
    """使用差分进化算法进行优化"""
    optimizer = SmokeOptimizerV2()

    # 定义参数边界
    bounds = [
        (0, 2 * np.pi),  # theta: 0-2π
        (70, 140),  # v: 70-140 m/s
        (0, 10),  # t_drop: 0-10s
        (0, 10)  # t_burst: 0-10s
    ]

    # 运行差分进化算法
    result = differential_evolution(
        optimizer.calculate_coverage,
        bounds,
        strategy='best1bin',
        popsize=20,
        maxiter=100,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42
    )

    return result


def optimize_with_sa():
    """使用模拟退火算法进行优化"""
    from scipy.optimize import dual_annealing

    optimizer = SmokeOptimizerV2()

    bounds = [
        (0, 2 * np.pi),
        (70, 140),
        (0, 10),
        (0, 10)
    ]

    result = dual_annealing(
        optimizer.calculate_coverage,
        bounds,
        maxiter=200,
        initial_temp=5230,
        restart_temp_ratio=2e-5,
        visit=2.62,
        accept=-5.0,
        seed=42
    )

    return result


def main():
    print("开始优化烟幕干扰弹投放策略（问题2）...")

    # 方法1：差分进化算法
    print("\n=== 使用差分进化算法优化 ===")
    de_result = optimize_with_de()

    print(f"最优参数: theta={np.degrees(de_result.x[0]):.2f}°, "
          f"v={de_result.x[1]:.2f}m/s, "
          f"t_drop={de_result.x[2]:.2f}s, "
          f"t_burst={de_result.x[3]:.2f}s")
    print(f"最大遮蔽时间: {-de_result.fun:.2f}s")

    # 方法2：模拟退火算法（验证）
    print("\n=== 使用模拟退火算法验证 ===")
    sa_result = optimize_with_sa()

    print(f"最优参数: theta={np.degrees(sa_result.x[0]):.2f}°, "
          f"v={sa_result.x[1]:.2f}m/s, "
          f"t_drop={sa_result.x[2]:.2f}s, "
          f"t_burst={sa_result.x[3]:.2f}s")
    print(f"最大遮蔽时间: {-sa_result.fun:.2f}s")

    # 选择更好的结果
    if -de_result.fun >= -sa_result.fun:
        best_result = de_result
        method = "差分进化算法"
    else:
        best_result = sa_result
        method = "模拟退火算法"

    print(f"\n=== 最优结果（{method}）===")
    theta_deg = np.degrees(best_result.x[0])
    v = best_result.x[1]
    t_drop = best_result.x[2]
    t_burst = best_result.x[3]
    coverage_time = -best_result.fun

    print(f"飞行方向角: {theta_deg:.2f}°")
    print(f"飞行速度: {v:.2f} m/s")
    print(f"投放时间: {t_drop:.2f} s")
    print(f"起爆时间: {t_burst:.2f} s")
    print(f"最大遮蔽时间: {coverage_time:.2f} s")

    # 计算投放点和起爆点坐标
    direction = np.array([np.cos(best_result.x[0]), np.sin(best_result.x[0]), 0])
    drop_pos = np.array([17800.0, 0.0, 1800.0])
    drop_pos[:2] += v * t_drop * direction[:2]

    v_hor = v * direction[:2]
    delta_xy = v_hor * t_burst
    delta_z = -0.5 * 9.8 * t_burst ** 2
    bang_pos = np.array([
        drop_pos[0] + delta_xy[0],
        drop_pos[1] + delta_xy[1],
        drop_pos[2] + delta_z
    ])

    print(f"\n投放点坐标: ({drop_pos[0]:.2f}, {drop_pos[1]:.2f}, {drop_pos[2]:.2f}) m")
    print(f"起爆点坐标: ({bang_pos[0]:.2f}, {bang_pos[1]:.2f}, {bang_pos[2]:.2f}) m")

    # 保存结果
    result_df = pd.DataFrame({
        '参数': ['飞行方向角(度)', '飞行速度(m/s)', '投放时间(s)', '起爆时间(s)',
                 '遮蔽时间(s)', '投放点X(m)', '投放点Y(m)', '投放点Z(m)',
                 '起爆点X(m)', '起爆点Y(m)', '起爆点Z(m)'],
        '数值': [theta_deg, v, t_drop, t_burst, coverage_time,
                 drop_pos[0], drop_pos[1], drop_pos[2],
                 bang_pos[0], bang_pos[1], bang_pos[2]]
    })

    result_df.to_excel('problem2_optimization_result.xlsx', index=False)
    print("\n结果已保存到 problem2_optimization_result.xlsx")

    # 可视化参数敏感性分析
    visualize_sensitivity(best_result.x)


def visualize_sensitivity(optimal_params):
    """可视化参数敏感性"""
    optimizer = SmokeOptimizerV2()
    base_coverage = -optimizer.calculate_coverage(optimal_params)

    variations = np.linspace(-0.2, 0.2, 9)  # ±20%的变化
    param_names = ['飞行方向角', '飞行速度', '投放时间', '起爆时间']
    param_units = ['°', 'm/s', 's', 's']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (ax, name, unit) in enumerate(zip(axes, param_names, param_units)):
        sensitivities = []
        test_params = optimal_params.copy()

        for var in variations:
            test_params[i] = optimal_params[i] * (1 + var)
            if i == 0:  # 角度需要特殊处理
                test_params[i] = test_params[i] % (2 * np.pi)

            coverage = -optimizer.calculate_coverage(test_params)
            sensitivities.append(coverage)

        if i == 0:
            x_values = np.degrees(optimal_params[i] * (1 + variations))
        else:
            x_values = optimal_params[i] * (1 + variations)

        ax.plot(x_values, sensitivities, 'bo-', linewidth=2)
        ax.axvline(x=optimal_params[i] if i != 0 else np.degrees(optimal_params[i]),
                   color='r', linestyle='--', label='最优值')
        ax.set_xlabel(f'{name} ({unit})')
        ax.set_ylabel('遮蔽时间 (s)')
        ax.set_title(f'{name}敏感性分析')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()