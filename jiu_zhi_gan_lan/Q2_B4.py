import numpy as np
import math


class CorrectedSmokeModel:
    def __init__(self):
        # 导弹M1初始位置和速度
        self.missile_start = np.array([20000, 0, 2000])
        self.fake_target = np.array([0, 200, 0])
        self.missile_speed = 300

        # 计算导弹飞行方向（单位向量）
        missile_dir = self.fake_target - self.missile_start
        self.missile_dir_unit = missile_dir / np.linalg.norm(missile_dir)

        # 无人机FY1初始位置
        self.drone_start = np.array([17800, 0, 1800])

        # 烟幕参数
        self.smoke_sink_speed = 3
        self.effective_radius = 10
        self.max_effective_time = 20

        # 重力加速度
        self.g = 9.8

    def missile_position(self, t):
        """计算导弹在时间t的位置"""
        return self.missile_start + self.missile_dir_unit * self.missile_speed * t

    def drone_position(self, theta, v, t_drop, t):
        """计算无人机在时间t的位置"""
        if t <= t_drop:
            # 投放前：匀速直线运动
            direction = np.array([math.cos(theta), math.sin(theta), 0])
            return self.drone_start + direction * v * t
        else:
            # 投放后：保持最后位置
            direction = np.array([math.cos(theta), math.sin(theta), 0])
            return self.drone_start + direction * v * t_drop

    def smoke_trajectory(self, drop_pos, drop_time, burst_time, t):
        """计算烟幕弹在时间t的位置"""
        if t < drop_time:
            return None

        # 投放后到起爆前：自由落体
        if t < drop_time + burst_time:
            fall_time = t - drop_time
            z_pos = drop_pos[2] - 0.5 * self.g * fall_time ** 2
            return np.array([drop_pos[0], drop_pos[1], z_pos])

        # 起爆后：匀速下沉
        else:
            burst_time_total = drop_time + burst_time
            burst_duration = t - burst_time_total

            # 起爆时刻的位置（自由落体终点）
            burst_z = drop_pos[2] - 0.5 * self.g * burst_time ** 2
            burst_pos = np.array([drop_pos[0], drop_pos[1], burst_z])

            # 起爆后匀速下沉
            return burst_pos - np.array([0, 0, self.smoke_sink_speed * burst_duration])

    def calculate_cover_time(self, theta, v, t_drop, t_burst):
        """计算有效遮蔽时间"""
        drop_pos = self.drone_position(theta, v, t_drop, t_drop)
        burst_time_total = t_drop + t_burst

        cover_time = 0
        cover_start = None

        # 模拟时间序列（从起爆时刻到最大有效时间）
        time_steps = np.arange(burst_time_total, burst_time_total + self.max_effective_time + 0.1, 0.1)

        for t in time_steps:
            missile_pos = self.missile_position(t)
            smoke_pos = self.smoke_trajectory(drop_pos, t_drop, t_burst, t)

            if smoke_pos is not None:
                distance = np.linalg.norm(smoke_pos - missile_pos)
                if distance <= self.effective_radius:
                    if cover_start is None:
                        cover_start = t
                else:
                    if cover_start is not None:
                        cover_time += (t - cover_start)
                        cover_start = None

        # 添加最后一段遮蔽时间（如果仍在遮蔽中）
        if cover_start is not None:
            cover_time += (burst_time_total + self.max_effective_time - cover_start)

        return cover_time


def direct_optimization():
    """直接优化方法 - 基于物理直觉"""
    model = CorrectedSmokeModel()

    # 计算导弹大致飞行时间
    missile_distance = np.linalg.norm(model.fake_target - model.missile_start)
    approx_missile_time = missile_distance / model.missile_speed

    # 最佳参数初始估计
    best_cover_time = 0
    best_params = None

    print("开始直接优化...")
    print(f"导弹大致飞行时间: {approx_missile_time:.2f}s")

    # 尝试不同的策略
    strategies = [
        # 策略1: 尽早投放，尽早起爆
        {'theta': math.atan2(200, -20000), 'v': 120, 't_drop': 5, 't_burst': 5},

        # 策略2: 中等时间投放和起爆
        {'theta': math.atan2(200, -20000), 'v': 120, 't_drop': 15, 't_burst': 5},

        # 策略3: 较晚投放，较早起爆
        {'theta': math.atan2(200, -20000), 'v': 120, 't_drop': 20, 't_burst': 3},

        # 策略4: 针对导弹路径优化
        {'theta': math.atan2(0, -1), 'v': 120, 't_drop': 10, 't_burst': 5},

        # 策略5: 高速飞行，较早投放
        {'theta': math.atan2(0, -1), 'v': 140, 't_drop': 5, 't_burst': 5},
    ]

    for i, strategy in enumerate(strategies):
        theta = strategy['theta']
        v = strategy['v']
        t_drop = strategy['t_drop']
        t_burst = strategy['t_burst']

        cover_time = model.calculate_cover_time(theta, v, t_drop, t_burst)

        print(f"策略 {i + 1}: theta={math.degrees(theta):.1f}°, v={v}, "
              f"t_drop={t_drop}, t_burst={t_burst}, 遮蔽时间={cover_time:.3f}s")

        if cover_time > best_cover_time:
            best_cover_time = cover_time
            best_params = (theta, v, t_drop, t_burst)

    # 在最佳策略附近进行微调
    if best_params is not None:
        theta, v, t_drop, t_burst = best_params
        print(f"\n在最佳策略附近微调...")

        # 微调参数
        for delta_t_drop in [-2, -1, 0, 1, 2]:
            for delta_t_burst in [-1, 0, 1]:
                new_t_drop = max(0, t_drop + delta_t_drop)
                new_t_burst = max(0, t_burst + delta_t_burst)

                cover_time = model.calculate_cover_time(theta, v, new_t_drop, new_t_burst)

                if cover_time > best_cover_time:
                    best_cover_time = cover_time
                    best_params = (theta, v, new_t_drop, new_t_burst)
                    print(f"改进: t_drop={new_t_drop}, t_burst={new_t_burst}, 遮蔽时间={cover_time:.3f}s")

    return best_params, best_cover_time


def analyze_solution(model, params):
    """分析解决方案"""
    if params is None:
        print("未找到有效解决方案")
        return

    theta, v, t_drop, t_burst = params

    print("\n" + "=" * 60)
    print("解决方案分析:")
    print("=" * 60)

    # 计算关键位置
    drop_pos = model.drone_position(theta, v, t_drop, t_drop)
    burst_pos = model.smoke_trajectory(drop_pos, t_drop, t_burst, t_drop + t_burst)

    # 计算关键时间点的导弹位置
    missile_at_drop = model.missile_position(t_drop)
    missile_at_burst = model.missile_position(t_drop + t_burst)

    print(f"投放时刻 (t={t_drop}s):")
    print(f"  无人机位置: ({drop_pos[0]:.1f}, {drop_pos[1]:.1f}, {drop_pos[2]:.1f})")
    print(f"  导弹位置: ({missile_at_drop[0]:.1f}, {missile_at_drop[1]:.1f}, {missile_at_drop[2]:.1f})")
    print(f"  水平距离: {np.linalg.norm(drop_pos[:2] - missile_at_drop[:2]):.1f}m")
    print(f"  垂直距离: {abs(drop_pos[2] - missile_at_drop[2]):.1f}m")

    print(f"\n起爆时刻 (t={t_drop + t_burst}s):")
    print(f"  烟幕位置: ({burst_pos[0]:.1f}, {burst_pos[1]:.1f}, {burst_pos[2]:.1f})")
    print(f"  导弹位置: ({missile_at_burst[0]:.1f}, {missile_at_burst[1]:.1f}, {missile_at_burst[2]:.1f})")
    print(f"  距离: {np.linalg.norm(burst_pos - missile_at_burst):.1f}m")

    # 计算遮蔽时间段
    cover_time = model.calculate_cover_time(theta, v, t_drop, t_burst)
    print(f"\n有效遮蔽时间: {cover_time:.3f}s")


# 主程序
if __name__ == "__main__":
    print("烟幕干扰弹投放策略优化 - 问题2")
    print("使用直接优化方法")

    # 创建模型
    model = CorrectedSmokeModel()

    # 运行优化
    best_params, best_cover_time = direct_optimization()

    # 分析解决方案
    analyze_solution(model, best_params)

    # 输出最终结果
    if best_params is not None:
        theta, v, t_drop, t_burst = best_params
        print("\n" + "=" * 60)
        print("最终推荐策略:")
        print("=" * 60)
        print(f"飞行方向: {math.degrees(theta):.1f}° (从正东方向逆时针)")
        print(f"飞行速度: {v:.1f} m/s")
        print(f"投放时间: {t_drop:.1f} s (受领任务后)")
        print(f"起爆时间: {t_burst:.1f} s (投放后)")
        print(f"预计有效遮蔽时间: {best_cover_time:.3f} s")
        print("=" * 60)
    else:
        print("未找到有效的投放策略")