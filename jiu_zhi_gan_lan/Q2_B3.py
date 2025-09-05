#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的黑洞式退火算法 - 烟幕干扰弹投放策略优化
加入完整的飞机速度验证和优化
温度衰减速度加快
"""
import numpy as np
import pandas as pd
import os
import random
import time
from tqdm import tqdm

# ---------------- 物理常数 -----------------
G = 9.8
CLOUD_R = 10
CLOUD_SINK = 3.0
MISSILE_SPEED = 300
PLANE_ALT = 1800
FY1_0 = np.array([17800, 0, PLANE_ALT])
TARGET = np.array([0, 200, 0])
MISSILE_INIT = np.array([20000, 0, 2000])
DT = 0.1
MIN_DRONE_SPEED = 70
MAX_DRONE_SPEED = 140

# 计算导弹总飞行时间
missile_dir = TARGET - MISSILE_INIT
missile_dir_unit = missile_dir / np.linalg.norm(missile_dir)
T_MAX = np.linalg.norm(MISSILE_INIT - TARGET) / MISSILE_SPEED  # ≈66.67s


# ---------------- 工具函数 -----------------
def missile_pos(t):
    """导弹在时间t的位置"""
    return MISSILE_INIT + missile_dir_unit * MISSILE_SPEED * t


def dist_point_to_line(p, a, b):
    """计算点到直线的距离"""
    ap = p - a
    ab = b - a
    ab_norm = np.linalg.norm(ab)
    if ab_norm < 1e-12:
        return np.linalg.norm(ap)

    projection = np.dot(ap, ab) / ab_norm
    projection = max(0, min(projection, ab_norm))

    closest_point = a + (ab / ab_norm) * projection
    return np.linalg.norm(p - closest_point)


def shielding_duration(burst_point, t_burst):
    """计算遮蔽持续时间"""
    total = 0.0
    t_start = max(t_burst, 0)

    for t in np.arange(t_start, T_MAX + DT, DT):
        mt = missile_pos(t)
        cloud_t = t - t_burst
        sink_distance = CLOUD_SINK * cloud_t
        cloud_pos = burst_point - np.array([0, 0, sink_distance])

        line_dist = dist_point_to_line(cloud_pos, mt, TARGET)
        if line_dist <= CLOUD_R:
            total += DT

    return total


def calculate_drone_strategy(burst_point, t_burst):
    """
    计算无人机策略，返回(速度, 航向角, 投放点, 是否有效)
    """
    dx = burst_point[0] - FY1_0[0]
    dy = burst_point[1] - FY1_0[1]
    dis_h = np.sqrt(dx ** 2 + dy ** 2)

    # 检查自由落体时间
    if PLANE_ALT <= burst_point[2]:
        return None, None, None, False

    t_free_fall = np.sqrt(2 * (PLANE_ALT - burst_point[2]) / G)
    if t_free_fall < 0:
        return None, None, None, False

    # 计算最小和最大飞行时间（基于速度约束）
    T_min = dis_h / MAX_DRONE_SPEED  # 最大速度对应最小时间
    T_max = dis_h / MIN_DRONE_SPEED  # 最小速度对应最大时间

    # 检查时间约束：飞行时间必须 >= 自由落体时间
    if T_max < t_free_fall:
        return None, None, None, False

    # 选择最优飞行时间（尽可能接近自由落体时间）
    T_opt = max(T_min, t_free_fall)
    T_opt = min(T_opt, T_max)

    # 计算对应的无人机速度
    v_opt = dis_h / T_opt

    # 验证速度约束
    if not (MIN_DRONE_SPEED <= v_opt <= MAX_DRONE_SPEED):
        return None, None, None, False

    # 计算航向角
    ux, uy = dx / dis_h, dy / dis_h
    theta = np.degrees(np.arctan2(uy, ux)) % 360

    # 计算投放点（考虑无人机飞行）
    flight_distance = v_opt * (T_opt - t_free_fall)
    drop_point = FY1_0 + np.array([ux, uy, 0]) * flight_distance

    return v_opt, theta, drop_point, True


def energy_function(burst_point, t_burst):
    """能量函数：返回遮蔽时间，考虑速度约束"""
    v_opt, theta, drop_point, is_valid = calculate_drone_strategy(burst_point, t_burst)

    if not is_valid:
        return -1e6  # 无效解惩罚

    # 有效解，计算遮蔽时间
    return shielding_duration(burst_point, t_burst)


def optimize_burst_time_for_point(burst_point):
    """对给定起爆点优化起爆时间，考虑速度约束"""
    best_t = 3.6
    best_energy = energy_function(burst_point, best_t)
    best_v = None

    # 在合理范围内搜索最佳起爆时间
    for t_burst in np.linspace(1.0, 20.0, 30):
        energy_val = energy_function(burst_point, t_burst)
        v_opt, _, _, is_valid = calculate_drone_strategy(burst_point, t_burst)

        if is_valid and energy_val > best_energy:
            best_t = t_burst
            best_energy = energy_val
            best_v = v_opt

    return best_t, best_energy, best_v


# ---------------- 改进的半径计算函数 ----------------
def adaptive_black_hole_radius(current_energy, best_energy, iteration, max_iterations,
                               base_radius, phase, energy_history):
    """自适应黑洞半径计算"""
    min_radius = 1.0
    max_radius = 200.0

    # 先快后慢的衰减策略
    if iteration < max_iterations * 0.2:
        progress = iteration / (max_iterations * 0.2)
        phase_factor = 1.0 - 0.8 * progress
    elif iteration < max_iterations * 0.6:
        progress = (iteration - max_iterations * 0.2) / (max_iterations * 0.4)
        phase_factor = 0.2 - 0.15 * progress
    else:
        progress = (iteration - max_iterations * 0.6) / (max_iterations * 0.4)
        phase_factor = 0.05 - 0.04 * progress

    # 根据解质量动态调整
    if current_energy > 0:
        if current_energy > best_energy * 0.9:
            quality_factor = 0.3
        elif current_energy > best_energy * 0.7:
            quality_factor = 0.6
        else:
            quality_factor = 1.0
    else:
        quality_factor = 1.5

    # 能量变化趋势调整
    if len(energy_history) >= 5:
        recent_improve = np.mean(energy_history[-5:]) - np.mean(energy_history[-10:-5])
        if recent_improve > 0.1:
            trend_factor = 1.2
        elif recent_improve < -0.1:
            trend_factor = 0.8
        else:
            trend_factor = 1.0
    else:
        trend_factor = 1.0

    radius = base_radius * phase_factor * quality_factor * trend_factor
    return np.clip(radius, min_radius, max_radius)


# ---------------- 速度约束验证函数 ----------------
def validate_drone_speed(burst_point, t_burst):
    """详细验证无人机速度约束"""
    v_opt, theta, drop_point, is_valid = calculate_drone_strategy(burst_point, t_burst)

    if not is_valid:
        print("❌ 速度约束验证失败")
        return False, None, None, None

    # 详细输出速度信息
    print(f"✅ 速度约束验证通过:")
    print(f"   无人机速度: {v_opt:.4f} m/s")
    print(f"   速度范围: [{MIN_DRONE_SPEED}, {MAX_DRONE_SPEED}] m/s")
    print(f"   航向角: {theta:.4f}°")
    print(f"   投放点: [{drop_point[0]:.4f}, {drop_point[1]:.4f}, {drop_point[2]:.4f}]")

    return True, v_opt, theta, drop_point


# ---------------- 改进的黑洞退火算法 ----------------
def find_optimized_initial_center():
    """基于问题1结果寻找优化的初始中心"""
    print("寻找优化的初始中心...")

    base_center = np.array([17188., 0., 1725.])
    base_t_burst = 3.6
    base_energy = energy_function(base_center, base_t_burst)

    best_center = base_center.copy()
    best_energy = base_energy
    best_t_burst = base_t_burst
    best_v = None

    search_radius = 50
    n_samples = 200

    for i in tqdm(range(n_samples)):
        offset = np.random.uniform(-search_radius, search_radius, 3)
        candidate = base_center + offset

        t_opt, energy_val, v_opt = optimize_burst_time_for_point(candidate)

        if energy_val > best_energy:
            best_center = candidate
            best_energy = energy_val
            best_t_burst = t_opt
            best_v = v_opt

    # 验证初始解的速度约束
    is_valid, validated_v, theta, drop_point = validate_drone_speed(best_center, best_t_burst)

    if is_valid:
        print(f"初始中心优化完成: 能量={best_energy:.4f}s, 速度={validated_v:.4f}m/s")
    else:
        print("⚠️ 初始解速度约束验证失败，继续优化...")

    return best_center, best_t_burst, best_energy


def enhanced_black_hole_anneal():
    """改进的黑洞退火算法 - 温度衰减速度加快"""
    center, center_t_burst, initial_energy = find_optimized_initial_center()

    # ==================== 参数设置 ====================
    base_radius = 100.0
    T0 = 50
    alpha = 0.92
    inner_iterations = 50
    max_phases = 100 # 总阶段数减少，加快收敛

    print(f"\n退火参数: base_radius={base_radius}, T0={T0}, alpha={alpha} (温度衰减加快)")
    print(f"速度约束: [{MIN_DRONE_SPEED}, {MAX_DRONE_SPEED}] m/s")

    boundary_range = np.array([800, 500, 300])
    low_b = center - boundary_range
    high_b = center + boundary_range

    S = center.copy()
    S_t_burst = center_t_burst
    best_S, best_t_burst, best_energy = S.copy(), S_t_burst, initial_energy
    best_v = None

    T = T0
    current_energy = initial_energy
    total_iterations = max_phases * inner_iterations
    iteration_count = 0
    energy_history = [current_energy]

    print("开始改进的黑洞退火优化...")
    print("温度衰减速度加快，收敛更快")

    # ==================== 随机测试数据 - 用于验证算法 ====================
    # 注释：这里可以取消注释来测试随机数据点
    test_point = np.array([-7700,2000,1500])  # 随机测试点
    #test_t = 4.2  # 随机测试时间
    # test_energy = energy_function(test_point, test_t)
    # print(f"随机测试点能量: {test_energy:.4f}s")

    for phase in range(max_phases):
        for inner_iter in range(inner_iterations):
            iteration_count += 1

            current_radius = adaptive_black_hole_radius(
                current_energy, best_energy, iteration_count, total_iterations,
                base_radius, phase, energy_history
            )

            if iteration_count % 30 == 0:
                v_info = f"速度={best_v:.4f}m/s" if best_v else "速度=未验证"
                print(f"iter {iteration_count:5d} | T={T:.4f} | "
                      f"当前={current_energy:.4f}s | 最优={best_energy:.4f}s | "
                      f"半径={current_radius:.4f}m | {v_info}")

            # 生成新解
            offset = np.random.normal(0, 1, 3)
            offset_norm = np.linalg.norm(offset)
            if offset_norm > 1e-8:
                offset *= current_radius / offset_norm
            else:
                offset = np.random.uniform(-current_radius, current_radius, 3)

            S_new = S + offset
            S_new = np.clip(S_new, low_b, high_b)

            # 优化新解的起爆时间
            t_burst_new, energy_new, v_new = optimize_burst_time_for_point(S_new)

            delta_E = energy_new - current_energy

            if delta_E > 0 or random.random() < np.exp(delta_E / max(T, 1e-8)):
                S, S_t_burst, current_energy = S_new, t_burst_new, energy_new
                energy_history.append(current_energy)

                if current_energy > best_energy:
                    best_S, best_t_burst, best_energy = S.copy(), t_burst_new, current_energy
                    best_v = v_new
                    print(f"✨ 发现新最优: {best_energy:.4f}s at iter {iteration_count}")

        # 温度衰减（加快速度）
        T *= alpha

        # 如果连续多个阶段没有改进，提前终止
        if len(energy_history) > 50 and np.std(energy_history[-20:]) < 0.1:
            print(f"提前终止: 阶段{phase + 1}已达到收敛条件")
            break

    return best_S, best_t_burst, best_energy, best_v


# ---------------- 主函数 ----------------
def main():
    print("=" * 60)
    print("烟幕干扰弹投放策略优化 - 带速度约束的黑洞退火算法")
    print("温度衰减速度加快版本")
    print("=" * 60)

    t0 = time.time()
    best_burst, best_t_burst, best_coverage, best_v = enhanced_black_hole_anneal()
    elapsed = time.time() - t0

    print(f"\n优化完成! 耗时: {elapsed:.1f}s")

    # 最终速度验证
    is_valid, final_v, theta, drop_point = validate_drone_speed(best_burst, best_t_burst)

    print("\n" + "=" * 40)
    print("最终投放策略")
    print("=" * 40)
    print(f"起爆点: [{best_burst[0]:.4f}, {best_burst[1]:.4f}, {best_burst[2]:.4f}]")
    print(f"起爆时间: {best_t_burst:.4f} s")
    print(f"无人机速度: {final_v:.4f} m/s")
    print(f"飞行方向: {theta:.4f}°")
    print(f"投放点: [{drop_point[0]:.4f}, {drop_point[1]:.4f}, {drop_point[2]:.4f}]")
    print(f"最大遮蔽时长: {best_coverage:.4f} s")

    # 详细的速度约束验证
    if MIN_DRONE_SPEED <= final_v <= MAX_DRONE_SPEED:
        print("✅ 无人机速度约束完全满足!")
    else:
        print("❌ 无人机速度约束未满足!")

    # ==================== 验证算法用的随机测试数据 ====================
    # 注释：取消注释以下代码来测试算法在不同点的表现
    """
    print("\n" + "=" * 40)
    print("算法验证 - 随机测试点")
    print("=" * 40)

    test_points = [
        np.array([17200.0, 10.0, 1730.0]),
        np.array([17150.0, -15.0, 1710.0]),
        np.array([17250.0, 25.0, 1750.0])
    ]

    for i, test_point in enumerate(test_points):
        test_t, test_energy, test_v = optimize_burst_time_for_point(test_point)
        is_valid, valid_v, valid_theta, valid_drop = validate_drone_speed(test_point, test_t)
        print(f"测试点{i+1}: 能量={test_energy:.4f}s, 速度={valid_v:.4f}m/s, 有效={is_valid}")
    """

    # 保存结果
    os.makedirs("output", exist_ok=True)
    result_df = pd.DataFrame([{
        'x_burst': best_burst[0], 'y_burst': best_burst[1], 'z_burst': best_burst[2],
        't_burst': best_t_burst, 'v_drone': final_v, 'theta_deg': theta,
        'x_drop': drop_point[0], 'y_drop': drop_point[1], 'z_drop': drop_point[2],
        'coverage_s': best_coverage, 'speed_valid': is_valid
    }])

    result_df.to_excel("output/Q2_optimized_with_speed_check.xlsx", index=False, float_format='%.4f')
    print("\n结果已保存到: output/Q2_optimized_with_speed_check.xlsx")


if __name__ == "__main__":
    main()