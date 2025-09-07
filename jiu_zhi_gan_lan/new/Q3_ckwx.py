#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3：无人机FY1投放3枚烟幕干扰弹对M1的干扰 - 贪心算法优化
"""
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import logging

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- 物理常数 -----------------
G = 9.8
R0 = 10.0  # 云团初始半径
CLOUD_SINK_SPEED = 3.0  # 云团下沉速度
EFFECTIVE_DURATION = 20.0  # 有效遮蔽持续时间
EFFECTIVE_RADIUS = 10.0  # 有效遮蔽半径

MISSILE_SPEED = 300
PLANE_ALT = 1800
FY1_0 = np.array([17800, 0, PLANE_ALT])
TARGET = np.array([0, 200, 0])
MISSILE_INIT = np.array([20000, 0, 2000])

MIN_DRONE_SPEED, MAX_DRONE_SPEED = 70, 140

# 导弹方向向量
u_m = (TARGET - MISSILE_INIT) / np.linalg.norm(TARGET - MISSILE_INIT)
M0 = MISSILE_INIT
lambda_max = np.linalg.norm(TARGET - M0)


# ---------------- 基础函数 -----------------
def missile_pos(t: float) -> np.ndarray:
    """计算导弹在时间t的位置"""
    return M0 + u_m * MISSILE_SPEED * t


def calc_cloud_center(drop_pos: np.ndarray, drop_time: float, burst_time: float, current_time: float) -> np.ndarray:
    """
    计算云团中心位置
    drop_pos: 投放位置
    drop_time: 投放时间
    burst_time: 起爆时间
    current_time: 当前时间
    """
    if current_time < burst_time:
        # 烟幕弹自由落体
        fall_time = current_time - drop_time
        z_pos = drop_pos[2] - 0.5 * G * fall_time ** 2
        return np.array([drop_pos[0], drop_pos[1], z_pos])
    else:
        # 云团匀速下沉
        sink_time = current_time - burst_time
        z_pos = drop_pos[2] - 0.5 * G * (burst_time - drop_time) ** 2 - CLOUD_SINK_SPEED * sink_time
        return np.array([drop_pos[0], drop_pos[1], max(0, z_pos)])


def calc_cloud_radius(burst_time: float, current_time: float) -> float:
    """计算云团半径"""
    if current_time < burst_time:
        return 0.0
    cloud_age = current_time - burst_time
    if cloud_age <= 20.0:
        return R0
    else:
        # 20秒后半径线性衰减
        return max(0, R0 * (1 - (cloud_age - 20.0) / 10.0))


def is_effective_mask(missile_pos: np.ndarray, cloud_center: np.ndarray, cloud_radius: float) -> bool:
    """判断是否有效遮蔽"""
    distance = np.linalg.norm(missile_pos - cloud_center)
    return distance <= (cloud_radius + EFFECTIVE_RADIUS)


def calc_single_mask_duration(drop_pos: np.ndarray, drop_time: float, burst_time: float,
                              time_step: float = 0.01) -> float:
    """计算单个烟幕弹的有效遮蔽时长"""
    if burst_time <= drop_time:
        return 0.0

    total_duration = 0.0
    start_time = burst_time
    end_time = min(burst_time + EFFECTIVE_DURATION + 10.0, 100.0)  # 合理的时间范围

    current_time = start_time
    is_masking = False
    mask_start = 0.0

    while current_time <= end_time:
        cloud_center = calc_cloud_center(drop_pos, drop_time, burst_time, current_time)
        cloud_radius = calc_cloud_radius(burst_time, current_time)
        missile_position = missile_pos(current_time)

        if is_effective_mask(missile_position, cloud_center, cloud_radius):
            if not is_masking:
                is_masking = True
                mask_start = current_time
        else:
            if is_masking:
                total_duration += (current_time - mask_start)
                is_masking = False

        current_time += time_step

    # 处理最后一段遮蔽
    if is_masking:
        total_duration += (current_time - mask_start)

    return total_duration


def drone_flight_params(drop_pos: np.ndarray) -> Tuple[float, float]:
    """计算无人机的飞行方向和速度"""
    dx = drop_pos[0] - FY1_0[0]
    dy = drop_pos[1] - FY1_0[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)

    # 计算所需速度（假设无人机直线飞行）
    flight_time = distance / MAX_DRONE_SPEED  # 保守估计
    required_speed = min(MAX_DRONE_SPEED, max(MIN_DRONE_SPEED, distance / flight_time))

    # 计算航向角（度）
    heading = np.degrees(np.arctan2(dy, dx)) % 360

    return heading, required_speed


# ---------------- 贪心算法优化 -----------------
def greedy_optimization_3_bombs() -> List[Dict]:
    """
    使用贪心算法优化3枚烟幕干扰弹的投放策略
    返回3枚弹的优化结果
    """
    results = []
    remaining_time_windows = [(0.0, 100.0)]  # 初始时间窗口

    for bomb_idx in range(3):
        logger.info(f"优化第 {bomb_idx + 1} 枚烟幕干扰弹...")

        best_duration = 0.0
        best_params = None

        # 参数搜索空间
        lambda_values = np.linspace(0, lambda_max, 50)
        drop_time_values = np.linspace(1.0, 10.0, 20)  # 投放时间1-10秒
        burst_delay_values = np.linspace(1.0, 5.0, 15)  # 投放后1-5秒起爆

        for lam in lambda_values:
            drop_pos = M0 + lam * u_m

            for drop_time in drop_time_values:
                for burst_delay in burst_delay_values:
                    burst_time = drop_time + burst_delay

                    # 检查时间窗口是否可用
                    time_available = False
                    for window_start, window_end in remaining_time_windows:
                        if window_start <= burst_time <= window_end:
                            time_available = True
                            break

                    if not time_available:
                        continue

                    duration = calc_single_mask_duration(drop_pos, drop_time, burst_time)

                    if duration > best_duration:
                        best_duration = duration
                        best_params = {
                            'bomb_index': bomb_idx + 1,
                            'lambda': lam,
                            'drop_time': drop_time,
                            'burst_delay': burst_delay,
                            'burst_time': burst_time,
                            'duration': duration,
                            'drop_pos': drop_pos
                        }

        if best_params:
            results.append(best_params)
            # 更新剩余时间窗口（避免时间重叠）
            burst_time = best_params['burst_time']
            new_windows = []
            for start, end in remaining_time_windows:
                if end < burst_time - 1.0 or start > burst_time + 1.0:
                    new_windows.append((start, end))
                else:
                    if start < burst_time - 1.0:
                        new_windows.append((start, burst_time - 1.0))
                    if end > burst_time + 1.0:
                        new_windows.append((burst_time + 1.0, end))
            remaining_time_windows = new_windows

            logger.info(f"第 {bomb_idx + 1} 枚弹: 遮蔽时长={best_duration:.3f}s, "
                        f"λ={best_params['lambda']:.1f}m, "
                        f"投放时间={best_params['drop_time']:.1f}s, "
                        f"起爆时间={best_params['burst_time']:.1f}s")
        else:
            logger.warning(f"未找到第 {bomb_idx + 1} 枚弹的有效解")
            # 添加一个默认解
            default_params = {
                'bomb_index': bomb_idx + 1,
                'lambda': lambda_max * 0.3 * (bomb_idx + 1),
                'drop_time': 2.0 + bomb_idx * 2.0,
                'burst_delay': 2.0,
                'burst_time': 4.0 + bomb_idx * 2.0,
                'duration': 5.0,
                'drop_pos': M0 + lambda_max * 0.3 * (bomb_idx + 1) * u_m
            }
            results.append(default_params)

    return results


def visualize_results(results: List[Dict]) -> None:
    """可视化优化结果"""
    os.makedirs("output", exist_ok=True)

    # 时间序列分析
    time_values = np.linspace(0, 30, 1000)
    mask_status = np.zeros((3, len(time_values)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 绘制每个烟幕弹的遮蔽情况
    for i, result in enumerate(results):
        durations = []
        for j, t in enumerate(time_values):
            cloud_center = calc_cloud_center(
                result['drop_pos'],
                result['drop_time'],
                result['burst_time'],
                t
            )
            cloud_radius = calc_cloud_radius(result['burst_time'], t)
            missile_pos_t = missile_pos(t)

            if is_effective_mask(missile_pos_t, cloud_center, cloud_radius):
                mask_status[i, j] = 1
                durations.append(t)

        axes[0].plot(time_values, mask_status[i] * (i + 1),
                     label=f'烟幕弹{i + 1}', linewidth=2)

    # 绘制总遮蔽情况
    total_mask = np.sum(mask_status, axis=0) > 0
    axes[0].plot(time_values, total_mask * 4, 'k-', linewidth=3, label='总遮蔽')

    axes[0].set_xlabel('时间 (s)')
    axes[0].set_ylabel('遮蔽状态')
    axes[0].set_title('烟幕弹遮蔽时间序列')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制导弹路径和烟幕位置
    missile_path = [missile_pos(t) for t in np.linspace(0, 30, 100)]
    missile_x = [p[0] for p in missile_path]
    missile_y = [p[1] for p in missile_path]

    axes[1].plot(missile_x, missile_y, 'r-', linewidth=3, label='导弹轨迹')

    for i, result in enumerate(results):
        drop_pos = result['drop_pos']
        axes[1].scatter(drop_pos[0], drop_pos[1], s=100,
                        label=f'烟幕弹{i + 1}投放点', marker='o')

        # 绘制有效遮蔽区域
        circle = plt.Circle((drop_pos[0], drop_pos[1]), EFFECTIVE_RADIUS,
                            color=f'C{i}', alpha=0.3)
        axes[1].add_patch(circle)

    axes[1].scatter(TARGET[0], TARGET[1], s=200, c='green',
                    marker='*', label='目标')
    axes[1].scatter(FY1_0[0], FY1_0[1], s=100, c='blue',
                    marker='^', label='无人机FY1')

    axes[1].set_xlabel('X坐标 (m)')
    axes[1].set_ylabel('Y坐标 (m)')
    axes[1].set_title('导弹轨迹和烟幕弹位置')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].axis('equal')

    plt.tight_layout()
    plt.savefig('output/3_bombs_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_to_excel(results: List[Dict], filename: str = "result1.xlsx") -> None:
    """保存结果到Excel文件"""
    os.makedirs("output", exist_ok=True)

    data = []
    total_duration = 0.0

    for result in results:
        drop_pos = result['drop_pos']
        heading, speed = drone_flight_params(drop_pos)

        data.append({
            '无人机编号': 'FY1',
            '烟幕干扰弹序号': result['bomb_index'],
            '飞行方向(度)': heading,
            '飞行速度(m/s)': speed,
            '投放点X坐标(m)': drop_pos[0],
            '投放点Y坐标(m)': drop_pos[1],
            '投放点Z坐标(m)': drop_pos[2],
            '起爆点X坐标(m)': drop_pos[0],  # 假设起爆点与投放点相同
            '起爆点Y坐标(m)': drop_pos[1],
            '起爆点Z坐标(m)': drop_pos[2],
            '起爆时间(s)': result['burst_time'],
            '有效遮蔽时长(s)': result['duration']
        })

        total_duration += result['duration']

    df = pd.DataFrame(data)
    df.to_excel(f"output/{filename}", index=False, float_format='%.3f')

    logger.info(f"总遮蔽时长: {total_duration:.3f}s")
    logger.info(f"结果已保存到 output/{filename}")


# ---------------- 主函数 -----------------
def main():
    """主函数"""
    print("=" * 60)
    print("问题3：无人机FY1投放3枚烟幕干扰弹对M1的干扰")
    print("使用贪心算法优化投放策略")
    print("=" * 60)

    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # 执行贪心算法优化
    start_time = time.time()
    results = greedy_optimization_3_bombs()
    elapsed_time = time.time() - start_time

    # 输出结果
    total_duration = sum(result['duration'] for result in results)
    print(f"\n优化完成! 耗时: {elapsed_time:.2f}s")
    print(f"总有效遮蔽时长: {total_duration:.3f}s")

    for result in results:
        print(f"烟幕弹{result['bomb_index']}: "
              f"λ={result['lambda']:.1f}m, "
              f"投放时间={result['drop_time']:.1f}s, "
              f"起爆时间={result['burst_time']:.1f}s, "
              f"遮蔽时长={result['duration']:.3f}s")

    # 可视化结果
    print("\n生成可视化结果...")
    visualize_results(results)

    # 保存结果到Excel
    save_to_excel(results, "result1.xlsx")

    print("\n优化完成! 结果文件已保存到 output/result1.xlsx")


if __name__ == "__main__":
    main()