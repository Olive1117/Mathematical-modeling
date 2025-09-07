import numpy as np

# 定义导弹和无人机位置
m1_pos = np.array([20000, 0, 2000])
m2_pos = np.array([19000, 600, 2100])
m3_pos = np.array([18000, -600, 1900])
missiles = [m1_pos, m2_pos, m3_pos]
missile_names = ['m1', 'm2', 'm3']

fy1_pos = np.array([17800, 0, 1800])
fy2_pos = np.array([12000, 1400, 1400])
fy3_pos = np.array([6000, -3000, 700])
fy4_pos = np.array([11000, 2000, 1800])
fy5_pos = np.array([13000, -2000, 1300])
drones = [fy1_pos, fy2_pos, fy3_pos, fy4_pos, fy5_pos]
drone_names = ['fy1', 'fy2', 'fy3', 'fy4', 'fy5']

# 烟雾持续时间限制（秒）
MAX_SMOKE_DURATION = 20.0
SMOKES_PER_DRONE = 3  # 每台无人机投放3个烟雾弹
SMOKE_DESCENT_TIME = 3.0  # 烟雾弹下降时间（秒）
GRAVITY = 9.8  # 重力加速度


def calculate_interception(m, fy, t, drone_speed):
    """计算拦截点和相关参数，使用固定的无人机速度"""
    m_xy = np.array([m[0], m[1]])
    fy_xy = np.array([fy[0], fy[1]])

    # 导弹参数
    V_m = 300  # 导弹速度
    d = 1000  # 最小距离

    # 计算导弹俯冲角度
    horizontal_range = np.sqrt(m[0] ** 2 + m[1] ** 2)
    theta = np.arctan(m[2] / horizontal_range)

    # 计算导弹飞行距离
    L = d + V_m * t
    L_horizontal = L * np.cos(theta)

    # 计算导弹预测落点
    alpha = np.arctan2(m[1], m[0])
    dx = L_horizontal * np.cos(alpha)
    dy = L_horizontal * np.sin(alpha)

    if m[1] > 0:
        dy = -dy

    P_xy = m_xy + np.array([-dx, dy])
    P_z = m[2] - L * np.sin(theta)
    blast_point = np.array([P_xy[0], P_xy[1], P_z])

    # 计算无人机到拦截点的距离
    distance_xy = np.linalg.norm(P_xy - fy_xy)

    # 计算所需时间
    required_time = distance_xy / drone_speed

    # 允许一定的时间偏差
    time_deviation = abs(required_time - t)
    if time_deviation > 2.0:
        return np.inf, None, None, None, None

    # 计算航向角
    direction_vector = P_xy - fy_xy
    heading = np.arctan2(direction_vector[1], direction_vector[0])

    # 根据爆点反推投放点（考虑烟雾弹下降时间）
    drop_height = blast_point[2] + 0.5 * GRAVITY * (SMOKE_DESCENT_TIME ** 2)
    drop_point = np.array([blast_point[0], blast_point[1], drop_height])

    return drone_speed, blast_point, drop_point, t, heading


def find_optimal_drone_speed(m, fy):
    """为无人机找到最佳速度（70-140 m/s范围内）"""
    best_speed = None
    best_score = float('-inf')

    # 尝试所有可能的速度
    for speed in np.arange(70, 141, 10):
        valid_times = 0
        total_time = 0

        # 检查多个时间点
        for t in np.arange(5, 60, 5):
            result = calculate_interception(m, fy, t, speed)
            if result[0] != np.inf:
                valid_times += 1
                total_time += t

        # 评分标准：有效时间点数量 + 总时间
        score = valid_times + total_time / 10

        if score > best_score and valid_times > 0:
            best_score = score
            best_speed = speed

    return best_speed


def assign_all_drones_to_missiles():
    """为所有五台无人机分配导弹，确保每台都有任务"""
    # 初始化分配结果
    assignments = {}
    missile_assignments = {'m1': [], 'm2': [], 'm3': []}
    drone_assignment_status = {name: False for name in drone_names}

    print("为所有五台无人机分配导弹...")
    print("=" * 60)

    # 第一轮：为每台无人机找到最佳导弹
    for i, drone in enumerate(drones):
        drone_name = drone_names[i]
        best_missile = None
        best_total_time = 0
        best_smoke_times = []
        best_speed = None
        best_smoke_details = []

        # 为每台无人机找到最佳导弹
        for missile_idx, missile in enumerate(missiles):
            missile_name = missile_names[missile_idx]

            # 找到最佳无人机速度
            drone_speed = find_optimal_drone_speed(missile, drone)
            if drone_speed is None:
                continue

            # 尝试多个时间点找到最佳的3个烟雾弹
            candidate_times = []
            for t in np.arange(5, 60, 5):
                result = calculate_interception(missile, drone, t, drone_speed)
                if result[0] != np.inf:
                    candidate_times.append(t)

            # 选择时间间隔较大的3个时间点
            if len(candidate_times) >= 3:
                # 选择间隔较大的时间点
                selected_times = [candidate_times[0], candidate_times[len(candidate_times) // 2], candidate_times[-1]]

                smoke_details = []
                total_time = 0
                for t in selected_times:
                    details = calculate_drop_and_blast_points(missile, drone, drone_speed, t, len(smoke_details) + 1)
                    if details:
                        smoke_details.append(details)
                        total_time += MAX_SMOKE_DURATION

                if total_time > best_total_time and len(smoke_details) == 3:
                    best_total_time = total_time
                    best_missile = missile_name
                    best_smoke_times = selected_times
                    best_speed = drone_speed
                    best_smoke_details = smoke_details

        if best_missile:
            missile_obj = missiles[missile_names.index(best_missile)]

            assignments[f"{best_missile}_{drone_name}"] = {
                'total_time': best_total_time,
                'smoke_details': best_smoke_details,
                'drone': drone,
                'missile': missile_obj,
                'drone_speed': best_speed
            }
            missile_assignments[best_missile].append(drone_name)
            drone_assignment_status[drone_name] = True

            print(f"\n无人机 {drone_name} -> 导弹 {best_missile}")
            print(f"选定速度: {best_speed:.1f} m/s")
            print(f"理论总遮蔽时间: {best_total_time:.2f}s")
            print("烟雾弹投放时间:", [f"{t:.1f}s" for t in best_smoke_times])

    # 第二轮：确保所有无人机都被分配
    unassigned_drones = [name for name, assigned in drone_assignment_status.items() if not assigned]

    if unassigned_drones:
        print(f"\n重新分配未分配的无人机: {', '.join(unassigned_drones)}")

        for drone_name in unassigned_drones:
            drone_idx = drone_names.index(drone_name)
            drone = drones[drone_idx]

            # 尝试所有导弹，找到可用的
            for missile_idx, missile in enumerate(missiles):
                missile_name = missile_names[missile_idx]

                # 跳过已经分配过多无人机的导弹
                if len(missile_assignments[missile_name]) >= 2:
                    continue

                drone_speed = find_optimal_drone_speed(missile, drone)
                if drone_speed is None:
                    continue

                # 尝试找到可用的时间点
                available_times = []
                for t in np.arange(5, 60, 10):
                    result = calculate_interception(missile, drone, t, drone_speed)
                    if result[0] != np.inf:
                        available_times.append(t)

                if len(available_times) >= 1:
                    # 使用可用时间点
                    selected_times = available_times[:min(3, len(available_times))]
                    smoke_details = []

                    for i, t in enumerate(selected_times):
                        details = calculate_drop_and_blast_points(missile, drone, drone_speed, t, i + 1)
                        if details:
                            smoke_details.append(details)

                    if smoke_details:
                        missile_obj = missiles[missile_idx]
                        total_time = len(smoke_details) * MAX_SMOKE_DURATION

                        assignments[f"{missile_name}_{drone_name}"] = {
                            'total_time': total_time,
                            'smoke_details': smoke_details,
                            'drone': drone,
                            'missile': missile_obj,
                            'drone_speed': drone_speed
                        }
                        missile_assignments[missile_name].append(drone_name)
                        drone_assignment_status[drone_name] = True

                        print(f"无人机 {drone_name} -> 导弹 {missile_name} (备用分配)")
                        print(f"投放时间: {[f'{t:.1f}s' for t in selected_times]}")
                        break

    return assignments, missile_assignments, drone_assignment_status


def calculate_drop_and_blast_points(m, fy, drone_speed, t_rel, smoke_index):
    """计算投放点和爆点"""
    result = calculate_interception(m, fy, t_rel, drone_speed)
    if result[0] != np.inf:
        v, blast_point, drop_point, explosion_time, heading = result

        return {
            'smoke_index': smoke_index,
            'drop_point': drop_point,
            'blast_point': blast_point,
            'explosion_time': explosion_time,
            'speed': drone_speed,
            'heading': heading,
            't_rel': t_rel
        }
    return None


def calculate_total_time(missile_assignments, assignments):
    """计算总遮蔽时间（不考虑重复遮挡）"""
    missile_total_times = {'m1': 0, 'm2': 0, 'm3': 0}
    total_time = 0

    # 按导弹计算总时间
    for missile_name in missile_names:
        assigned_drones = missile_assignments[missile_name]

        for drone_name in assigned_drones:
            key = f"{missile_name}_{drone_name}"
            if key in assignments:
                info = assignments[key]
                missile_total_times[missile_name] += info['total_time']
                total_time += info['total_time']

    return total_time, missile_total_times


def radians_to_degrees(rad):
    """弧度转角度"""
    deg = rad * 180 / np.pi
    # 确保角度在0-360范围内
    if deg < 0:
        deg += 360
    if deg >= 360:
        deg -= 360
    return deg


def format_results_to_excel(assignments, missile_assignments):
    """格式化结果为Excel输出格式"""
    results = []

    for drone_name in drone_names:
        # 找到这台无人机分配的导弹
        assigned_missile = None
        for missile_name in missile_names:
            if drone_name in missile_assignments[missile_name]:
                assigned_missile = missile_name
                break

        if assigned_missile:
            key = f"{assigned_missile}_{drone_name}"
            info = assignments[key]

            for smoke_detail in info['smoke_details']:
                heading_deg = radians_to_degrees(smoke_detail['heading'])

                results.append({
                    '无人机编号': drone_name,
                    '无人机运动方向': f"{heading_deg:.1f}",
                    '无人机运动速度': f"{info['drone_speed']:.1f}",
                    '烟幕干扰弹编号': smoke_detail['smoke_index'],
                    '投放点X': f"{smoke_detail['drop_point'][0]:.1f}",
                    '投放点Y': f"{smoke_detail['drop_point'][1]:.1f}",
                    '投放点Z': f"{smoke_detail['drop_point'][2]:.1f}",
                    '起爆点X': f"{smoke_detail['blast_point'][0]:.1f}",
                    '起爆点Y': f"{smoke_detail['blast_point'][1]:.1f}",
                    '起爆点Z': f"{smoke_detail['blast_point'][2]:.1f}",
                    '有效干扰时长': f"{MAX_SMOKE_DURATION:.1f}",
                    '干扰的导弹编号': assigned_missile
                })
        else:
            # 如果没有分配，创建空行
            for i in range(1, 4):
                results.append({
                    '无人机编号': drone_name,
                    '无人机运动方向': '',
                    '无人机运动速度': '',
                    '烟幕干扰弹编号': i,
                    '投放点X': '',
                    '投放点Y': '',
                    '投放点Z': '',
                    '起爆点X': '',
                    '起爆点Y': '',
                    '起爆点Z': '',
                    '有效干扰时长': '',
                    '干扰的导弹编号': ''
                })

    return results


if __name__ == '__main__':
    # 为所有五台无人机分配导弹
    assignments, missile_assignments, drone_status = assign_all_drones_to_missiles()

    # 计算总遮蔽时间（不考虑重复遮挡）
    total_time, missile_total_times = calculate_total_time(missile_assignments, assignments)

    print("\n" + "=" * 80)
    print("最终无人机分配方案（所有五台无人机）：")
    print("=" * 80)

    # 显示详细分配结果
    for missile_name in missile_names:
        assigned_drones = missile_assignments[missile_name]

        print(f"\n导弹 {missile_name}:")
        print("-" * 60)
        print(f"分配的无人机: {', '.join(assigned_drones) if assigned_drones else '无'}")
        print(f"总遮蔽时间: {missile_total_times[missile_name]:.2f}s")

        for drone_name in assigned_drones:
            key = f"{missile_name}_{drone_name}"
            if key in assignments:
                info = assignments[key]

                print(f"\n无人机 {drone_name}:")
                print(f"飞行速度: {info['drone_speed']:.1f} m/s")
                print(f"遮蔽时间: {info['total_time']:.2f}s")
                print("烟雾弹投放详情:")

                for smoke_detail in info['smoke_details']:
                    print(f"  烟雾弹{smoke_detail['smoke_index']}:")
                    print(f"    投放时间: {smoke_detail['t_rel']:.1f}s")
                    print(
                        f"    投放点: ({smoke_detail['drop_point'][0]:.1f}, {smoke_detail['drop_point'][1]:.1f}, {smoke_detail['drop_point'][2]:.1f})")
                    print(
                        f"    爆点: ({smoke_detail['blast_point'][0]:.1f}, {smoke_detail['blast_point'][1]:.1f}, {smoke_detail['blast_point'][2]:.1f})")
                    print(f"    飞行速度: {smoke_detail['speed']:.1f} m/s")
                    print(f"    航向角: {radians_to_degrees(smoke_detail['heading']):.1f}°")

    print(f"\n所有无人机总遮蔽时间: {total_time:.2f} 秒")

    # 统计信息
    print(f"\n统计信息:")
    print(f"单烟雾最大持续时间: {MAX_SMOKE_DURATION}秒")
    print(f"每台无人机烟雾弹数量: {SMOKES_PER_DRONE}个")

    for missile_name in missile_names:
        count = len(missile_assignments[missile_name])
        print(f"导弹 {missile_name}: {count} 台无人机")

    # 检查无人机分配状态
    assigned_count = sum(1 for status in drone_status.values() if status)
    print(f"\n已分配无人机: {assigned_count}/5")

    if assigned_count == 5:
        print("✓ 所有五台无人机都已成功分配！")
    else:
        unassigned = [name for name, status in drone_status.items() if not status]
        print(f"⚠ 未分配无人机: {', '.join(unassigned)}")

    # 格式化结果用于Excel输出
    excel_results = format_results_to_excel(assignments, missile_assignments)
    print(f"\nExcel格式结果已准备，共 {len(excel_results)} 行数据")