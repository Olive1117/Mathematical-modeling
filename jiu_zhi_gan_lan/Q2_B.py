import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class Vec3:
    """三维向量类"""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(self.y * other.z - self.z * other.y,
                    self.z * other.x - self.x * other.z,
                    self.x * other.y - self.y * other.x)

    def norm(self):
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        n = self.norm()
        if n == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / n, self.y / n, self.z / n)

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Missile:
    """导弹类"""

    def __init__(self, initial_pos, speed, target_pos):
        self.initial_pos = Vec3(*initial_pos)
        self.speed = speed
        self.target_pos = Vec3(*target_pos)
        self.direction = (self.target_pos - self.initial_pos).normalize()

    def position_at_time(self, t):
        """导弹在时间t的位置"""
        distance = self.speed * t
        return self.initial_pos + self.direction * distance


class Cloud:
    """烟幕云团类"""

    def __init__(self, burst_pos, burst_time, sink_speed=3, radius=10):
        self.burst_pos = Vec3(*burst_pos)
        self.burst_time = burst_time
        self.sink_speed = sink_speed
        self.radius = radius

    def position_at_time(self, t):
        """云团在时间t的位置"""
        if t < self.burst_time:
            return self.burst_pos  # 还未起爆
        sink_time = t - self.burst_time
        sink_distance = self.sink_speed * sink_time
        return Vec3(self.burst_pos.x, self.burst_pos.y, self.burst_pos.z - sink_distance)


class CylindricalTarget:
    """圆柱形目标类"""

    def __init__(self, center_bottom, radius, height):
        self.center_bottom = Vec3(*center_bottom)
        self.radius = radius
        self.height = height
        self.center_top = Vec3(center_bottom[0], center_bottom[1], center_bottom[2] + height)


def missile_can_see_target(missile_pos, cloud, target, time_step):
    """
    判断导弹是否能看见目标（简化的LOS判断）
    使用距离判断作为快速近似
    """
    # 导弹到目标的连线
    missile_to_target = target.center_bottom - missile_pos
    missile_to_target_dist = missile_to_target.norm()

    if missile_to_target_dist == 0:
        return False

    # 云团到连线的距离
    missile_to_cloud = cloud.position_at_time(time_step) - missile_pos
    cross_product = missile_to_cloud.cross(missile_to_target)
    distance_to_line = cross_product.norm() / missile_to_target_dist

    return distance_to_line > cloud.radius


def calculate_coverage_time(burst_params, missile, target, total_time=100, time_step=0.1):
    """
    计算给定起爆参数下的遮蔽时间
    burst_params: [x, y, z, t_burst]
    """
    x, y, z, t_burst = burst_params
    cloud = Cloud([x, y, z], t_burst)

    coverage_time = 0
    for t in np.arange(0, total_time, time_step):
        missile_pos = missile.position_at_time(t)
        if not missile_can_see_target(missile_pos, cloud, target, t):
            coverage_time += time_step

    return coverage_time


def is_drone_speed_valid(drone_initial_pos, drop_pos, drop_time, min_speed=70, max_speed=140):
    """
    检查无人机速度是否在有效范围内
    """
    dx = drop_pos[0] - drone_initial_pos[0]
    dy = drop_pos[1] - drone_initial_pos[1]
    dz = drop_pos[2] - drone_initial_pos[2]

    # 水平距离（等高度飞行）
    horizontal_distance = np.sqrt(dx ** 2 + dy ** 2)

    if drop_time <= 0:
        return False, 0

    speed = horizontal_distance / drop_time
    return min_speed <= speed <= max_speed, speed


def calculate_drop_position(burst_pos, burst_time, g=9.8):
    """
    根据起爆点和起爆时间计算投放点
    """
    # 自由落体位移补偿
    drop_z = burst_pos[2] + 0.5 * g * burst_time ** 2
    return [burst_pos[0], burst_pos[1], drop_z]


def density_based_refinement(initial_params, missile, target, drone_initial_pos, iterations=100,
                             search_radius=100, learning_rate=0.1):
    """
    基于密度累积的细筛优化，考虑无人机速度约束
    """
    best_params = initial_params.copy()
    best_coverage = calculate_coverage_time(best_params, missile, target)

    density_map = {}
    improvement_history = []
    valid_solutions = 0

    for i in range(iterations):
        # 在当前位置周围随机采样
        perturbation = np.random.normal(0, search_radius, 4)
        candidate_params = best_params + learning_rate * perturbation

        # 确保参数在合理范围内
        candidate_params[3] = max(0.1, candidate_params[3])  # t_burst >= 0.1s

        # 计算投放点
        drop_pos = calculate_drop_position(candidate_params[:3], candidate_params[3])

        # 检查无人机速度是否有效
        is_valid, drone_speed = is_drone_speed_valid(drone_initial_pos, drop_pos, candidate_params[3])

        if not is_valid:
            # 无效解，给予惩罚
            coverage = -1
        else:
            # 有效解，计算遮蔽时间
            coverage = calculate_coverage_time(candidate_params, missile, target)
            valid_solutions += 1

        # 更新密度图
        key = tuple(np.round(candidate_params[:3]).astype(int))
        density_map[key] = density_map.get(key, 0) + 1

        # 更新最优解（只考虑有效解）
        if coverage > best_coverage and is_valid:
            improvement = coverage - best_coverage
            best_coverage = coverage
            best_params = candidate_params.copy()
            improvement_history.append(improvement)

            if len(improvement_history) > 10:
                improvement_history.pop(0)

        # 自适应调整搜索半径
        if len(improvement_history) >= 5:
            avg_improvement = np.mean(improvement_history[-5:])
            if avg_improvement < 0.1:
                search_radius *= 0.9
            else:
                search_radius = min(200, search_radius * 1.1)

        # 每20次迭代输出结果
        if (i + 1) % 20 == 0:
            valid_ratio = valid_solutions / (i + 1) * 100
            print(f"Iteration {i + 1}: Best coverage = {best_coverage:.2f}s, "
                  f"Valid solutions = {valid_ratio:.1f}%, "
                  f"Search radius = {search_radius:.2f}")

    # 分析密度热点
    if density_map:
        dense_point = max(density_map.items(), key=lambda x: x[1])[0]
        print(f"Density hotspot: {dense_point} (count: {density_map[dense_point]})")

    return best_params, best_coverage


def coarse_search(missile, target, drone_initial_pos, n_samples=1000):
    """
    粗筛：在可能的空间范围内随机采样，考虑无人机速度约束
    """
    print("Starting coarse search with drone speed constraints...")
    best_coverage = 0
    best_params = None
    valid_solutions = 0

    # 定义搜索范围（根据问题设定调整）
    x_range = (17000, 19000)
    y_range = (-500, 500)
    z_range = (1700, 2000)
    t_range = (1, 15)  # 最短投放时间1秒，确保速度合理

    for i in tqdm(range(n_samples)):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = np.random.uniform(*z_range)
        t_burst = np.random.uniform(*t_range)

        params = [x, y, z, t_burst]

        # 计算投放点并检查速度约束
        drop_pos = calculate_drop_position([x, y, z], t_burst)
        is_valid, drone_speed = is_drone_speed_valid(drone_initial_pos, drop_pos, t_burst)

        if not is_valid:
            continue

        valid_solutions += 1
        coverage = calculate_coverage_time(params, missile, target)

        if coverage > best_coverage:
            best_coverage = coverage
            best_params = params

    valid_ratio = valid_solutions / n_samples * 100
    print(f"Coarse search completed. Valid solutions: {valid_ratio:.1f}%")
    print(f"Best coverage: {best_coverage:.2f}s")

    return best_params, best_coverage


def optimize_missile_coverage():
    """主优化函数"""
    # 初始化参数
    missile_initial_pos = [20000, 0, 2000]
    missile_speed = 300
    target_center = [0, 200, 0]
    target_radius = 7
    target_height = 10
    drone_initial_pos = [17800, 0, 1800]

    # 创建对象
    missile = Missile(missile_initial_pos, missile_speed, target_center)
    target = CylindricalTarget(target_center, target_radius, target_height)

    # 阶段1: 粗筛（考虑速度约束）
    start_time = time.time()
    coarse_params, coarse_coverage = coarse_search(missile, target, drone_initial_pos, n_samples=2000)
    coarse_time = time.time() - start_time
    print(f"Coarse search time: {coarse_time:.2f}s")

    # 阶段2: 细筛（考虑速度约束）
    print("\nStarting density-based refinement with drone speed constraints...")
    start_time = time.time()
    refined_params, refined_coverage = density_based_refinement(
        coarse_params, missile, target, drone_initial_pos, iterations=200, search_radius=50
    )
    refine_time = time.time() - start_time
    print(f"Refinement time: {refine_time:.2f}s")

    # 最终结果
    print(f"\nFinal result:")
    print(f"Optimal burst position: ({refined_params[0]:.2f}, {refined_params[1]:.2f}, {refined_params[2]:.2f})")
    print(f"Optimal burst time: {refined_params[3]:.2f}s")

    # 计算无人机策略
    drop_pos = calculate_drop_position(refined_params[:3], refined_params[3])
    is_valid, drone_speed = is_drone_speed_valid(drone_initial_pos, drop_pos, refined_params[3])

    if not is_valid:
        print("Warning: Final solution violates drone speed constraints!")
        # 重新计算速度（即使超出范围也显示）
        dx = drop_pos[0] - drone_initial_pos[0]
        dy = drop_pos[1] - drone_initial_pos[1]
        horizontal_distance = np.sqrt(dx ** 2 + dy ** 2)
        drone_speed = horizontal_distance / refined_params[3]

    print(f"Maximum coverage time: {refined_coverage:.2f}s")
    print(f"\nDrone strategy:")
    print(f"Drop position: ({drop_pos[0]:.2f}, {drop_pos[1]:.2f}, {drop_pos[2]:.2f})")
    print(f"Flight time: {refined_params[3]:.2f}s")
    print(f"Drone speed: {drone_speed:.2f}m/s")

    # 计算航向角
    dx = drop_pos[0] - drone_initial_pos[0]
    dy = drop_pos[1] - drone_initial_pos[1]
    heading_xy = np.arctan2(dy, dx)

    print(f"XY heading: {np.degrees(heading_xy):.2f}°")

    # 检查速度是否在有效范围内
    if 70 <= drone_speed <= 140:
        print(f"✓ Drone speed is within valid range [70, 140] m/s")
    else:
        print(f"✗ Drone speed {drone_speed:.2f} m/s is outside valid range [70, 140] m/s")

    return refined_params, refined_coverage, drone_speed


# 运行优化
if __name__ == "__main__":
    optimal_params, max_coverage, drone_speed = optimize_missile_coverage()