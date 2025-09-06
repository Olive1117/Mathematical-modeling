from scipy.optimize import basinhopping, minimize
from core import *
from missile_search import validity_time
"""
Q1爆点为[17188, 0, 1736.496]
以该爆点为退火起始点
"""
x = np.array([17188, 0, 1736.496, 5.1])
def objective(x):
    c1 = cloud_closure(x[0], x[1], x[2], x[3])
    return -validity_time(m1, target_true_pos, c1, x[3])

# ---------- 1. 生成邻域：各维度独立自适应步长 ----------
def generate_neighbor(x, bounds, step_ratio):
    """
    step_ratio: 长度=dim 的数组，表示每一步对该维度区间宽度的比例
    """
    neighbor = x.copy()
    dim = len(x)
    idx = np.random.randint(dim)          # 随机挑一个维度动
    width = bounds[idx][1] - bounds[idx][0]
    delta = np.random.uniform(-1, 1) * step_ratio[idx] * width
    neighbor[idx] += delta
    neighbor[idx] = max(bounds[idx][0], min(neighbor[idx], bounds[idx][1]))
    return neighbor

# 模拟退火算法
def simulated_annealing(objective_func, bounds, initial_temp, cooling_rate,
                        max_iter, step_size, verbose=2):
    dim = len(bounds)
    # 初始比例：让第一步的「平均位移 ≈ 5 % 区间宽度」
    initial_step_ratio = np.full(4, 0.05)  # 5 %，可自己调
    current_step_ratio = initial_step_ratio.copy()
    # 初始化当前解
    current_solution = x
    current_value = objective_func(current_solution)

    # 初始化最优解
    best_solution = current_solution.copy()
    best_value = current_value

    # 记录历史
    current_temp = initial_temp
    best_history = [best_value]
    step_count = 0
    accept_window = 0  # 最近 100 步的接受计数
    history_accept = []  # 滑动窗口，记录最近 100 步是否接受

    def log(header=False):
        """打印一行日志"""
        if header:
            print("{:>6} {:>10} {:>12} {:>12} {:>8} {:>10}".format(
                "iter", "temp", "current", "best", "acc_rate", "step_size"))
            print("-" * 64)
        else:
            accept_rate = accept_window / max(1, len(history_accept))
            print("{:>6} {:>10.2f} {:>12.4f} {:>12.4f} {:>7.1%} {:>10.2f} {}".format(
                step_count, current_temp, current_value, best_value, accept_rate, current_step_ratio, current_solution))

    if verbose >= 1:
        log(header=True)

    # 主循环
    for i in range(max_iter):
        step_count += 1
        # 生成邻域解
        # 在主循环里替换掉原来的 generate_neighbor 调用：
        neighbor = generate_neighbor(current_solution, bounds, current_step_ratio)
        neighbor_value = objective_func(neighbor)

        # 计算能量差（目标函数差）
        delta = neighbor_value - current_value

        # 接受准则：如果更优则接受，否则以一定概率接受
        if delta < 0 or np.random.rand() < np.exp(-delta / current_temp):
            current_solution = neighbor.copy()
            current_value = neighbor_value

            # 更新最优解
            if current_value < best_value:
                best_solution = current_solution.copy()
                best_value = current_value

        # 降温
        current_temp *= cooling_rate

        # 记录历史
        best_history.append(best_value)

        # 每 100 步根据接受率再缩放一次
        if i % 100 == 0 and i > 0:
            accept_rate = accept_window / max(1, len(history_accept))
            if accept_rate > 0.5:
                current_step_ratio *= 1.2  # 接受太多，步子放大
            elif accept_rate < 0.2:
                current_step_ratio *= 0.8  # 接受太少，步子缩小
            # 给每一步比例设上下限，防止爆炸或冻住
            current_step_ratio = np.clip(current_step_ratio, 0.001, 0.3)

        # 日志输出
        if verbose == 2 or (verbose == 1 and i % 100 == 0):
            log()

    return best_solution, best_value, best_history


# 参数设置
bounds = [
    (10000, 18000),  # x
    (-200, 200),   # y
    (0, 2000),    # z
    (0, 50)          # t_blast
]
initial_temp = 100   # 初始温度
cooling_rate = 0.98 # 冷却速率
max_iter = 1000  # 最大迭代次数
step_size = 150.0  # 初始步长

# 运行模拟退火算法
best_solution, best_value, best_history = simulated_annealing(
    validity_time_array, bounds, initial_temp, cooling_rate, max_iter, step_size
)
# 输出结果
print("模拟退火算法求解干扰时间最小化结果：")
print(f"最优爆点参数：x={best_solution[0]:.2f}, y={best_solution[1]:.2f}, z={best_solution[2]:.2f}, t={best_solution[3]:.2f}")
print(f"最小干扰时间：{best_value:.4f} 秒")
print(validity_time_array(x))