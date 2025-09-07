import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import basinhopping, minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from core import *
from missile_search import validity_time
import random
a = 1
while a:
    rand_seed = random.randint(0, 2 ** 32 - 1)
    print('本次随机种子 =', rand_seed)
    np.random.seed(rand_seed)


    def objective(params):
        """
        带速度约束的 objective
        如果 FY1 速度不在 [70,140] 之间，直接返回一个极大值
        """
        # 1. 先计算当前参数对应的投弹速度
        x, y, z, t_0 = params[0] * 1000, params[1], params[2]*100, params[3]
        fy1 = np.array([17800, 0, 1800])
        c_ = np.array([x, y, 1800])  # 保持你原来的假设
        dist = np.linalg.norm(c_ - fy1)
        v = dist / t_0  # 飞行速度

        # 2. 速度约束：不在 70–140 就罚
        if v < 70 or v > 140:
            return 1e2  # 极大正数（basinhopping 会主动远离）

        # 3. 原逻辑继续
        c1 = cloud_closure(x, y, z, t_0)
        time = validity_time(m1, target_true_pos, c1, t_0)
        if time == 0:
            return 1e2
        return -time


    class Optimization:
        def __init__(self):
            self.history = []
            self.params_history = []
            self.best_history = []
            self.best_params_history = []
            self.best_value = float('inf')
            self.best_params = None

        def __call__(self, x, f, accepted):
            self.history.append(f)
            self.params_history.append(x.copy())

            if f < self.best_value:
                self.best_value = f
                self.best_params = x.copy()

            self.best_history.append(self.best_value)
            self.best_params_history.append(self.best_params.copy())

            if len(self.history) % 100 == 0:
                print(f"Iteration: {len(self.history)}:Current value: {-f:.4f}, Best Value: {-self.best_value:.4f}")


    bounds = [(10, 20), (-10, 10), (0, 20), (0, 70)]
    initial_params = np.array([17.188, 0, 17.36496, 5.1])
    tracker = Optimization()

    print("开始模拟退火...")
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "options": {"maxiter": 100}
    }

    result_sa = basinhopping(
        objective,
        initial_params,
        niter=1000,
        minimizer_kwargs=minimizer_kwargs,
        stepsize=0.5,
        accept_test=None,
        callback=tracker,

    )

    best_params_sa = result_sa.x
    best_value_sa = -result_sa.fun

    print("\n 模拟退火优化结果")
    print(f"最佳坐标： ({best_params_sa[0] * 1000}, {best_params_sa[1]}, {best_params_sa[2]*100})")
    print(f"最佳引爆时间： {best_params_sa[3]}")
    print(f"最大有效遮蔽时间： {best_value_sa}")

    x = best_params_sa[0] * 1000
    y = best_params_sa[1]
    z = best_params_sa[2] * 100
    c = np.array([x, y, z])
    fy1 = np.array([17800, 0, 1800])
    t_z = fy1[2] - z
    t_0 = best_params_sa[3]
    t_1 = np.sqrt(2 * t_z / 9.8)
    t_2 = t_0 - t_1
    c_ = np.array([c[0], c[1], 1800])
    fy1_to_c = c_ - fy1
    fy1_to_c_len = np.linalg.norm(fy1_to_c)
    v = fy1_to_c_len / t_0
    fy1_to_c_n = fy1_to_c / fy1_to_c_len
    toudandian = fy1 + fy1_to_c_n * t_2 * v
    if (v > 140 or v < 70):
        print("FT1来不及赶到投弹点！！！")
    print("FY1的飞行方向的单位向量为：", fy1_to_c_n)
    print("FY1的飞行速度为：", v)
    print("FY1的投弹点为：", toudandian)
    print("FY1的起爆点为：", c)

    x_norm = [(p[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) for p in tracker.params_history]
    y_norm = [(p[1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) for p in tracker.params_history]
    z_norm = [(p[2] - bounds[2][0]) / (bounds[2][1] - bounds[2][0]) for p in tracker.params_history]
    t_norm = [(p[3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0]) for p in tracker.params_history]
    plt.plot(x_norm, label='x')
    plt.plot(y_norm, label='y')
    plt.plot(z_norm, label='z')
    plt.plot(t_norm, label='t_det')
    plt.legend()
    plt.xlabel('迭代次数')
    plt.ylabel('归一化参数值')
    plt.title('参数优化过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
    best_time = 0
    if (best_value_sa > best_time):
        best_time = best_value_sa
        best_seed = rand_seed
    if best_time > 4.7:
        a = 0
