from core import *
from missile_search import validity_time
from matplotlib import pyplot as plt
from scipy.optimize import basinhopping, minimize
import random
import numpy as np
from matplotlib.ticker import MaxNLocator
# 最简化的字体设置，确保兼容性
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 基础样式设置
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 80
plt.rcParams["font.size"] = 12

def objective(params):
    a, v, t_release, t_detonate = params
    t_release = t_release / 10
    t_detonate = t_detonate / 10
    a = a * np.pi/6/10
    a = angle_to_unit_vector(a)
    v = v
    g = 9.8
    fy1 = np.array([17800, 0, 1800])
    pos_release = fy1 + a * v * t_release
    # print(pos_release)
    pos_detonate = fy1 + a * v * (t_detonate+t_release)
    # print("爆点坐标无z",pos_detonate, "v", v, "t", t_release+t_detonate, "a", a)
    pos_detonate[2] = 1800 - 0.5 * g * t_detonate ** 2
    c = cloud_closure(pos_detonate[0], pos_detonate[1], pos_detonate[2], t_release + t_detonate)
    # print("爆点坐标有z",pos_detonate[0] , pos_detonate[1], pos_detonate[2],t_release + t_detonate)
    time = validity_time(m1, target_true_pos, c, t_release + t_detonate)
    return -time * 100

def objective_user(params):
    a, v, t_release, t_detonate = params
    t_release = t_release / 10
    t_detonate = t_detonate / 10
    a = a * np.pi/6/10
    v = v
    g = 9.8
    fy1 = np.array([17800, 0, 1800])
    pos_release = fy1 + a * v * t_release
    pos_detonate = fy1 + a * v * t_detonate
    pos_detonate[2] = 1800 - 0.5 * g * t_detonate ** 2
    c = cloud_closure(pos_detonate[0], pos_detonate[1], pos_detonate[2], t_release + t_detonate)
    time = validity_time(m1, target_true_pos, c, t_release + t_detonate)
    return pos_release, pos_detonate, time

def angle_to_unit_vector(a, deg=False):
    if deg:
        a = np.deg2rad(a)

    # 基准向量：−x 轴
    base = np.array([-1.0, 0.0, 0.0])

    # 绕 z 轴旋转矩阵（右手系，顺时针即负角度）
    cos = np.cos(-a)
    sin = np.sin(-a)
    Rz = np.array([[cos, -sin, 0],
                   [sin, cos, 0],
                   [0, 0, 1]])

    e = Rz @ base  # 旋转后的向量
    return e / np.linalg.norm(e)

a = 1
while a:
    a -= 1
    rand_seed = random.randint(0, 2 ** 32 - 1)
    print('本次随机种子 =', rand_seed)
    np.random.seed(rand_seed)


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

            if len(self.history) % 1 == 0:
                print(f"Iteration: {len(self.history)}:Current value: {-f:.4f}, Best Value: {-self.best_value:.4f}{x}")


    bounds = [(-1*10, 1*10), (70, 140), (0, 50), (0, 50)]
    initial_params = np.array([0, 120, 15, 36])
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
        niter=50,
        minimizer_kwargs=minimizer_kwargs,
        stepsize=0.5,
        accept_test=None,
        callback=tracker,
    )

    best_params_sa = result_sa.x
    best_value_sa = -result_sa.fun/100
    pos_release, pos_detonate, time = objective_user(best_params_sa)
    M = np.linalg.norm(pos_detonate - m1(best_params_sa[3]/10))
    print("\n 模拟退火优化结果")
    print(f"最佳转向角：{best_params_sa[0]*np.pi/6/10}")
    print(f"最佳速度：{best_params_sa[1]}")
    print(f"最佳投弹时间：{best_params_sa[2]/10}")
    print(f"最佳投弹点：{pos_release}")
    print(f"最佳引爆时间：{best_params_sa[3]/10}")
    print(f"最佳引爆点：{pos_detonate}")
    print(f"最大有效遮蔽时间： {best_value_sa}")
    print(f"爆时烟雾与导弹距离：{M}")

    a_norm = [(p[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) for p in tracker.params_history]
    v_norm = [(p[1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) for p in tracker.params_history]
    t_release_norm = [(p[2] - bounds[2][0]) / (bounds[2][1] - bounds[2][0]) for p in tracker.params_history]
    t_detonate_norm = [(p[3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0]) for p in tracker.params_history]
    plt.plot(a_norm, label='角度')
    plt.plot(v_norm, label='速度')
    plt.plot(t_release_norm, label='投弹时间')
    plt.plot(t_detonate_norm, label='引爆时间')
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