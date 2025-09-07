from core import *
from missile_search import validity_time
from matplotlib import pyplot as plt
from scipy.optimize import basinhopping, minimize
import random
from Q4_2 import *
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
    t_release = t_release /10
    t_detonate = t_detonate /10
    a = a/10
    # a = a * np.pi/10
    # a = angle_to_unit_vector_as(a)
    v = v
    # pos_release = fy1_pos + a * v * t_release
    # print(pos_release, a)
    # # print("爆点坐标无z",pos_detonate, "v", v, "t", t_release+t_detonate, "a", a)
    # pos_detonate = fy1_pos + a * v * (t_detonate+t_release)
    # pos_detonate[2] = 1800 - 0.5 * g * t_detonate ** 2
    pos_detonate = reverse_projectile_point(fy1_pos, t_detonate, t_release, a, v)
    # print("爆点坐标有z",pos_detonate[0] , pos_detonate[1], pos_detonate[2],t_release + t_detonate)
    c = cloud_closure(pos_detonate[0], pos_detonate[1], pos_detonate[2], t_release + t_detonate)
    time = validity_time(m1, target_true_pos, c, t_release + t_detonate)
    return -time * 100

def objective_user(params):
    a, v, t_release, t_detonate = params
    t_release = t_release /10
    t_detonate = t_detonate /10
    a = a * np.pi/6/10
    v = v
    g = 9.8
    # fy1 = np.array([17800, 0, 1800])
    pos_release = fy1_pos + a * v * t_release
    pos_detonate = fy1_pos + a * v * t_detonate
    pos_detonate[2] = 1800 - 0.5 * g * t_detonate ** 2
    c = cloud_closure(pos_detonate[0], pos_detonate[1], pos_detonate[2], t_release + t_detonate)
    time = validity_time(m1, target_true_pos, c, t_release + t_detonate)
    return pos_release, pos_detonate, time
def reverse_projectile_point(fy, t0, t1, angle_rad, v_xy):
    g = 9.8
    total_time = t0 + t1
    h = 0.5 * g * t0 ** 2
    pz = fy[2] - h
    d = v_xy * total_time
    px = fy[0] + d * np.cos(angle_rad)
    py = fy[1] + d * np.sin(angle_rad)
    return np.array([px, py, pz])
def angle_to_unit_vector_as(a, deg=True):
    if deg:
        a = np.deg2rad(a)

    # 基准向量：x 轴
    base = np.array([1.0, 0.0, 0.0])

    # 绕 z 轴旋转矩阵（右手系，顺时针即负角度）
    cos = np.cos(-a)
    sin = np.sin(-a)
    Rz = np.array([[cos, -sin, 0],
                   [sin, cos, 0],
                   [0, 0, 1]])

    e = Rz @ base  # 旋转后的向量
    return e / np.linalg.norm(e)
def angle_to_unit_vector(p, fy):
    fy_ = np.array([fy[0], fy[1]])
    p_ = np.array([p[0], p[1]])
    fy_to_p = p_ - fy_
    angle_rad = np.arctan2(fy_to_p[1], fy_to_p[0])
    return angle_rad
def flat_projectile_time(p, fy, t):
    h = fy[2] - p[2]
    t0 = np.sqrt(2*h/9.8)
    t1 = t-t0
    return t1, t0
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

angle = angle_to_unit_vector(fy1_best_p, fy1_pos)
t_release, t_detonate = flat_projectile_time(fy1_best_p, fy1_pos, fy1_best_t)
print([angle, fy1_best_v, t_release, t_detonate])
bounds = [(0, 2*np.pi * 10), (70, 140), (0, 50), (0, 50)]
initial_params = np.array([angle*10, fy1_best_v, t_release*10, t_detonate*10])
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
    niter=100,
    minimizer_kwargs=minimizer_kwargs,
    stepsize=0.5,
    accept_test=None,
    callback=tracker,
)

best_params_sa = result_sa.x
best_value_sa = -result_sa.fun / 100
pos_release, pos_detonate, time = objective_user(best_params_sa)
M = np.linalg.norm(pos_detonate - m1(best_params_sa[3]))
print("\n 模拟退火优化结果")
print(f"最佳转向角：{best_params_sa[0] * np.pi / 6 / 10}")
print(f"最佳速度：{best_params_sa[1]}")
print(f"最佳投弹时间：{best_params_sa[2]}")
print(f"最佳投弹点：{pos_release}")
print(f"最佳引爆时间：{best_params_sa[3]}")
print(f"最佳引爆点：{pos_detonate}")
print(f"最大有效遮蔽时间： {best_value_sa}")
print(f"爆时烟雾与导弹距离：{M}")
plt.figure()
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

plt.figure()
# plt.subplot(1, 3, 3)
initial_value = -objective(initial_params) / 100
plt.bar(['初始参数', '优化后参数'], [initial_value, best_value_sa], alpha=0.7)
plt.ylabel('有效遮蔽时间(s)')
plt.ylim(bottom=initial_value-2)
plt.title('优化前后对比')
plt.grid(True, alpha=0.3)
for i, v, in enumerate([initial_value, best_value_sa]):
    plt.text(i, v +0.05, f'{v:.2f}s', ha='center', va='bottom')
plt.tight_layout()
plt.show()