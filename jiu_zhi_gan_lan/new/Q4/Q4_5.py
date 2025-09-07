from core import *
from missile_search import validity_time, validity_time_set
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

init_as = [
    [fy1_pos, fy1_best_v, fy1_best_p, fy1_best_t],
    [fy2_pos, fy2_best_v, fy2_best_p, fy2_best_t],
    [fy3_pos, fy3_best_v, fy3_best_p, fy3_best_t]
]
total_effective_shaded_area = set()
for i in range(0, 3):
    def objective(params):
        a, v, t_release, t_detonate = params
        # print(t_release, t_detonate)
        t_release = t_release / 10
        t_detonate = t_detonate / 10
        a = a / 10
        # a = a * np.pi/10
        # a = angle_to_unit_vector_as(a)
        v = v
        # pos_release = init_as[i][0] + a * v * t_release
        # print(pos_release, a)
        # # print("爆点坐标无z",pos_detonate, "v", v, "t", t_release+t_detonate, "a", a)
        # pos_detonate = init_as[i][0] + a * v * (t_detonate+t_release)
        # pos_detonate[2] = 1800 - 0.5 * g * t_detonate ** 2

        pos_detonate = reverse_projectile_point(init_as[i][0].copy(), t_detonate, t_release, a, v)
        # print("爆点坐标有z",pos_detonate[0] , pos_detonate[1], pos_detonate[2],t_release + t_detonate)
        c = cloud_closure(pos_detonate[0], pos_detonate[1], pos_detonate[2], t_release + t_detonate)
        time = validity_time(m1, target_true_pos, c, t_release + t_detonate)
        return -time * 100


    def objective_user(params):
        a, v, t_release, t_detonate = params
        a, v, t_release, t_detonate = params
        t_release = t_release / 10
        t_detonate = t_detonate / 10
        a = a / 10
        # a = a * np.pi/10
        # a = angle_to_unit_vector_as(a)
        v = v
        # pos_release = init_as[i][0] + a * v * t_release
        # print(pos_release, a)
        # # print("爆点坐标无z",pos_detonate, "v", v, "t", t_release+t_detonate, "a", a)
        # pos_detonate = init_as[i][0] + a * v * (t_detonate+t_release)
        # pos_detonate[2] = 1800 - 0.5 * g * t_detonate ** 2
        # 1. 把角度转成水平单位向量（二维）
        angle_rad = a  # 已经是弧度
        dir_vec = np.array([np.cos(angle_rad),
                            np.sin(angle_rad)])  # 二维方向

        # 2. 水平位移
        d_release = v * t_release  # 水平距离
        delta_xy = dir_vec * d_release  # 二维偏移量

        # 3. 投弹点坐标（假设飞机高度不变）
        pos_release = init_as[i][0].copy()  # 复制飞机初始位置
        pos_release[0] += delta_xy[0]  # x
        pos_release[1] += delta_xy[1]  # y
        pos_detonate = reverse_projectile_point(init_as[i][0].copy(), t_detonate, t_release, a, v)
        # print("爆点坐标有z",pos_detonate[0] , pos_detonate[1], pos_detonate[2],t_release + t_detonate)
        c = cloud_closure(pos_detonate[0], pos_detonate[1], pos_detonate[2], t_release + t_detonate)
        time = validity_time(m1, target_true_pos, c, t_release + t_detonate)
        return pos_release, pos_detonate, time
    def objective_user_set(params):
        a, v, t_release, t_detonate = params
        a, v, t_release, t_detonate = params
        t_release = t_release / 10
        t_detonate = t_detonate / 10
        a = a / 10
        # a = a * np.pi/10
        # a = angle_to_unit_vector_as(a)
        v = v
        # pos_release = init_as[i][0] + a * v * t_release
        # print(pos_release, a)
        # # print("爆点坐标无z",pos_detonate, "v", v, "t", t_release+t_detonate, "a", a)
        # pos_detonate = init_as[i][0] + a * v * (t_detonate+t_release)
        # pos_detonate[2] = 1800 - 0.5 * g * t_detonate ** 2
        pos_detonate = reverse_projectile_point(init_as[i][0].copy(), t_detonate, t_release, a, v)
        # print("爆点坐标有z",pos_detonate[0] , pos_detonate[1], pos_detonate[2],t_release + t_detonate)
        c = cloud_closure(pos_detonate[0], pos_detonate[1], pos_detonate[2], t_release + t_detonate)
        timeasd = validity_time_set(m1, target_true_pos, c, t_release + t_detonate)
        return timeasd

    def reverse_projectile_point(fy, t0, t1, angle_rad, v_xy):
        # print(fy, t0, t1, angle_rad, v_xy)
        g = 9.8
        total_time = t0 + t1
        h = 0.5 * g * t0 ** 2
        pz = fy[2] - h
        d = v_xy * total_time
        px = fy[0] + d * np.cos(angle_rad)
        py = fy[1] + d * np.sin(angle_rad)
        # print(py, fy[1], d, np.sin(angle_rad), angle_rad)
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
        # print(p_, fy_)
        fy_to_p = p_ - fy_
        angle_rad = np.arctan2(fy_to_p[1], fy_to_p[0])
        if angle_rad < 0:
            angle_rad += 2 * np.pi
        return angle_rad


    def flat_projectile_time(p, fy, t):
        h = fy[2] - p[2]
        t0 = np.sqrt(2 * h / 9.8)
        t1 = t - t0
        # print(t1, t0, t)
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

            if len(self.history) % 100 == 0:
                print(f"Iteration: {len(self.history)}:Current value: {-f:.4f}, Best Value: {-self.best_value:.4f}{x}")


    angle = angle_to_unit_vector(init_as[i][2].copy(), init_as[i][0].copy())
    t_release, t_detonate = flat_projectile_time(init_as[i][2].copy(), init_as[i][0].copy(), init_as[i][3].copy())
    # print([angle, init_as[i][1].copy(), t_release, t_detonate])
    bounds = [(0, 2 * np.pi * 10), (70, 140), (0, 500), (0, 50)]
    initial_params = np.array([angle * 10, init_as[i][1].copy(), t_release * 10, t_detonate * 10])
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
    best_value_sa = -result_sa.fun / 100
    pos_release, pos_detonate, time = objective_user(best_params_sa)
    M = np.linalg.norm(pos_detonate - m1(best_params_sa[3]/10))
    print("\n 模拟退火优化结果")
    print(f"最佳转向角：{best_params_sa[0]/ 10}")
    print(f"最佳速度：{best_params_sa[1]}")
    print(f"最佳投弹时间：{best_params_sa[2]/10}")
    print(f"最佳投弹点：{pos_release}")
    print(f"最佳引爆时间：{best_params_sa[3]/10}")
    print(f"最佳引爆点：{pos_detonate}")
    print(f"最大有效遮蔽时间： {best_value_sa}")
    print(f"爆时烟雾与导弹距离：{M}")
    ji = objective_user_set(best_params_sa)
    # print(ji)
    total_effective_shaded_area = total_effective_shaded_area | ji
print(len(total_effective_shaded_area)/100)
