from itertools import combinations
from missile_search import validity_time, validity_time_set
from matplotlib import pyplot as plt
from scipy.optimize import basinhopping, minimize
import random
from terms_P import *
import time
import numpy as np
from Q4_2 import *
# 最简化的字体设置，确保兼容性
plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 基础样式设置
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 80
plt.rcParams["font.size"] = 12

rand_seed = random.randint(0, 2 ** 32 - 1)
print('本次随机种子 =', rand_seed)
np.random.seed(rand_seed)
start = time.perf_counter()

fy1_pos = np.array([11000, 2000, 1800])
fy1_init_v_list_b1 = []
fy1_init_p_list_b1 = []
fy1_init_t_list_b1 = []
fy1_best_time_b1 = -1
fy1_best_v_b1 = None
fy1_best_p_b1 = None
fy1_best_t_b1 = None
fy1_best_time_b2 = -1
fy1_best_v_b2 = None
fy1_best_p_b2 = None
fy1_best_t_b2 = None
fy1_best_time_b3 = -1
fy1_best_v_b3 = None
fy1_best_p_b3 = None
fy1_best_t_b3 = None

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

def sliding_dwell_init(m_pos, fy_pos, corridor):
    m_pos_ = m_pos
    fy_pos_ = fy_pos
    corridor_ = corridor
    def sliding_dwell(x):
        """
            返回无人机在滑动走廊 [300,1400] m 内的总停留时长（秒）
            全程 torch 可微，theta 单位 rad，v 单位 m/s
        """
        theta, v = x
        v = v*10
        dir = np.array([np.cos(theta), np.sin(theta)])
        dwell = 0.
        t = 0.
        dt = 0.01
        m_pos = np.array([m_pos_[0], m_pos_[1]])
        UM = -m_pos / np.linalg.norm(-m_pos)
        while t <= 70:
            p = np.array([fy_pos_[0], fy_pos_[1]]) + v * t * dir  # 无人机水平位置
            m = np.array([m_pos[0], m_pos[1]]) + 300 * t * UM  # 导弹水平位置
            xi = np.dot(p - m, UM)  # 导弹坐标系下的前向距离
            inside = (xi >= corridor_[0]) & (xi <= corridor_[1])
            dwell = dwell + (dt if inside else 0.)
            t += dt
            # print(t, xi.item(), inside.item())
        return -dwell
    return sliding_dwell

objective_1 = sliding_dwell_init(m2_pos, fy1_pos, [0, 7000])
# step_size = ConstrainedRandomDisplacement(stepsize=0.5)
tracker_1 = Optimization()
bounds_1 = [(0, 2*np.pi), (70/10, 140/10)]
initial_params_1 = np.array([np.pi, 80/10])
print("开始模拟退火...")
print("开始第一步退火，查找最佳航向角theta和速度v")
minimizer_kwargs_1 = {
    "method": "L-BFGS-B",
    "bounds": bounds_1,
    "options": {"maxiter": 100}
}

result_sa1 = basinhopping(
    objective_1,
    initial_params_1,
    niter=500,
    minimizer_kwargs=minimizer_kwargs_1,
    stepsize = 0.5,
    accept_test=None,
    callback=tracker_1,
)
best_params_sa = result_sa1.x
best_value_sa = -result_sa1.fun
best_params_sa[0] = np.pi
# best_params_sa[1] = 14
best_value_sa = -objective_1(best_params_sa)
print(f"最佳航线角：{best_params_sa[0]}")
print(f"最佳速度：{best_params_sa[1]*10}")
print(f"最长潜在遮蔽能力：{best_value_sa:.4f}")
avo = best_params_sa[0]
v = best_params_sa[1]*10
def grid_guess(theta, v, fy_pos,
               rt_rng=(0.0, 4.0, 0.5),
               bt_rng=(0.0, 6.0, 0.5),
               min_total=0.01):
    """
    暴力网格扫一遍三弹 rt*/bt*，返回第一个总遮蔽 > min_total 的 6 维向量
    若无解返回 None
    """
    rt1s = np.arange(*rt_rng)
    rt2s = np.arange(*rt_rng)
    rt3s = np.arange(*rt_rng)
    bt1s = np.arange(*bt_rng)
    bt2s = np.arange(*bt_rng)
    bt3s = np.arange(*bt_rng)

    obj_func = bomb6_bh_obj_init_user(theta, v, fy_pos)   # 复用你已有的闭包

    # 四层循环——先扫 rt1,bt1，再 rt2,bt2，再 rt3,bt3
    for rt1 in rt1s:
        for bt1 in bt1s:
            for rt2 in rt2s:
                for bt2 in bt2s:
                    for rt3 in rt3s:
                        for bt3 in bt3s:
                            # 保证时序单调
                            # print(rt1, rt2,rt3)
                            if rt2 < rt1 +1 or rt3 < rt2+rt1:
                                continue
                            x = np.array([rt1, rt2, rt3, bt1, bt2, bt3])
                            f, mask_time1, mask_time2, mask_time3 = obj_func(x)
                            if mask_time1> 0 and mask_time2 > 0 and mask_time3 > 0 \
                                    and -f > min_total:
                                return x
    print('网格未找到有效初值')
    return None

def fast_3mask_grid(theta, v, fy_pos,
                    rt_range=(0, 4, 0.2),
                    bt_range=(0, 6, 0.3),
                    min_single=0.1,      # 单弹至少遮 0.01 s
                    min_total=0.3):       # 总遮 0.03 s
    """
    返回第一个 [rt1,rt2,rt3,bt1,bt2,bt3] 满足
    每弹遮蔽>0 且 rt1<rt2<rt3 且间隔≥1 s
    """
    obj = bomb2_bh_obj_init_user(theta, v, fy_pos)  # 要返回4元组
    rt_step, bt_step = rt_range[2], bt_range[2]
    rt_axis = np.arange(*rt_range)
    bt_axis = np.arange(*bt_range)

    # ---------- ① 单弹预筛 ----------
    # 只存 (rt, bt) 二元组，遮蔽>0
    tbl1, tbl2, tbl3 = [], [], []
    for rt in rt_axis:
        boolunjump = True
        Nmx = 0
        tab = False
        for bt in bt_axis:
            m1 = 0
            if Nmx == 3:
                break
            # 测试第一弹
            if boolunjump:
                m1 = obj(np.array([rt, bt]))  # 只测第一弹
            if m1<0.5:
                m1 = 0
            if m1:
                tab = True
                # print(f"投弹时间{rt:.2f}爆炸时间{bt:.2f}有效遮挡{m1:.2f},m1:{Nmx}")
            if tab == True and m1 == False:
                Nmx += 1
            if m1 > min_single:
                tbl1.append((rt, bt, m1))

    print("————————————————————————————————")
    if not tbl1:
        return None
    # ---------- ② 有序化 ----------
    tbl1.sort(key=lambda x: x[0])
    print("hello")
    # print(tbl1)

    # 生成所有可能的三个点的组合
    max_coverage = 0
    best_combination = None
    for comb in combinations(tbl1, 3):
        rt1, bt1, m1 = comb[0]
        rt2, bt2, m2 = comb[1]
        rt3, bt3, m3 = comb[2]
        # 检查时间间隔是否满足要求
        if rt2 - rt1 >= 1 and rt3 - rt2 >= 1:
            total_coverage = m1 + m2 + m3
            if total_coverage > max_coverage:
                max_coverage = total_coverage
                best_combination = comb
                # print(comb)
    return max_coverage, best_combination
def bomb6_bh_obj_init(theta, v, fy_pos):
    _fy_pos = fy_pos
    _theta = theta
    _v = v
    def bomb6_bh_obj(x):
        rt1, rt2, rt3, bt1, bt2, bt3 = x
        release_pos_1 = np.array([fy_pos[0] + rt1 * np.cos(_theta) * _v, fy_pos[1] + rt1 * np.sin(_theta) * _v, fy_pos[2]])
        impact_pos_1 = np.array([release_pos_1[0]+ bt1 * np.cos(_theta) * _v, release_pos_1[1] + bt1 * np.sin(_theta) * _v, release_pos_1[2] - 0.5*9.8*bt1**2])
        c1 = cloud_closure(impact_pos_1[0], impact_pos_1[1], impact_pos_1[2], rt1+bt1)
        mask_time1 = validity_time_set(m2, target_true_pos, c1, rt1+bt1)
        # print("第一爆点遮蔽时长:", len(mask_time1)/10, "投弹时间：", rt1, "引爆时间", bt1)
        # total_mask_time1 = _total_mask_time | mask_time1
        # print(total_mask_time1)
        release_pos_2 = np.array([release_pos_1[0] + rt2 * np.cos(_theta) * _v, release_pos_1[1] + rt2 * np.sin(_theta) * _v, release_pos_1[2]])
        impact_pos_2 = np.array([release_pos_2[0] + bt2 * np.cos(_theta) * _v, release_pos_2[1] + bt2 * np.sin(_theta) * _v,release_pos_2[2] - 0.5 * 9.8 * bt2 ** 2])
        c2 = cloud_closure(impact_pos_2[0], impact_pos_2[1], impact_pos_2[2],rt1+ rt2 + bt2)
        mask_time2 = validity_time_set(m2, target_true_pos, c2, rt1+rt2 + bt2)
        # print("第二爆点遮蔽时长:", len(mask_time2)/10, "投弹时间：", rt1+rt2, "引爆时间", bt2)
        # total_mask_time2 = total_mask_time1 | mask_time2
        # print(total_mask_time2)
        release_pos_3 = np.array([release_pos_2[0] + rt3 * np.cos(_theta) * _v, release_pos_2[1] + rt3 * np.sin(_theta) * _v, release_pos_2[2]])
        impact_pos_3 = np.array([release_pos_3[0] + bt3 * np.cos(_theta) * _v, release_pos_3[1] + bt3 * np.sin(_theta) * _v,release_pos_3[2] - 0.5 * 9.8 * bt3 ** 2])
        c3 = cloud_closure(impact_pos_3[0], impact_pos_3[1], impact_pos_3[2], rt1+ rt2+rt3 + bt3)
        mask_time3 = validity_time_set(m2, target_true_pos, c3, rt1+ rt2+rt3 + bt3)
        # print("第三爆点遮蔽时长:", len(mask_time3)/10, "投弹时间：", rt1+rt2+rt3, "引爆时间", bt3)
        # total_mask_time3 = total_mask_time2 | mask_time3
        # print(total_mask_time3)
        total_mask_time = mask_time1 | mask_time2 | mask_time3
        # print("总遮蔽时长", len(total_mask_time)/10)
        return -len(total_mask_time)/10
    return bomb6_bh_obj
def bomb6_bh_obj_init_user(theta, v, fy_pos):
    _fy_pos = fy_pos
    _theta = theta
    _v = v
    _total_mask_time = set()
    def bomb6_bh_obj(x):
        rt1, rt2, rt3, bt1, bt2, bt3 = x
        release_pos_1 = np.array([fy_pos[0] + rt1 * np.cos(_theta) * _v, fy_pos[1] + rt1 * np.sin(_theta) * _v, fy_pos[2]])
        impact_pos_1 = np.array([release_pos_1[0]+ bt1 * np.cos(_theta) * _v, release_pos_1[1] + bt1 * np.sin(_theta) * _v, release_pos_1[2] - 0.5*9.8*bt1**2])
        c1 = cloud_closure(impact_pos_1[0], impact_pos_1[1], impact_pos_1[2], rt1+bt1)
        mask_time1 = validity_time_set(m2, target_true_pos, c1, rt1+bt1)
        # print("第一爆点遮蔽时长:", len(mask_time1)/100, "投弹时间：", rt1, "引爆时间", bt1)
        total_mask_time1 = _total_mask_time | mask_time1
        release_pos_2 = np.array([release_pos_1[0] + rt2 * np.cos(_theta) * _v, release_pos_1[1] + rt2 * np.sin(_theta) * _v, release_pos_1[2]])
        impact_pos_2 = np.array([release_pos_2[0] + bt2 * np.cos(_theta) * _v, release_pos_2[1] + bt2 * np.sin(_theta) * _v,release_pos_2[2] - 0.5 * 9.8 * bt2 ** 2])
        c2 = cloud_closure(impact_pos_2[0], impact_pos_2[1], impact_pos_2[2],rt1+ rt2 + bt2)
        mask_time2 = validity_time_set(m2, target_true_pos, c2, rt1+rt2 + bt2)
        # print("第二爆点遮蔽时长:", len(mask_time2)/100, "投弹时间：", rt1+rt2, "引爆时间", bt2)
        total_mask_time2 = total_mask_time1 | mask_time2
        release_pos_3 = np.array([release_pos_2[0] + rt3 * np.cos(_theta) * _v, release_pos_2[1] + rt3 * np.sin(_theta) * _v, release_pos_2[2]])
        impact_pos_3 = np.array([release_pos_3[0] + bt3 * np.cos(_theta) * _v, release_pos_3[1] + bt3 * np.sin(_theta) * _v,release_pos_3[2] - 0.5 * 9.8 * bt3 ** 2])
        c3 = cloud_closure(impact_pos_3[0], impact_pos_3[1], impact_pos_3[2], rt1+ rt2+rt3 + bt3)
        mask_time3 = validity_time_set(m2, target_true_pos, c3, rt1+ rt2+rt3 + bt3)
        # print("第三爆点遮蔽时长:", len(mask_time3)/100, "投弹时间：", rt1+rt2+rt3, "引爆时间", bt3)
        total_mask_time3 = total_mask_time2 | mask_time3
        # print("总遮蔽时长", len(total_mask_time3)/10)
        return -len(total_mask_time3)/10, len(mask_time1)/10, len(mask_time2)/10, len(mask_time3)/10
    return bomb6_bh_obj
def bomb2_bh_obj_init_user(theta, v, fy_pos):
    _fy_pos = fy_pos
    _theta = theta
    _v = v
    _total_mask_time = set()
    def bomb2_bh_obj(x):
        rt1, bt1= x
        release_pos_1 = np.array([fy_pos[0] + rt1 * np.cos(_theta) * _v, fy_pos[1] + rt1 * np.sin(_theta) * _v, fy_pos[2]])
        impact_pos_1 = np.array([release_pos_1[0]+ bt1 * np.cos(_theta) * _v, release_pos_1[1] + bt1 * np.sin(_theta) * _v, release_pos_1[2] - 0.5*9.8*bt1**2])
        c1 = cloud_closure(impact_pos_1[0], impact_pos_1[1], impact_pos_1[2], rt1+bt1)
        # print("搜索功能debug：：：：", m1, target_true_pos, c1, rt1+bt1)
        mask_time1 = validity_time_set(m2, target_true_pos, c1, rt1+bt1)
        # print("搜索功能返回数", mask_time1)
        # print("第一爆点遮蔽时长:", len(mask_time1)/100, "投弹时间：", rt1, "引爆时间", bt1)
        return len(mask_time1)/10
    return bomb2_bh_obj
# def bomb6_bh_obj_init(theta, v, fy_pos):
#     _fy_pos = fy_pos
#     _theta = theta
#     _v = v
#     _total_mask_time = set()
#     def bomb6_bh_obj(x):
#         rt1, rt2, rt3, bt1, bt2, bt3 = x
#         release_pos_1 = np.array([fy_pos[0] + rt1 * np.cos(_theta) * _v, fy_pos[1] + rt1 * np.sin(_theta) * _v, fy_pos[2]])
#         impact_pos_1 = np.array([release_pos_1[0]+ bt1 * np.cos(_theta) * _v, release_pos_1[1] + bt1 * np.sin(_theta) * _v, release_pos_1[2] - 0.5*9.8*bt1**2])
#         c1 = cloud_closure(impact_pos_1[0], impact_pos_1[1], impact_pos_1[2], rt1+bt1)
#         mask_time1 = validity_time_set(m1, target_true_pos, c1, rt1+bt1)
#         print("第一爆点遮蔽时长:", len(mask_time1)/100)
#         total_mask_time1 = _total_mask_time | mask_time1
#         release_pos_2 = np.array([fy_pos[0] + rt2 * np.cos(_theta) * _v, fy_pos[1] + rt2 * np.sin(_theta) * _v, fy_pos[2]])
#         impact_pos_2 = np.array([release_pos_2[0] + bt2 * np.cos(_theta) * _v, release_pos_2[1] + bt2 * np.sin(_theta) * _v,release_pos_2[2] - 0.5 * 9.8 * bt2 ** 2])
#         c2 = cloud_closure(impact_pos_2[0], impact_pos_2[1], impact_pos_2[2], rt2 + bt2)
#         mask_time2 = validity_time_set(m1, target_true_pos, c2, rt2 + bt2)
#         print("第二爆点遮蔽时长:", len(mask_time2)/100)
#         total_mask_time2 = total_mask_time1 | mask_time2
#         release_pos_3 = np.array([fy_pos[0] + rt3 * np.cos(_theta) * _v, fy_pos[1] + rt3 * np.sin(_theta) * _v, fy_pos[2]])
#         impact_pos_3 = np.array([release_pos_3[0] + bt3 * np.cos(_theta) * _v, release_pos_3[1] + bt3 * np.sin(_theta) * _v,release_pos_3[2] - 0.5 * 9.8 * bt3 ** 2])
#         c3 = cloud_closure(impact_pos_3[0], impact_pos_3[1], impact_pos_3[2], rt3 + bt3)
#         mask_time3 = validity_time_set(m1, target_true_pos, c3, rt3 + bt3)
#         print("第三爆点遮蔽时长:", len(mask_time3)/100)
#         total_mask_time3 = total_mask_time2 | mask_time3
#         print("总遮蔽时长", len(total_mask_time3)/100)
#         return -len(total_mask_time3)/100
#     return bomb6_bh_obj

# 1. 先跑完第一段，拿到
#    best_params_sa[0] = theta
#    best_params_sa[1]*10 = v
# 2. 调用下面函数
good_x0 = fast_3mask_grid(
        theta   = best_params_sa[0],
        v       = best_params_sa[1]*10,
        fy_pos  = fy1_pos,
        rt_range=(0.0, 7, 0.1),   # 步长可再调细
        bt_range=(0.0, 8, 0.1),
        min_single=0.01,   # 单弹至少 0.01 s
        min_total=0.03)    # 总遮 0.03 s
print('网格找到有效初值:', good_x0)


# 3. 用 good_x0 代替原来的 initial_params_2 即可
comb = good_x0[1] if good_x0 is not None else np.array([[0.1, 0, 0], [4.8, 1.3, 0], [6.9, 2.2, 0]])
print(comb)
rt1, bt1, _ = comb[0]
rt2, bt2, _ = comb[1]
rt3, bt3, _ = comb[2]
initial_params_2 = np.array([rt1, rt2-rt1, rt3-rt2, bt1, bt2, bt3])
print("————————————————————————————————————")
print("开始模拟退火...")
print("开始第二步退火，查找三个烟雾弹的最佳爆点，以及最大遮蔽时长")
objective_2 = bomb6_bh_obj_init(best_params_sa[0], best_params_sa[1]*10.0, fy1_pos)
bounds_2 = [(0, 4), (1, 4), (1, 4), (0, 6), (0, 6), (0, 6)]
minimizer_kwargs_2 = {
    "method": "L-BFGS-B",
    "bounds": bounds_2,
    "options": {"maxiter": 100}
}
tracker_2 = Optimization()
# take_step = ConstrainedRandomDisplacement(stepsize=0.5, bounds=bounds_2)
result_sa = basinhopping(
    objective_2,
    initial_params_2,
    niter=2000,
    minimizer_kwargs=minimizer_kwargs_2,
    stepsize = 0.2,
    accept_test=None,
    callback=tracker_2,
)

init_as = [
    [fy1_pos, fy1_best_v, fy1_best_p, fy1_best_t],
    [fy2_pos, fy2_best_v, fy2_best_p, fy2_best_t],
    [fy3_pos, fy3_best_v, fy3_best_p, fy3_best_t]
]
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
def objective_user(params):
    a, v, t_release, t_detonate = params
    t_release = t_release
    t_detonate = t_detonate
    a = a
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
    pos_release = np.array([17800, 0, 1800])  # 复制飞机初始位置
    pos_release[0] += delta_xy[0]  # x
    pos_release[1] += delta_xy[1]  # y
    pos_detonate = reverse_projectile_point(np.array([17800, 0, 1800]), t_detonate, t_release, a, v)
    # print("爆点坐标有z",pos_detonate[0] , pos_detonate[1], pos_detonate[2],t_release + t_detonate)
    c = cloud_closure(pos_detonate[0], pos_detonate[1], pos_detonate[2], t_release + t_detonate)
    time = validity_time(m1, target_true_pos, c, t_release + t_detonate)
    return pos_release, pos_detonate, time
def objective_user_111(params):
    a, v, t_release, t_detonate = params
    # 水平方向单位向量
    dir_x = np.cos(a)
    dir_y = np.sin(a)

    # 水平位移
    d_xy = v * t_release
    # 投弹点
    x_release = 17800 + dir_x * d_xy
    y_release = 0 + dir_y * d_xy
    z_release = 1800  # 高度不变
    # 总飞行时间
    t_total = t_release + t_detonate

    # 水平总位移
    d_total_xy = v * t_total

    # 爆点水平坐标
    x_det = 17800 + dir_x * d_total_xy
    y_det = 0 + dir_y * d_total_xy

    # 垂直自由落体
    g = 9.8
    z_det = 1800 - 0.5 * g * t_detonate ** 2
    casdas = cloud_closure(x_det, y_det, z_det, t_release + t_detonate)
    timeasasda = validity_time_set(m1, target_true_pos, casdas, t_release + t_detonate)
    return np.array([x_release, y_release, z_release]), np.array([x_det, y_det, z_det]), timeasasda


best_params_sa = result_sa.x
rt1, rt2, rt3, bt1, bt2, bt3 = best_params_sa
best_value_sa = -result_sa.fun
print(f"最佳投弹时间：{best_params_sa[0], best_params_sa[1], best_params_sa[2]}")
print(f"最佳爆炸时间：{best_params_sa[3], best_params_sa[4], best_params_sa[5]}")
print(f"最大遮蔽时长：{best_value_sa:.4f}")
end = time.perf_counter()
print(f"本次耗时：{end - start:.6f} 秒")
bom1 = np.array([avo, v ,rt1, bt1])
bom2 = np.array([avo, v ,rt1+rt2, bt2])
bom3 = np.array([avo, v ,rt1+rt2+rt3, bt3])
print(bom1)
pos_release, pos_detonate, time11 = objective_user(bom1)

print("————————————————————————————————————")
print(bom1)
print(bom2)
print(bom3)
re1, de1, ti1 = objective_user_111(bom1)
re2, de2, ti2 = objective_user_111(bom2)
re3, de3, ti3 = objective_user_111(bom3)
print(re1, de1, len(ti1))
print(re2, de2, len(ti2))
print(re3, de3, len(ti3))
print("ti1 时段数:", len(ti1))
print("ti2 时段数:", len(ti2))
print("ti3 时段数:", len(ti3))
print("并集时段数:", len(ti1 | ti2 | ti3))