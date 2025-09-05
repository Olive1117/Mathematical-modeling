#!/usr/bin/env python3
# -*- coding: utf- -*-
"""
烟幕干扰弹投放策略 – 密度云遮蔽时长最大化
A 题问题 2 示例（单弹单导弹）
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import os

# ---------------- 常量 -----------------
G              = 9.8                    # m/s^2
MISSILE_SPEED  = 300                    # m/s
CLOUD_SINK     = 3.0                    # m/s
CLOUD_R        = 10                     # m
PLANE_SPEED_MIN, PLANE_SPEED_MAX = 70, 140
PLANE_ALT      = 1800                   # FY1 初始高度
M0             = np.array([20000, 0, 2000])  # M1 初始
TARGET         = np.array([0, 200, 0])       # 真目标
FY1_0          = np.array([17800, 0, PLANE_ALT])  # FY1 初始

DT             = 0.1                    # 时间步长
T_MAX          = np.linalg.norm(M0) / MISSILE_SPEED  # 最大飞行时间

# ---------------- 工具函数 -----------------
def missile_pos(t: float) -> np.ndarray:
    """导弹直线飞向原点"""
    dir_vec = -M0.astype(float)
    dir_vec /= np.linalg.norm(dir_vec)
    return M0 + MISSILE_SPEED * t * dir_vec

def dist_point_to_line(p, a, b):
    """点 p 到线段 ab 的距离"""
    ab = b - a
    ap = p - a
    cross = np.cross(ap, ab)
    return np.linalg.norm(cross) / (np.linalg.norm(ab) + 1e-12)

def shielding_duration(burst: np.ndarray) -> float:
    """给定起爆点，返回对真目标的总遮蔽时长（秒）"""
    total = 0.0
    t = 0.0
    while t <= T_MAX:
        mt = missile_pos(t)
        cloud = burst - np.array([0, 0, CLOUD_SINK * t])
        if dist_point_to_line(cloud, mt, TARGET) <= CLOUD_R:
            total += DT
        t += DT
    return total

# ---------------- 空间网格 -----------------
def generate_grid(xrng, yrng, zrng):
    """生成三维网格点（N,3）"""
    xx, yy, zz = np.meshgrid(xrng, yrng, zrng, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

# 初筛：围绕导弹-目标走廊
COARSE = generate_grid(
    np.linspace(-600, 600, 25),
    np.linspace(-200, 600, 25),
    np.linspace(200, 1800, 30)
)

# ---------------- 并行评估 -----------------
print(">>> 粗网格评估中 ...")
coarse_dur = Parallel(n_jobs=-1, backend='threading')(
    delayed(shielding_duration)(p) for p in tqdm(COARSE, leave=False)
)
coarse_dur = np.array(coarse_dur)
top10_idx = np.argsort(coarse_dur)[-10:][::-1]
cand_points = COARSE[top10_idx]

# 细网格 refine：在 top-1 附近 100 m 内加密
best_coarse = cand_points[0]
refine_rng = np.linspace(-50, 50, 11) + best_coarse[:, None]
FINE = generate_grid(
    refine_rng[0], refine_rng[1], refine_rng[2]
)

print(">>> 细网格 refine ...")
fine_dur = Parallel(n_jobs=-1, backend='threading')(
    delayed(shielding_duration)(p) for p in tqdm(FINE, leave=False)
)
best_idx = np.argmax(fine_dur)
BEST_BURST = FINE[best_idx]
BEST_DUR = fine_dur[best_idx]

print(f"最优遮蔽时长：{BEST_DUR:.2f} s")
print(f"最优起爆点：{BEST_BURST}")

# ---------------- 反推无人机轨迹 -----------------
def solve_plane_param(burst: np.ndarray):
    """返回 (vx, vy, t_fly, t_b, deploy_point)"""
    # 1. 起爆时间由自由落体决定
    delta_z = PLANE_ALT - burst[2]
    if delta_z < 0:
        return None
    t_b = np.sqrt(2 * delta_z / G)

    # 2. 水平位移
    dx, dy = burst[0] - FY1_0[0], burst[1] - FY1_0[1]

    # 3. 无人机飞行时间 = 投放前飞行时间
    #    投放点 D = FY1_0 + v * t_fly
    #    起爆点 = D + v * t_b - 0.5*g*t_b^2 * z_hat
    # => 水平：burst[0:2] = D[0:2] + v[0:2]*t_b
    #         D[0:2] = FY1_0[0:2] + v[0:2]*t_fly
    # =>  burst[0:2] = FY1_0[0:2] + v*(t_fly + t_b)
    # =>  v = (burst[0:2] - FY1_0[0:2]) / (t_fly + t_b)
    # 但 t_fly 未知，我们只需 |v| ∈ [70,140]
    # 令 T = t_fly + t_b → v = dis / T
    dis = np.linalg.norm([dx, dy])
    T_min = dis / PLANE_SPEED_MAX
    T_max = dis / PLANE_SPEED_MIN
    # 任取 T ∈ [T_min, T_max] 即可，这里取中点
    T = (T_min + T_max) / 2
    v_vec = np.array([dx, dy]) / T
    v_mag = np.linalg.norm(v_vec)
    t_fly = T - t_b
    deploy = FY1_0 + np.array([v_vec[0], v_vec[1], 0]) * t_fly
    return (*v_vec, t_fly, t_b, deploy)

sol = solve_plane_param(BEST_BURST)
if sol is None:
    raise ValueError("起爆点高于飞机高度！")
vx, vy, t_fly, t_b, deploy = sol

print(f"无人机速度：(vx={vx:.2f}, vy={vy:.2f}) |v|={np.hypot(vx,vy):.2f} m/s")
print(f"飞行时间：{t_fly:.2f} s，投放点：{deploy}")

# ---------------- 输出 result1.xlsx -----------------
out = pd.DataFrame([{
    "无人机编号": "FY1",
    "导弹编号": "M1",
    "飞行速度(m/s)": np.hypot(vx, vy),
    "飞行方向(°)": np.degrees(np.arctan2(vy, vx)) % 360,
    "投放点X": deploy[0],
    "投放点Y": deploy[1],
    "投放点Z": deploy[2],
    "起爆点X": BEST_BURST[0],
    "起爆点Y": BEST_BURST[1],
    "起爆点Z": BEST_BURST[2],
    "起爆时间(s)": t_b,
    "有效遮蔽时长(s)": BEST_DUR
}])
os.makedirs("output", exist_ok=True)
out.to_excel("output/result1.xlsx", index=False)
print(">>> 已保存 output/result1.xlsx")