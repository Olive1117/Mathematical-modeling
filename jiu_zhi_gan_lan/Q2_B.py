#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粗搜索（空列表保护）+ 三维距离退火细筛
单文件独立，直接运行
"""
import numpy as np
import pandas as pd
import os, random, time

# ---------------- 物理常数 -----------------
G             = 9.8
CLOUD_R       = 10
CLOUD_SINK    = 3.0
MISSILE_SPEED = 300
PLANE_ALT     = 1800
FY1_0         = np.array([17800, 0, PLANE_ALT])
TARGET        = np.array([0, 200, 0])
DT            = 0.1
T_MAX         = np.linalg.norm([20000, 0, 2000]) / MISSILE_SPEED  # ≈67 s

# ---------------- 工具函数 -----------------
def missile_pos(t):
    dir_vec = -np.array([20000, 0, 2000], dtype=float)
    dir_vec /= np.linalg.norm(dir_vec)
    return np.array([20000, 0, 2000]) + MISSILE_SPEED * t * dir_vec

def dist_point_to_line(p, a, b):
    ab, ap = b - a, p - a
    return np.linalg.norm(np.cross(ap, ab)) / (np.linalg.norm(ab) + 1e-12)

def shielding_duration(burst, t_burst):
    total = 0.0
    t = t_burst
    while t <= T_MAX:
        mt = missile_pos(t)
        cloud = burst - np.array([0, 0, CLOUD_SINK * (t - t_burst)])
        if dist_point_to_line(cloud, mt, TARGET) <= CLOUD_R:
            total += DT
        t += DT
    return total

def parse_v_theta(burst):
    dx, dy = burst[0] - FY1_0[0], burst[1] - FY1_0[1]
    dis_h  = np.hypot(dx, dy)
    t_b    = np.sqrt(2 * (PLANE_ALT - burst[2]) / G)
    if t_b < 0: return None, None, None, None
    T_min, T_max = dis_h / 140.0, dis_h / 70.0
    if T_max < t_b: return None, None, None, None
    T_opt = max(T_min, t_b)
    v_opt = dis_h / T_opt
    ux, uy = dx / dis_h, dy / dis_h
    drop = FY1_0 + np.array([ux, uy, 0]) * (T_opt - t_b) * v_opt
    theta = np.degrees(np.arctan2(uy, ux)) % 360
    return v_opt, theta, t_b, drop

# ---------------- 1. 粗搜索（空列表保护 + 每30次打印）-----------------
def coarse_screen(steps=21):
    axis = [np.linspace(-1000, 1000, steps),
            np.linspace(-500, 1000, steps),
            np.linspace(200, 1800, steps)]
    XX, YY, ZZ = np.meshgrid(*axis, indexing='ij')
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    scored = []
    n_tot = len(pts)
    print(">>> 粗搜索（空列表保护 + 每30次打印）...")
    for i, p in enumerate(pts):
        v, theta, tb, drop = parse_v_theta(p)
        if v is None: continue
        cov = shielding_duration(p, tb)
        scored.append((cov, *p, v, theta, tb))
        if (i + 1) % 30 == 0:
            top = scored[0][0] if scored else 0.0
            print(f"  已筛 {i+1:8d}/{n_tot}  当前TOP1 = {top:.2f}s")
    # 空列表保护
    if not scored:
        print("警告：粗搜索未筛到任何有效点，返回默认中心")
        scored = [(0.0, 18914, 6, 1895, 70.0, 0.29, 15.21)]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:10]

# ---------------- 2. 三维距离退火细筛（每30次迭代打印）-----------------
def anneal_fine(center, radius0=60.0, T0=40, Tf=0.05, alpha=0.93, inner=240):
    low_b, high_b = np.array([-1500, -1000, 200]), np.array([1500, 1000, 1800])
    S = center.copy()
    best_S, best_E = S.copy(), -1e6
    iter_cnt = 0

    def energy(pos):
        v, theta, tb, drop = parse_v_theta(pos)
        if v is None: return -1e6
        return shielding_duration(pos, tb)

    T = T0
    E_cur = energy(S)
    best_E = E_cur

    while T > Tf:
        for _ in range(inner):
            iter_cnt += 1
            if iter_cnt % 30 == 0:
                print(f"  iter {iter_cnt:5d}  T={T:.2f}  当前最优遮蔽 = {best_E:.2f}s")
            # 球内均匀采样，半径随温度衰减
            radius = radius0 * (T / T0)
            offset = np.random.normal(0, 1, 3)
            offset *= radius / np.linalg.norm(offset)
            S_new = S + offset
            S_new = np.clip(S_new, low_b, high_b)
            E_new = energy(S_new)
            delta_E = E_new - E_cur
            # 更好但未接受→仍以新解为中心继续搜索
            if delta_E > 0:
                if random.random() < np.exp(delta_E / T):
                    S, E_cur = S_new, E_new
                else:
                    S = S_new
                    E_cur = energy(S)
            else:
                if random.random() < np.exp(delta_E / T):
                    S, E_cur = S_new, E_new
            if E_cur > best_E:
                best_S, best_E = S.copy(), E_cur
        T *= alpha
    return best_S, best_E

# ---------------- 主入口 -----------------
if __name__ == "__main__":
    t0 = time.time()
    top10 = coarse_screen(steps=21)          # 21^3 ≈ 9e4 点
    print(f"粗搜索完成，TOP1 遮蔽 = {top10[0][0]:.2f}s  ({time.time()-t0:.1f}s)")

    center1 = np.array(top10[0][1:4])
    side1   = 60
    print(">>> 三维距离退火细筛（每30次迭代打印）...")
    t0 = time.time()
    best_burst, best_cov = anneal_fine(center1, radius0=60, T0=40, Tf=0.05, alpha=0.93, inner=240)
    print(f"退火完成，最优遮蔽 = {best_cov:.2f}s  ({time.time()-t0:.1f}s)")

    v_best, theta_best, tb_best, drop_best = parse_v_theta(best_burst)
    print("\n>>> 最终策略 <<<")
    print(f"起爆点: {best_burst}")
    print(f"速度方向: θ = {theta_best:.2f}°")
    print(f"速度大小: v = {v_best:.2f} m/s")
    print(f"起爆时间: t_b = {tb_best:.2f} s")
    print(f"最大遮蔽时长: {best_cov:.2f} s")

    os.makedirs("output", exist_ok=True)
    pd.DataFrame([{
        "x_b": best_burst[0], "y_b": best_burst[1], "z_b": best_burst[2],
        "theta°": theta_best, "v": v_best, "t_b": tb_best, "coverage": best_cov
    }]).to_excel("output/Q2_safe_coarse_anneal.xlsx", index=False)
    print(">>> 已保存 output/Q2_safe_coarse_anneal.xlsx")