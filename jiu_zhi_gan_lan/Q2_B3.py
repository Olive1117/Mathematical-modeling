#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完善模型：云团膨胀+导弹圆柱 | 连线1D-SA | 多策略退火
"""
import numpy as np, pandas as pd, os, random, time, matplotlib.pyplot as plt, matplotlib.font_manager as fm

# ---------------- 物理常数 -----------------
G = 9.8
R0, k = 10.0, 2.0          # 云团初始半径+膨胀率
r_m = 1.5                  # 导弹圆柱半径
MISSILE_SPEED = 300
PLANE_ALT = 1800
FY1_0 = np.array([17800, 0, PLANE_ALT])
TARGET = np.array([0, 200, 0])
MISSILE_INIT = np.array([20000, 0, 2000])
DT = 0.05
MIN_DRONE_SPEED, MAX_DRONE_SPEED = 70, 140
T_MAX = np.linalg.norm(MISSILE_INIT - TARGET) / MISSILE_SPEED

# 导弹方向常数
u_m = (TARGET - MISSILE_INIT) / np.linalg.norm(TARGET - MISSILE_INIT)
M0 = MISSILE_INIT
lambda_max = np.linalg.norm(TARGET - M0)
Delta_t_const = (R0 + r_m) / (MISSILE_SPEED + k)   # 基础相交半窗口

# ---------------- 解析函数 -----------------
def missile_pos(t):
    return M0 + u_m * MISSILE_SPEED * t

def calc_t_burst(lam, dz=0):
    """解析最佳起爆时间"""
    P = M0 + lam * u_m + np.array([0, 0, dz])
    delta_z = PLANE_ALT - P[2]
    if delta_z <= 0:
        return np.inf
    t_fall = np.sqrt(2 * delta_z / G)
    dis_h = np.hypot(P[0] - FY1_0[0], P[1] - FY1_0[1])
    t_drone = dis_h / MAX_DRONE_SPEED
    return t_fall + t_drone + 0.2

def cover_1d(lam, t_burst, dz=0):
    """1D遮蔽时长：解析窗口"""
    t_center = lam / MISSILE_SPEED
    Delta_t = Delta_t_const + k * dz / (MISSILE_SPEED * (MISSILE_SPEED + k))
    t_enter = max(t_burst, t_center - Delta_t)
    t_exit  = min(T_MAX, t_center + Delta_t)
    return max(0., t_exit - t_enter)

def energy_1d(lam, dz=0):
    """1D能量函数"""
    if lam < 0 or lam > lambda_max:
        return -np.inf, 0, 0, False
    t_burst = calc_t_burst(lam, dz)
    if t_burst > T_MAX:
        return -np.inf, 0, 0, False
    cover = cover_1d(lam, t_burst, dz)
    # 速度约束
    P = M0 + lam * u_m + np.array([0, 0, dz])
    dis_h = np.hypot(P[0] - FY1_0[0], P[1] - FY1_0[1])
    t_fall = np.sqrt(2 * (PLANE_ALT - P[2]) / G)
    need_v = dis_h / (t_burst - t_fall)
    valid = MIN_DRONE_SPEED <= need_v <= MAX_DRONE_SPEED
    return (cover, t_burst, need_v, valid) if valid else (-abs(need_v), t_burst, need_v, False)

# ---------------- 多策略1D-SA -----------------
def sa_multistrategy(n_iter=1000, use_2d=False):
    T0, alpha = 50, 0.92
    # 初始解
    lam = np.random.rand() * lambda_max
    dz = 0 if not use_2d else np.random.uniform(-50, 50)
    best_lam, best_dz, best_E = lam, dz, energy_1d(lam, dz)[0]
    T = T0
    for i in range(n_iter):
        # 1. 自适应邻域
        rad_lam = lambda_max * (0.02 + 0.1 * T / T0)
        lam_new = np.clip(lam + np.random.normal(0, rad_lam), 0, lambda_max)
        dz_new = dz if not use_2d else np.clip(dz + np.random.normal(0, 20), -100, 100)
        # 2. 接受准则
        E_new, _, _, _ = energy_1d(lam_new, dz_new)
        dE = E_new - energy_1d(lam, dz)[0]
        if dE > 0 or np.random.rand() < np.exp(dE / max(T, 1e-8)):
            lam, dz = lam_new, dz_new
            if E_new > best_E:
                best_lam, best_dz, best_E = lam, dz, E_new
        # 3. 高温重启动
        if T < 1e-2 and np.random.rand() < 0.05:
            T = T0 * 0.5
        T *= alpha
    P_best = M0 + best_lam * u_m + np.array([0, 0, best_dz])
    return P_best, *energy_1d(best_lam, best_dz)

# ---------------- 主函数 -----------------
def main():
    print("="*60)
    print("烟幕干扰弹优化 - 完善模型1D-SA (膨胀云团+圆柱导弹)")
    print("="*60)
    t0 = time.time()
    best_P, best_E, best_t, best_v, best_valid = sa_multistrategy(use_2d=False)

    elapsed = time.time() - t0
    print(f"\n优化完成! 总耗时: {elapsed:.3f}s")
    print(f"最佳λ: {(best_P - M0).dot(u_m):.4f} m")
    print(f"最佳爆点: [{best_P[0]:.4f}, {best_P[1]:.4f}, {best_P[2]:.4f}]")
    print(f"最佳起爆时间: {best_t:.4f}s")
    if best_valid:
        print(f"最大遮蔽时间: {best_E:.4f}s")
        print(f"无人机速度: {best_v:.4f} m/s ✅ 速度约束满足!")
    else:
        print(f"速度违反: {best_E:.4f} ❌")

    # 可选俯视散点图
    if input("绘制1D搜索历史散点图?(y/n)")=='y':
        plt.figure(figsize=(6,6))
        lam_hist = np.linspace(0,lambda_max,300)
        cov_hist = [cover_1d(l,calc_t_burst(l)) for l in lam_hist]
        plt.plot(lam_hist,cov_hist,'b',lw=2,label='解析遮蔽时长')
        plt.scatter((best_P-M0).dot(u_m),best_E,c='r',s=120,zorder=5,label='最优')
        plt.xlabel('λ / m');plt.ylabel('cover / s');plt.legend();plt.grid()
        plt.title('1D-SA 收敛位置');plt.savefig('output/1D_converge.png',dpi=300)
        print("已保存 → output/1D_converge.png")

    # Excel输出
    os.makedirs("output",exist_ok=True)
    dx,dy=best_P[0]-FY1_0[0],best_P[1]-FY1_0[1]
    heading=np.degrees(np.arctan2(dy,dx))%360
    result={'x_burst':best_P[0],'y_burst':best_P[1],'z_burst':best_P[2],
            't_burst':best_t,'energy_value':best_E,'strategy_valid':best_valid,
            'v_drone':best_v,'theta_deg':heading,'lambda':(best_P-M0).dot(u_m)}
    pd.DataFrame([result]).to_excel("output/Q2_perfect_1D_SA.xlsx",index=False,float_format='%.4f')
    print("结果已保存 → output/Q2_perfect_1D_SA.xlsx")

if __name__=="__main__":
    main()