# -*- coding: utf-8 -*-
"""
为 FY2、FY3 生成退火初值参数
调用之前已经写好的遮蔽时长函数
"""
import numpy as np
from core import *
from missile_search import validity_time

# ---------------- 场景常量 ----------------
V_UAV   = np.arange(70, 141, 20)          # 速度离散 10 m/s 一档
T_DROP  = np.arange(0.5, 5.1, 1)        # 投放延迟离散 0.5 s
T_BOOM  = np.arange(1.0, 6.1, 1)        # 起爆延迟离散 0.5 s
PSI_RES = np.deg2rad(5)                   # 航向角分辨率 3°
TOP_K   = 20                              # 保留前 K 组种子

# ---------------- 无人机初始配置 ----------------
uav_config = {
    'FY1': {'pos': np.array([17800, 0, 1800])},
    'FY2': {'pos': np.array([12000, 1400, 1400])},
    'FY3': {'pos': np.array([6000, -3000, 700])},
}

def anneal_param_generator(uav_id, missile_fun, top_k=TOP_K):
    """
    对指定无人机 + 指定导弹，粗搜 4 维参数空间
    返回可直接喂给退火算法的 TOP_K 组初值
    形状：(top_k, 4)  →  [psi, v, t_drop, t_boom]
    """
    uav_pos = uav_config[uav_id]['pos']
    buffer  = []          # (cover, psi, v, t_drop, t_boom)
    n = 0
    for v in V_UAV:
        for td in T_DROP:
            for tb in T_BOOM:
                # 1. 投放点
                drop_pt = uav_pos[:2] + v * td * np.array([1, 0])  # 先假设初始航向 0°
                # 2. 航向粗搜
                for psi in np.arange(-np.pi, np.pi, PSI_RES):
                    n+=1
                    print(f"当前第{n}次, 共{len(V_UAV)*len(T_DROP)*len(T_BOOM)*len(np.arange(-np.pi, np.pi, PSI_RES))}")
                    drop_pt = uav_pos[:2] + v * td * rotate_uv(psi)
                    tbops_pt = uav_pos[:2] + v * (td+tb) * rotate_uv(psi)
                    H = uav_pos[2] - 0.5 *9.8*tb**2
                    # 3. 云团函数
                    cloud_fun = cloud_closure(tbops_pt[0], tbops_pt[1], H, td + tb)
                    # 4. 遮蔽时长
                    cover = validity_time(missile_fun, target_true_pos, cloud_fun, td + tb)
                    print(cover)
                    buffer.append((cover, psi, v, td, tb))

    buffer.sort(key=lambda x: x[0], reverse=True)
    top = buffer[:top_k]
    return np.array([(p[1], p[2], p[3], p[4]) for p in top])
def rotate_uv(a, deg=False):
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
    e / np.linalg.norm(e)
    ans = np.array([e[0], e[1]])
    return ans

# ---------------- 主调 ----------------
if __name__ == "__main__":
    # 问题 4 只干扰 M1
    m1_fun = missile_closure(20000, 0, 2000)

    for uav in ['FY2', 'FY3']:
        params = anneal_param_generator(uav, m1, top_k=TOP_K)
        np.save(f"{uav}_anneal_init.npy", params)
        print(f"{uav} 退火初值已生成，形状：{params.shape}")
        # 顺手打印最优一组
        best = params[0]
        print(f"  最优种子：ψ={np.rad2deg(best[0]):5.1f}°  v={best[1]:3.0f}  td={best[2]:4.1f}  tb={best[3]:4.1f}")