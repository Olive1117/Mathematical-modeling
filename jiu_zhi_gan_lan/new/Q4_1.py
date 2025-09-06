import numpy as np
from typing import List, Dict
from missile_search import *

g = 9.8
CLOUD_DOWN = 3.0
EFF_RANGE = 10.0
V_UAV = np.arange(70, 141, 15)   # 70-140 m/s 每隔 10

def missile_pos(s: np.ndarray, m0: np.ndarray, target: np.ndarray):
    """导弹线段参数化：s=0→m0, s=1→target"""
    return m0 + s[:, None] * (target - m0)

def pipe_sample(m0: np.ndarray, target: np.ndarray,
                n_long: int = 200, n_pipe: int = 300) -> np.ndarray:
    """
    在导弹-目标线段附近「管道内」均匀随机采样
    返回 (N,3) 点云
    """
    # 1. 沿轨迹 s
    s = np.linspace(0, 1, n_long)
    # 2. 局部正交基
    ab = target - m0
    e1 = ab / np.linalg.norm(ab)                  # 纵向
    e2 = np.array([1, 0, 0])                      # 随便找一个与 e1 不平行的
    if abs(np.dot(e1, e2)) > 0.95:                # 防止共线
        e2 = np.array([0, 1, 0])
    e2 = e2 - np.dot(e2, e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)                         # 第三个正交方向

    # 3. 随机径向
    r = np.random.uniform(0, EFF_RANGE, n_pipe)
    theta = np.random.uniform(0, 2*np.pi, n_pipe)
    u = r * np.cos(theta)
    v = r * np.sin(theta)

    # 4. 随机纵向
    s_rand = np.random.choice(s, n_pipe)
    base = missile_pos(s_rand, m0, target)
    pts = base + u[:, None]*e2 + v[:, None]*e3
    return pts

def refine_to_reachable(pts: np.ndarray,
                        fy_pos: np.ndarray,
                        m0: np.ndarray,
                        target: np.ndarray) -> List[Dict]:
    """
    把管道采样点「 refine 」成无人机真正可达的起爆点
    用牛顿法一步把下落时间压收敛，再判飞行时间
    """
    ab = target - m0
    t_missile = np.linalg.norm(ab) / 300.0   # 导弹从 m0→target 需时

    # 牛顿法求 z_D 使得  z_D - 0.5*g*t_f^2 = z
    z = pts[:, 2]
    z_D = z + 15.0
    for _ in range(5):
        t_f = np.sqrt(2 * (z_D - z) / g)
        z_D = z + 0.5 * g * t_f**2
    D = np.column_stack([pts[:, 0], pts[:, 1], z_D])   # 投放点
    fall_t = t_f

    fly_dist = np.linalg.norm(D - fy_pos, axis=1)
    feasible = []
    for v in V_UAV:
        fly_t = fly_dist / v
        mask = (fly_t + fall_t) < t_missile
        for idx in np.where(mask)[0]:
            feasible.append({
                'x': float(pts[idx, 0]),
                'y': float(pts[idx, 1]),
                'z': float(pts[idx, 2]),
                'v': int(v),
                't_fly': float(fly_t[idx]),
                't_fall': float(fall_t[idx])
            })
    return feasible
def refine_to_reachable_2k(pts: np.ndarray,
                           fy_pos: np.ndarray,
                           m0: np.ndarray,
                           target: np.ndarray,
                           max_keep: int = 2000) -> List[Dict]:
    ab = target - m0
    t_missile = np.linalg.norm(ab) / 300.0

    z = pts[:, 2]
    z_D = z + 15.0
    for _ in range(5):
        t_f = np.sqrt(2 * (z_D - z) / g)
        z_D = z + 0.5 * g * t_f**2
    D = np.column_stack([pts[:, 0], pts[:, 1], z_D])
    fall_t = t_f


    # 只飞最快档 140 m/s
    v_fast = 140.0
    fly_t = np.linalg.norm(D - fy_pos, axis=1) / v_fast
    mask = (fly_t + fall_t) < t_missile

    cand = []
    for idx in np.where(mask)[0]:
        cand.append({
            'x': float(pts[idx, 0]),
            'y': float(pts[idx, 1]),
            'z': float(pts[idx, 2]),
            'v': int(v_fast),
            't_fly': float(fly_t[idx]),
            't_fall': float(fall_t[idx]),
        })
    return cand[:max_keep]
def dict_to_theta_v_tf(d: dict, fy_pos: np.ndarray) -> dict:
    """
    输入：{'x','y','z','v','t_fly','t_fall'}
    输出：{'theta','v','t_fly','t_fall'}
    theta ∈ [0, 2π) 为无人机在水平面内航向角（北偏东为正）
    """
    dx = d['x'] - fy_pos[0]
    dy = d['y'] - fy_pos[1]
    theta = np.atan2(dy, dx)          # -π..π
    if theta < 0:
        theta += 2 * np.pi            # 统一到 0..2π
    return {
        'theta': theta,
        'v': d['v'],
        't_fly': d['t_fly'],
        't_fall': d['t_fall']
    }
def find_init_as(params):
    N =0
    lon = len(params)
    best_time = -1
    best_theta = None
    best_v = None
    best_t_release = None
    best_t_detonate = None
    for i in params:
        N += 1
        theta = i['theta']
        v = i['v']
        t_release = i['t_fly']
        t_detonate = i['t_fall']
        n = angle_to_unit_vector(theta)
        v = v
        g = 9.8
        fy1 = np.array([17800, 0, 1800])
        pos_release = fy1 + n * v * t_release
        # print(pos_release)
        pos_detonate = fy1 + n * v * (t_detonate + t_release)
        # print("爆点坐标无z",pos_detonate, "v", v, "t", t_release+t_detonate, "a", a)
        pos_detonate[2] = 1800 - 0.5 * g * t_detonate ** 2
        c = cloud_closure(pos_detonate[0], pos_detonate[1], pos_detonate[2], t_release + t_detonate)
        # print("爆点坐标有z",pos_detonate[0] , pos_detonate[1], pos_detonate[2],t_release + t_detonate)
        time = validity_time(m1, target_true_pos, c, t_release + t_detonate)
        print(f"当前第{N}轮查找，共有{lon}轮，当前最优time:{best_time},最优theta:{best_theta},最优v{best_v},最优release{best_t_release},最优detonate{best_t_detonate} | ", end='')
        print(f"theta:{theta},v:{v},release:{t_release},detonate:{t_detonate}有效时长：{time}")
        if time > best_time:
            best_time = time
            best_theta = theta
            best_v = v
            best_t_release = t_release
            best_t_detonate = t_detonate
    return np.array([best_theta, best_v, best_t_release, best_t_detonate])
def angle_to_unit_vector(a, deg=True):
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
def generate_cloud_centers_dense(m0, target, ds=0.2, dr=1.0, dtheta=15):
    ab = target - m0
    L = np.linalg.norm(ab)
    e1 = ab / L
    e2 = np.array([1,0,0])
    if abs(np.dot(e1, e2)) > 0.95:
        e2 = np.array([0,1,0])
    e2 = e2 - np.dot(e2, e1)*e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)

    centers = []
    s_vals = np.arange(0, L, ds)
    r_vals = np.arange(0, EFF_RANGE+1e-3, dr)
    theta_vals = np.arange(0, 360, dtheta)

    for s in s_vals:
        base = m0 + s * e1
        for r in r_vals:
            for theta in theta_vals:
                rad = np.deg2rad(theta)
                offset = r * (np.cos(rad) * e2 + np.sin(rad) * e3)
                centers.append(base + offset)
    return np.array(centers)
m0   = np.array([20000, 0, 2000])
tgt  = np.array([0, 200, 0])

fy   = np.array([17800, 0, 1800])
pts_cloud = pipe_sample(m0, tgt, n_long=500, n_pipe=500)
cands = refine_to_reachable(pts_cloud, fy, m0, tgt)
print(f"管道采样 5000 → 合格 {len(cands)} 个")
# 直接当模拟退火初始池
print("构造云团中心点数：", len(pts_cloud))
print("合格可达点数：", len(cands))















