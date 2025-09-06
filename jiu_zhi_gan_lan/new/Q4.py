import numpy as np

from core import *
from missile_search import validity_time


def init_burst_candidates(fy_pos, missile_pos, target_pos, dx, dy, dz, nx, ny, nz, dt, nt):
    init_point = foot_on_segment(fy_pos, missile_pos, target_pos)
    x0, y0, z0 = init_point
    # 起点 = 中心 - 半宽
    x = np.linspace(x0 - dx / 2, x0 + dx / 2, nx)
    y = np.linspace(y0 - dy / 2, y0 + dy / 2, ny)
    z = np.linspace(z0 - dz / 2, z0 + dz / 2, nz)
    grid = {'x': x, 'y': y, 'z': z}
    X, Y, Z = np.meshgrid(grid['x'], grid['y'], grid['z'], indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N,3)
    init_time = np.linalg.norm(init_point - missile_pos) / 300
    time = np.linspace(init_time - dt / 2, init_time + dt / 2, nt)
    print("init point", init_point, "init time", init_time)
    return pts, init_point, time, init_time

def foot_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    点到线段 AB 的垂足（若垂足超出线段则返回最近端点）

    参数
    ----
    p : (3,)  线外点
    a : (3,)  线段起点
    b : (3,)  线段终点

    返回
    ----
    q : (3,)  垂足/最近点
    """
    ab = b - a
    ap = p - a
    # 投影参数 t ∈ [0, 1]
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
    return a + t * ab


pts, init_point, time, init_time = init_burst_candidates(np.array([12000, 1400, 1400]), m1(0), target_true_pos, 1000, 500, 500, 10, 5, 10, 10, 5)
print("pts", pts, "init_point", init_point, "time", time, init_time)
print(len(pts), len(time))
time_ = [3, 5, 7, 9, 11]
def find_init_best_point(pts, time):
    n = 0
    lon = len(pts)* len(time)
    best_time = np.inf
    best_point = None
    best_validity_time = np.inf
    for i in pts:
        for j in time:
            n += 1
            print(f"当前第{n}轮查找，共有{lon}轮，当前最优time:{best_time},最优point:{best_point} | ", end='')
            c = cloud_closure(i[0], i[1], i[2], j)
            validity_time_ = validity_time(m1, target_true_pos, c, j)
            print(f"point:{i}, timeL{j}, 有效时长：{validity_time_}")
            if validity_time_ > best_validity_time:
                best_validity_time = validity_time_
                best_point = i
                best_time = j
    return best_point, best_time
best_point, best_time = find_init_best_point(pts, time_)
print("best point", best_point, "best time", best_time)