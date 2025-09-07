import numpy as np

Vec3 = np.ndarray   # shape (3,)

def normalize(v: Vec3) -> Vec3:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros(3)

target_false_pos = np.array([0, 0, 0])
target_true_pos = np.array([0, 200, 0])
# 1. 离散圆柱
N = 16
ang = np.linspace(0, 2*np.pi, N, endpoint=False)
dR  = np.stack([np.cos(ang), np.sin(ang), np.zeros(N)], axis=1)
bottom = target_true_pos + dR * 7
top    = bottom + np.array([0, 0, 10])
# 2. 生成所有边（底面、顶面、母线）
edges = []
for i in range(N):
    edges.append((bottom[i], top[i]))
# 3. 对每条边采样
n_sample = 8                       # 每条边采样点数
ts = np.linspace(0, 1, n_sample)
# 4. 逐线段检查
sampling_point = []
for A, B in edges:
    for t in ts:
        p = (1-t)*A + t*B
        sampling_point.append(p)
sampling_point = np.array(sampling_point)


def missile_closure(x, y, z):
    pos = np.array([x, y, z])
    n = normalize(np.array([-x, -y, -z]))
    v = 300
    distance = np.linalg.norm(pos)
    time_to_target = distance / v

    def f(dt):
        if dt < 0:
            raise ValueError("时间不能为负数")
        dt = min(dt, time_to_target)
        return pos + n * v * dt
    return f

def cloud_closure(x, y, z, t):
    pos = np.array([x, y, z])
    n = np.array([0, 0, -1])
    v = 3
    duration = 20

    def f(dt):
        if dt > duration + t:
            return None
        if dt < t:
            return None
        return pos + n * v * (dt - t)
    return f

m1 = missile_closure(20000, 0, 2000)
m2 = missile_closure(19000, 600, 2100)
m3 = missile_closure(18000, -600, 1900)

# c1 = cloud_closure(17188, 0, 1736, 5.1)
