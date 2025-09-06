from core import *

def init_burst_candidates(fy_pos, missile_pos, target_pos):
    foot
    pass

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

