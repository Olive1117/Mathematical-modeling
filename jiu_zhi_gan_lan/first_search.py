import numpy as np

from Q2_1 import eval_block

def find_best_burst_at_t(t: float,
                         x0: float = 17188,
                         y0: float = 0,
                         z0: float = 1736,
                         step: float = 1.0,):
    """
    单峰六邻域爬山：一轮无改进即认为到达峰值
    返回: (best_x, best_y, best_z, best_time)
    """
    # 六个方向
    dirs = np.array([[step, 0, 0], [-step, 0, 0],
                     [0, step, 0], [0, -step, 0],
                     [0, 0, step], [0, 0, -step]])

    best_p = np.array([x0, y0, z0], dtype=float)
    best_v = eval_block(*best_p, t)
    print(f"[init]  t={t:.2f}s  p=[{best_p[0]:.2f} {best_p[1]:.2f} {best_p[2]:.2f}]  block={best_v:.3f}s")

    while True:
        for d in dirs:
            new_p = best_p + d
            v = eval_block(*new_p, t)
            if v > best_v + 1e-6:  # 严格更好
                best_p, best_v = new_p, v
                print(f"[climb] t={t:.2f}s  p=[{best_p[0]:.2f} {best_p[1]:.2f} {best_p[2]:.2f}]  block={best_v:.3f}s")
                break  # 立即重新扫六向
        else:  # 六个方向都未能改进
            break  # 到峰顶，直接退出
    return *best_p, best_v

def get_global_best_burst(t: float,
                          x0: float = 17188,
                          y0: float = 0,
                          z0: float = 1736,
                          steps: tuple = (100, 10, 1)):
    """
    给定起爆时刻 t，返回全局最优爆点坐标与最大遮蔽时长
    返回: (best_x, best_y, best_z, best_time)
    """
    x, y, z = x0, y0, z0
    for dt in steps:
        x, y, z, tb = find_best_burst_at_t(t, x0=x, y0=y, z0=z, step=dt)
    return x, y, z, tb

# ------------------ demo ------------------
if __name__ == "__main__":
    t_fix = 5.1          # 你想优化的固定起爆时刻
    steps = [10, 1, 0.1]  # 从大到小
    x, y, z = 17188, 0, 1736  # 初值
    for dt in steps:
        x, y, z, tb = find_best_burst_at_t(t_fix, x0=x, y0=y, z0=z, step=dt)
    print("\nFinal best burst:")
    print(f"  pos = ({x:.1f}, {y:.1f}, {z:.1f}) m")
    print(f"  block time = {tb:.3f} s")