# search_time_axis.py
import numpy as np
from first_search import get_global_best_burst, find_best_burst_at_t

def search_time_axis(t_start: float, t_end: float, dt: float = 0.1, space_step: float = 1.0):
    """
    沿时间轴步进搜索最优爆点
    参数:
        t_start : 起始时刻（s）
        t_end   : 终止时刻（s）
        dt      : 时间步长，+0.1 表示正向，-0.1 表示反向
        space_step : 每个时刻六邻域 refine 的空间步长
        (x0,y0,z0) : 初始爆点坐标（t_start 时刻）
    返回:
        ans     : list，元素为 [t, x, y, z, block_time]
    """
    ans = []
    # 用 t_start 的全局最优当 **初始爆点**
    x, y, z, _ = get_global_best_burst(t_start)
    t = t_start

    # 决定循环方向
    if dt > 0:
        cond = lambda: t <= t_end
    else:
        cond = lambda: t >= t_end

    while cond():
        x, y, z, bt = find_best_burst_at_t(t, x, y, z, space_step)
        ans.append([t, x, y, z, bt])
        t += dt

    return ans


# ------------------ 简单测试 ------------------
if __name__ == "__main__":
    result = search_time_axis(5.1, 7, 0.1)   # 正向 5.1→6.0 s
    # result = search_time_axis(5.1, 4.0, -0.1)  # 反向 5.1→4.0 s
    for row in result:
        t, x, y, z, bt = row
        print(f"t={t:.1f}s  ({x:.1f},{y:.1f},{z:.1f})m  block={bt:.3f}s")