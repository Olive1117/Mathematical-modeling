from core import *

from jiu_zhi_gan_lan.new.core import cloud_closure, target_true_pos
from missile_search import validity_time

# 已知参数
fy1_pos = np.array([17800.0, 0.0, 1800.0])  # FY1初始位置 (m)
fake_pos = np.array([0.0, 0.0, 0.0])        # 假目标位置 (m)
v_fy = 120.0                                # FY1飞行速度 (m/s)
t_drop = 1.5                                # 受领任务到投放的时间 (s)
t_bang = 3.6                                # 投放到起爆的时间 (s)
g = 9.8                                     # 重力加速度 (m/s^2)

# 1. 计算FY1飞行方向单位向量
direction = fake_pos - fy1_pos
direction[2] = 0.0
direction = direction / np.linalg.norm(direction)

# 2. 计算投放点位置（z坐标不变）
drop_pos = fy1_pos.copy()
drop_pos[:2] += v_fy * t_drop * direction[:2]  # 仅更新x,y

# 3. 计算起爆点位置
# 水平速度向量（x, y 分量）
v_hor = v_fy * direction[:2]
# 初始垂直速度（z 分量）
v_z0 = 0

# 水平位移
delta_xy = v_hor * t_bang
# 垂直位移（向下为负）
delta_z = v_z0 * t_bang - 0.5 * g * t_bang**2

# 起爆点坐标
bang_pos = np.array([
    drop_pos[0] + delta_xy[0],
    drop_pos[1] + delta_xy[1],
    drop_pos[2] + delta_z
])
c1 = cloud_closure(bang_pos[0], bang_pos[1], bang_pos[2], 5.1)
time = (validity_time(m1, target_true_pos, c1, 5.1))
print(c1(5.1))
print(time)
c1(5.1)
print()