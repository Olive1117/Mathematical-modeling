import numpy as np
from q3 import validity_time, missile, target_true_pos

# 1. 参数
t_blast = 1.0
times = np.arange(0, 6.1, 0.2)
centers = np.array([missile(t) for t in times])
cloud_func = lambda t: centers[np.clip(int((t - 0.) / 0.2), 0, len(centers) - 1)] + np.array([0, 0, -3 * (t - 0.)])

# 2. 正确调用（4个参数必须全传）
print('紧贴弹道+50 m 遮蔽=', validity_time(missile, target_true_pos, cloud_func, t_blast))