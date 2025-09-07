import numpy as np

from missile_search import validity_time
from terms_P import *
from core import *
import time

start = time.perf_counter()
fy1_init_v_list = []
fy1_init_p_list = []
fy1_t_init_list = []
fy2_init_v_list = []
fy2_init_p_list = []
fy2_t_init_list = []
fy3_init_v_list = []
fy3_init_p_list = []
fy3_t_init_list = []
for t in np.arange(0.1, 70, 0.1):
    fy1_init_v, fy1_init_p = terms(m1_pos, fy5_pos, t, 1200)
    if 70 <= fy1_init_v <= 140 and fy1_init_p[0] >= 0 and fy1_init_p[1] >= 0 and fy1_init_p[2] >= 0:
        fy1_init_v_list.append(fy1_init_v)
        fy1_init_p_list.append(fy1_init_p)
        fy1_t_init_list.append(t)

    fy2_init_v, fy2_init_p = terms(m1_pos, fy5_pos, t, 1150)
    if 70 <= fy2_init_v <= 140 and fy2_init_p[0] >= 0 and fy2_init_p[1] >= 0 and fy2_init_p[2] >= 0:
        fy2_init_v_list.append(fy2_init_v)
        fy2_init_p_list.append(fy2_init_p)
        fy2_t_init_list.append(t)

    fy3_init_v, fy3_init_p = terms(m1_pos, fy5_pos, t, 1000)
    if 70 <= fy3_init_v <= 140 and fy3_init_p[0] >= 0 and fy3_init_p[1] >= 0 and fy3_init_p[2] >= 0:
        fy3_init_v_list.append(fy3_init_v)
        fy3_init_p_list.append(fy3_init_p)
        fy3_t_init_list.append(t)

print(fy1_init_v_list)
print(fy1_init_p_list)
print(fy1_t_init_list)
print(len(fy1_init_v_list))
print(len(fy2_init_v_list))
print(len(fy3_init_v_list))
fy1_best_time = -1
fy1_best_v = -1
fy1_best_p = None
fy1_best_t = -1
for i in range(len(fy1_init_v_list)):
    fy1_init_v = fy1_init_v_list[i]
    fy1_init_p = fy1_init_p_list[i]
    fy1_t_init = fy1_t_init_list[i]
    c = cloud_closure(fy1_init_p[0], fy1_init_p[1], fy1_init_p[2], fy1_t_init)
    time_ = validity_time(m1, target_true_pos, c, fy1_t_init)
    if time_ > fy1_best_time:
        fy1_best_time = time_
        fy1_best_v = fy1_init_v
        fy1_best_p = fy1_init_p
        fy1_best_t = fy1_t_init
    # print(time)
print("fy1 to m1!", fy1_best_v, fy1_best_p, fy1_best_time, fy1_best_t)
fy2_best_time = -1
fy2_best_v = -1
fy2_best_p = None
fy2_best_t = -1
for i in range(len(fy2_init_v_list)):
    fy2_init_v = fy2_init_v_list[i]
    fy2_init_p = fy2_init_p_list[i]
    fy2_t_init = fy2_t_init_list[i]
    c = cloud_closure(fy2_init_p[0], fy2_init_p[1], fy2_init_p[2], fy2_t_init)
    time_ = validity_time(m1, target_true_pos, c, fy2_t_init)
    if time_ > fy2_best_time:
        fy2_best_time = time_
        fy2_best_v = fy2_init_v
        fy2_best_p = fy2_init_p
        fy2_best_t = fy2_t_init
    # print(time)
print("fy2 to m1!", fy2_best_v, fy2_best_p, fy2_best_time, fy2_best_t)
fy3_best_time = -1
fy3_best_v = -1
fy3_best_p = None
fy3_best_t = -1
for i in range(len(fy3_init_v_list)):
    fy3_init_v = fy3_init_v_list[i]
    fy3_init_p = fy3_init_p_list[i]
    fy3_t_init = fy3_t_init_list[i]
    c = cloud_closure(fy3_init_p[0], fy3_init_p[1], fy3_init_p[2], fy3_t_init)
    time_ = validity_time(m1, target_true_pos, c, fy3_t_init)
    if time_ > fy3_best_time:
        fy3_best_time = time_
        fy3_best_v = fy3_init_v
        fy3_best_p = fy3_init_p
        fy3_best_t = fy3_t_init
    # print(time)
print("fy3 to m1!", fy3_best_v, fy3_best_p, fy3_best_time, fy3_best_t)
end = time.perf_counter()

print(f"获取退火初始值耗时：{end - start:.6f} 秒")