from skopt import gp_minimize

from Q2_1 import eval_block

# 搜索空间：x y z t 四维
space = [
    (16900, 17300),   # x  导弹航线附近 ±2 km
    (-10, 10),    # y  左右 1 km
    (1700, 1800),        # z  0~1500 m
    (4.8, 5.3)           # t  0~80 s
]

def obj(v):
    x, y, z, t = v
    print("x=", x, "y=", y, "z=", z, "t=", t)
    return -eval_block(x, y, z, t)   # 负号：skopt 求最小

res = gp_minimize(obj, space, n_calls=40, random_state=0)
x_opt, y_opt, z_opt, t_opt = res.x
print("最优爆点：x=%.1f y=%.1f z=%.1f t=%.2f s" % (x_opt, y_opt, z_opt, t_opt))
print("最大遮挡时长：%.3f s" % -res.fun)

# eval_block(17188,0, 1736, 5.1)