import numpy as np
from skopt import gp_minimize

from Q2_1 import eval_block

n = 20
while (n>0):
    seed = np.random.default_rng().integers(0, 2 ** 31)  # 0~2 147 483 647

    # 搜索空间：x y z t 四维
    space = [
        (16900.000, 17300.000),  # x  导弹航线附近 ±2 km
        (-10.000, 10.000),  # y  左右 1 km
        (1700.000, 1800.000),  # z  0~1500 m
        (4.800, 5.300)  # t  0~80 s
    ]


    def obj(v):
        x, y, z, t = v
        # print("x=", x, "y=", y, "z=", z, "t=", t)
        return -eval_block(x, y, z, t)  # 负号：skopt 求最小

    # 这行代码的作用是：
    # 启动一个基于高斯过程的贝叶斯优化循环，在 4 维参数空间里最多做 40 次“爆炸-评估”实验，找到能让“有效遮挡时长”最长的 (x, y, z, t) 组合。
    res = gp_minimize(obj, space, n_calls=40, random_state=seed)
    x_opt, y_opt, z_opt, t_opt = res.x
    print("当前第", n, "次仿真")
    print("最优爆点：x=%.1f y=%.1f z=%.1f t=%.2f s" % (x_opt, y_opt, z_opt, t_opt))
    print("本次全局最优遮挡时长：%.3f s" % -res.fun)
    print("本次随机种子：", seed)
    n -= 1


