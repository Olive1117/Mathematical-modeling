import numpy as np
from core import *

m1_pos = np.array([20000, 0, 2000])
m2_pos = np.array([19000, 600, 2100])
m3_pos = np.array([18000, -600, 1900])
fy1_pos = np.array([17800, 0, 1800])
fy2_pos = np.array([12000, 1400, 1400])
fy3_pos = np.array([6000, -3000, 700])
fy4_pos = np.array([11000, 2000, 1800])
fy5_pos = np.array([13000, -2000, 1300])
# def terms(m, fy, t):
#     return t*np.sqrt(
#         (m[0] + (V_m * t + d) / (np.sqrt(m[2] ** 2 / (m[0] ** 2 + m[1] ** 2) + 1) * np.sqrt((m[1] ** 2 / m[0] ** 2) + 1)) - fy[0]) ** 2 +
#         (m[1] + (V_m * t + d) / (np.sqrt(m[2] ** 2 / (m[0] ** 2 + m[1] ** 2) + 1) * np.sqrt((m[0] ** 2 / m[1] ** 2) + 1)) - fy[1]) ** 2
#         )
def terms(m, fy, t, d):
    m_ = np.array([m[0], m[1]])
    fy_ = np.array([fy[0], fy[1]])
    V_m = 300
    alpha = np.arctan2(m[1], m[0])

    thema = np.arctan(m[2]/np.sqrt(m[0]**2+m[1]**2))
    L = d + V_m * t
    Lm = np.abs((L * np.cos(thema)))
    dx = np.abs(Lm * np.cos(alpha))
    dy = np.abs(Lm * np.sin(alpha))
    H = np.abs(Lm * np.tan(thema))
    if m[1] > 0:
        dy = -dy
    P_ = m_ + np.array([-dx, dy])
    h = m[2] - H
    # print("now", t, "lm", Lm, "fy", fy, "m", m, "P", P_, "alpha", alpha, "h", H)
    if h > fy[2]:
        return np.inf, None
    P = np.array([P_[0], P_[1], m[2] - H])
    fy_to_p = P_ - fy_
    Lfy = np.linalg.norm(fy_to_p)
    v = Lfy / t
    # print("now", t, "v", v, "lfy", Lfy, "lm", Lm, "fy", fy, "m", m, "P", P, "alpha", alpha, "h", H)
    return v, P

if __name__ == '__main__':

    for d in np.arange(2000, 2100, 100):
        n = 0
        for t in np.arange(0.1, 70, 0.1):
            fyv, p = terms(m1_pos, fy1_pos, t, d)
            if 70 <= fyv <= 140 and p[0] >= 0 and p[1] >= 0 and p[2] >= 0:
                n += 1
                # print("now:", t, "fy_v：", fyv, "p", p)
        print("于导弹前", d, "米拦截，共", n, "条有效数据")

