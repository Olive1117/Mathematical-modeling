import numpy as np

from core import Vec3, normalize, norm
# from box_targets import BoxTarget
# from cloud import Cloud
# from missiles import Missile

# TODO 导弹视觉模拟已经写好了，需要的参数有导弹类的当前坐标，云团类的坐标半径，目标类的
# ---------- 导弹视觉模拟 ----------

def missile_can_see_target(missile, cloud, target) -> bool:
    """
    视锥判断：

    1. 把圆柱表面离散成若干条线段；

    2. 对线段上均匀采样的点，求“导弹→该点”的射线与云团圆盘的交点；

    3. 若存在至少一个采样点不与云盘相交，则认为导弹能看见目标，立即返回 True；

    4. 所有线段都被云盘“挡”住才返回 False。
    """

    """
        返回射线与圆形平面的交点坐标，若无交点返回 None
        o : 射线起点
        d : 射线方向（已单位化）
        c : 圆盘中心
        r : 圆盘半径
        n : 圆盘法向量（已单位化）
    """
    o = missile.pos()
    c = cloud.centre
    r = cloud.radius
    n = normalize(c - o)

    # 1. 离散圆柱
    N = 16
    ang = np.linspace(0, 2*np.pi, N, endpoint=False)
    dR  = np.stack([np.cos(ang), np.sin(ang), np.zeros(N)], axis=1)
    bottom = target.centre_bottom + dR * target.radius
    top    = bottom + np.array([0, 0, target.height])

    # 2. 生成所有边（底面、顶面、母线）
    edges = []
    for i in range(N):
        # j = (i + 1) % N
        # edges.append((bottom[i], bottom[j]))  # 底边
        # edges.append((top[i],    top[j]))     # 顶边
        edges.append((bottom[i], top[i]))     # 竖边

    # 3. 对每条边采样
    n_sample = 8                       # 每条边采样点数
    ts = np.linspace(0, 1, n_sample)

    def ray_hit_cloud(p: Vec3) -> bool:
        """返回 True 表示射线 M_pos→p 与云盘相交"""

        """
         公式
         平面方程：n · (X − c) = 0 （两向量点乘为0）
         射线方程：X(t) = o + t*d, t ≥ 0 （让动点X从起点o开始，沿d方向走t个单位）
         联立平面方程和射线方程得
         n · (o + t*d − c) = 0
         -> n · (o − c) + t (n · d) = 0
         -> t = − (n · (o − c)) / (n · d)
         -> t = (n · (c − o)) / (n · d)
         """
        d = normalize(p - o)
        denom = n.dot(d)            # = n·d = |D| > 0
        if abs(denom) < 1e-8:       # 射线与平面平行，视为不相交
            return False
        t = n.dot(c - o) / denom    # 移动系数t
        if t < 0:                   # 交点在射线反向延长线上
            return False
        x = o + t * d               # 复原交点x
        # 判断交点是否在圆盘内
        return norm(x - c) <= r

    # 4. 逐线段检查
    for A, B in edges:
        for t in ts:
            p = (1-t)*A + t*B
            if not ray_hit_cloud(p):    # 只要有一个点不被云挡，就看见了
                return True
    return False