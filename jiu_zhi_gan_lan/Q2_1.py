from box_targets import BoxTarget
from cloud import Cloud
from missiles import *
def eval_block(x, y, z, t):
    """
    起爆点坐标 + 起爆时刻 → 遮挡比例（0~1）
    云团寿命 20 s，但导弹落地就停算，返回遮挡比例
    """
    scene = Scene()
    scene.targets = BoxTarget(0, scene)
    m = Missile(0, np.array([20000, 0, 2000]), scene)
    scene.missile.append(m)
    m.ir_on = True

    # 1. 先跑到 t
    dt = 0.1
    t_sim = 0.0
    while t_sim < t:
        scene.step(t_sim, dt)
        t_sim += dt

    # 2. 起爆
    cloud = Cloud(1, np.array([x, y, z]), scene)
    scene.cloud.append(cloud)

    # 3. 跑到导弹落地或 20 s 到期
    cloud_lifetime = 20.0
    while t_sim < t + cloud_lifetime:
        print("m1导弹有效被遮挡时长：", scene.missile[0].get_blocked_time())
        scene.step(t_sim, dt)
        t_sim += dt

    # print("已经执行一次仿真，最大遮挡时长：", scene.missile[0].get_blocked_time())
    # print("m1导弹有效被遮挡时长：", scene.missile[0].get_blocked_time())
    # print("M1导弹当前位置：", scene.missile[0].pos())
    # print("烟雾弹坐标 (m):", scene.cloud[0].pos())
    return m.get_blocked_time()
    # total_possible = t_sim - t
    # if total_possible <= 0:
    #     return 0.0
    # return m.get_blocked_time() / total_possible