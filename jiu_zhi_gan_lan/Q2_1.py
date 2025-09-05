from box_targets import BoxTarget
from cloud import Cloud
from missiles import *
def eval_block(x, y, z, t):
    """
    起爆点坐标 + 起爆时刻 → 有效遮挡时长（秒）
    内部直接调你现成的 Scene，跑 20 s 云团寿命即可
    """
    scene = Scene()
    scene.targets.append(BoxTarget(0, scene))
    m = Missile(0, np.array([20000, 0, 2000]), scene)
    scene.missile.append(m)

    # 开启导弹视觉
    scene.missile[0].ir_on = True

    print("烟雾弹起爆点坐标 (m):", np.array([x, y, z]), "时机", t)
    # 1. 把时间轴直接拨到起爆瞬间，导弹先插值到 t 时刻
    dt = 0.1
    for _ in range(int(t/dt)):
        scene.step(t, dt)
        t += dt
    cloud = Cloud(1, np.array([x, y, z]), scene)
    scene.cloud.append(cloud)
    # 2. 跑完云团 20 s 寿命
    dt = 0.01
    for _ in range(int(20/dt)):
        scene.step(t, dt)
        t += dt
    print("已经执行一次仿真，最大遮挡时长：", scene.missile[0].get_blocked_time())
    print("m1导弹有效被遮挡时长：", scene.missile[0].get_blocked_time())
    print("M1导弹当前位置：", scene.missile[0].pos())
    print("烟雾弹坐标 (m):", scene.cloud[0].pos())
    return scene.missile[0].get_blocked_time()