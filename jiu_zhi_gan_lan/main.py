import numpy as np
from core import Scene, Vec3
from missiles import Missile
from drones import Drone

def run_one():
    scene = Scene()
    # 真目标在 (0,200,0)
    scene.truth_pos = np.array([0.0, 200.0, 0.0])

    m = Missile(1, np.array([20000.0, 0.0, 2000.0]),
                np.array([-300.0, 0.0, -100.0]), scene)
    scene.add(m)

    d = Drone(1, np.array([17800.0, 0.0, 1800.0]),
              np.array([120.0, 0.0, 0.0]))
    scene.add(d)
    # 1.5 s 后投放，起爆点算好了在 (17200,0,1700)
    d.drop_bomb(delay=1.5, burst_pos=np.array([17200.0, 0.0, 1700.0]), scene=scene)

    dt = 0.05
    max_t = 100.0
    t = 0.0
    while t < max_t and not m.dead:
        scene.step(t, dt)
        # 6 km 开启末制导
        if not m.guidance_on and np.linalg.norm(m.pos - scene.truth_pos) <= 6000:
            m.enable_guidance()
        t += dt

    df = scene.to_df()
    print("脱靶量", np.linalg.norm(m.pos - scene.truth_pos))
    df.to_csv("run.csv", index=False)


if __name__ == "__main__":
    run_one()