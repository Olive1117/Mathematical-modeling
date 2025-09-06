# ========== Q2_shadow_screen_fix.py ==========
import numpy as np
from typing import List
from dataclasses import dataclass

# ---------- 基础工具 ----------
Vec3 = np.ndarray
def norm(v: Vec3) -> float:
    return float(np.linalg.norm(v))
def normalize(v: Vec3) -> Vec3:
    n = norm(v)
    return v / n if n > 1e-8 else np.zeros(3)

# ---------- 实体 ----------
class Entity:
    def __init__(self):
        self.dead = None

    def update(self, dt: float): ...
    def pos(self) -> Vec3: ...

class Scene:
    def __init__(self):
        self.targets: List[Entity] = []
        self.missile: List[Entity] = []
        self.cloud: List[Entity] = []

class BoxTarget(Entity):
    def __init__(self, _id: int, scene: Scene, pos0: Vec3 = np.array([0.0, 200.0, 0.0])):
        self.id = _id
        self.centre_bottom = pos0
        self.radius, self.height = 7.0, 10.0
        self.dead = False
    def pos(self) -> Vec3:
        return self.centre_bottom

class Cloud(Entity):
    def __init__(self, _id: int, pos0: Vec3, scene: Scene):
        self.id_ = _id
        self.centre = pos0.astype(float).copy()
        self.radius, self.duration = 10.0, 20.0
        self.dead = False
        self.scene = scene
    def pos(self) -> Vec3:
        return self.centre
    def update(self, dt: float):
        if self.dead:
            return
        self.centre[2] -= 3.0 * dt
        self.duration -= dt
        if self.duration <= 0:
            self.dead = True

# ---------- 导弹 ----------
V = 300.0
MAX_G = 5.0
g = 9.81
MAX_OMEGA = MAX_G * g / V
IR_ON_RANGE = 6000.0

def missile_can_see_target(missile, cloud, target) -> bool:
    o = missile.pos()
    c = cloud.centre
    r = cloud.radius
    n = normalize(c - o)
    N = 16
    ang = np.linspace(0, 2*np.pi, N, endpoint=False)
    dR = np.stack([np.cos(ang), np.sin(ang), np.zeros(N)], axis=1)
    bottom = target.centre_bottom + dR * target.radius
    top = bottom + np.array([0, 0, target.height])
    edges = []
    for i in range(N):
        j = (i + 1) % N
        edges.append((bottom[i], bottom[j]))
        edges.append((top[i], top[j]))
        edges.append((bottom[i], top[i]))
    n_sample = 8
    ts = np.linspace(0, 1, n_sample)

    def ray_hit_cloud(p: Vec3) -> bool:
        d = normalize(p - o)
        denom = n.dot(d)
        if abs(denom) < 1e-8:
            return False
        t = n.dot(c - o) / denom
        if t < 0:
            return False
        x = o + t * d
        return norm(x - c) <= r

    for A, B in edges:
        for t in ts:
            p = (1-t)*A + t*B
            if not ray_hit_cloud(p):
                return True
    return False

class Missile(Entity):
    def __init__(self, _id: int, pos0: Vec3, scene: Scene):
        self.id = _id
        self._pos = pos0.astype(float).copy()
        self._vel = normalize(-pos0) * V
        self.scene = scene
        self.ir_on = False
        self.locked = False
        self.dead = False
        self.blocked_timer = 0.0
        self.prev_blocked = False
    def pos(self) -> Vec3:
        return self._pos
    def get_blocked_time(self) -> float:
        return self.blocked_timer
    def update(self, dt: float):
        if self.dead:
            return
        if not self.ir_on:
            fake_tgt = np.array([0., 0., 0.])
            if norm(self._pos - fake_tgt) <= IR_ON_RANGE:
                self.ir_on = True
            else:
                self._vel = normalize(fake_tgt - self._pos) * V
        if self.ir_on and not self.locked:
            now_blocked = False
            for cloud in self.scene.cloud:
                if cloud.dead:
                    continue
                target = self.scene.targets[0]
                if not missile_can_see_target(self, cloud, target):
                    now_blocked = True
                    break
            if now_blocked:
                self.blocked_timer += dt
            self.prev_blocked = now_blocked
            if not now_blocked:
                self.locked = True
        if self.locked:
            tgt = self.scene.targets[0].pos()
            los = normalize(tgt - self._pos)
            old_dir = normalize(self._vel)
            cos_theta = np.clip(old_dir @ los, -1, 1)
            theta = np.arccos(cos_theta)
            if theta > MAX_OMEGA * dt:
                axis = normalize(np.cross(old_dir, los))
                delta = MAX_OMEGA * dt
                new_dir = (old_dir * np.cos(delta) +
                           np.cross(axis, old_dir) * np.sin(delta))
                self._vel = new_dir * V
            else:
                self._vel = los * V
        self._pos += self._vel * dt
        if norm(self._pos - self.scene.targets[0].centre_bottom) <= 7.0:
            self.dead = True

# ---------- 瞬幕法核心 ----------
def optimal_cloud_center(missile_pos: np.ndarray, target: BoxTarget) -> np.ndarray:
    O = target.centre_bottom
    H = target.height
    R = target.radius
    r = 10.0
    vec_OM = missile_pos - O
    axis = np.array([0, 0, 1])
    t_close = np.clip(vec_OM.dot(axis) / H, 0, 1)
    closest_on_axis = O + t_close * np.array([0, 0, H])
    d_mc = closest_on_axis - missile_pos
    d_mc[2] = 0
    if np.allclose(d_mc, 0):
        d_mc = np.array([1, 0, 0])
    d_mc = normalize(d_mc)
    centre = closest_on_axis + d_mc * (R + r)
    centre[2] = np.clip(missile_pos[2] - 20, 1600, 1800)
    return centre

def occlusion_density(missile_pos: np.ndarray, cloud_centre: np.ndarray,
                      target: BoxTarget, dt_cloud: float = 0.1) -> float:
    blocked_time = 0.0
    cloud = Cloud(0, cloud_centre, None)
    missile = Missile(0, missile_pos, None)
    missile.ir_on = True
    for step in range(int(20.0 / dt_cloud)):
        cloud.update(dt_cloud)
        missile.update(dt_cloud)
        if cloud.dead:
            break
        if not missile_can_see_target(missile, cloud, target):
            blocked_time += dt_cloud
    return blocked_time

def shadow_screen(max_t=60.0, dt=0.5):
    best_rho, best_t, best_center = 0.0, 0.0, None
    target = BoxTarget(0, None)
    for i in range(int(max_t / dt)):
        t_i = i * dt
        M_i = np.array([20000 - 300 * t_i, 0, 2000 - 100 * t_i])
        if M_i[0] < 6000:
            break
        centre = optimal_cloud_center(M_i, target)
        rho = occlusion_density(M_i, centre, target)
        if rho > best_rho:
            best_rho, best_t, best_center = rho, t_i, centre
    return best_center, best_t, best_rho

def uav_param(opt_centre: np.ndarray, uav_init: np.ndarray, t_burst: float):
    vec = opt_centre - uav_init
    vec[2] = 0
    dist = np.linalg.norm(vec)
    speed = np.clip(dist / t_burst, 70, 140)
    heading = np.arctan2(vec[1], vec[0])
    return speed, heading

# ---------- 一键运行 + 调试 ----------
if __name__ == "__main__":
    centre, t_burst, rho = shadow_screen()
    speed, heading = uav_param(centre, np.array([17800, 0, 1800]), t_burst)
    print("=== 瞬幕法结果 ===")
    print("最优起爆点", centre)
    print("起爆时刻", f"{t_burst:.2f} s")
    print("遮挡时长", f"{rho:.2f} s")
    print("无人机速度", f"{speed:.2f} m/s")
    print("航向角", f"{np.degrees(heading):.1f}°")