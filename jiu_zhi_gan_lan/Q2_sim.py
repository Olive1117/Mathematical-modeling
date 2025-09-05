# ========== Q2_MC_standalone.py ==========
import numpy as np
from typing import List
import pandas as pd
from dataclasses import dataclass

# ---------- 基础工具 ----------
Vec3 = np.ndarray   # shape (3,)
def norm(v: Vec3) -> float:
    return float(np.linalg.norm(v))
def normalize(v: Vec3) -> Vec3:
    n = norm(v)
    return v / n if n > 1e-8 else np.zeros(3)

# ---------- 实体 ----------
class Entity:
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

# ---------- 参数 & 仿真 ----------
class Params:
    __slots__ = ("x", "y", "z", "t_burst")
    def __init__(self, x, y, z, t_burst):
        self.x, self.y, self.z, self.t_burst = x, y, z, t_burst

def simulate_occlusion_time(p: Params) -> float:
    scene = Scene()
    scene.targets.append(BoxTarget(0, scene))
    m = Missile(1, np.array([20000., 0., 2000.]), scene)
    scene.missile.append(m)
    dt = 0.05
    t = 0.0
    while t < p.t_burst:
        for e in scene.missile + scene.cloud + scene.targets:
            e.update(dt)
        t += dt
        if m.dead:
            break
    if m.dead:
        return 0.0
    cloud = Cloud(1, np.array([p.x, p.y, p.z]), scene)
    scene.cloud.append(cloud)
    for _ in range(int(20.0 / dt)):
        for e in scene.missile + scene.cloud + scene.targets:
            e.update(dt)
        t += dt
        if m.dead:
            break
    return m.get_blocked_time()

# ---------- 蒙特卡洛 ----------
def mc_baseline(n: int = 300_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(17000, 17500, n)
    y = rng.uniform(-500, 500, n)
    z = rng.uniform(1600, 1800, n)
    t = rng.uniform(0, 8, n)
    scores = np.empty(n, dtype=float)
    for i in range(n):
        scores[i] = simulate_occlusion_time(Params(x[i], y[i], z[i], t[i]))
    best = np.argmax(scores)
    return scores[best], x[best], y[best], z[best], t[best]

# ---------- 运行 ----------
if __name__ == "__main__":
    best_score, best_x, best_y, best_z, best_t = mc_baseline(300_000)
    print("=== 朴素 MC baseline ===")
    print("最佳遮蔽时长", f"{best_score:.2f}", "s")
    print("起爆点 X", f"{best_x:.2f}", "m")
    print("起爆点 Y", f"{best_y:.2f}", "m")
    print("起爆点 Z", f"{best_z:.2f}", "m")
    print("起爆时刻", f"{best_t:.2f}", "s")