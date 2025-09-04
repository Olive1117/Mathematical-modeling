import numpy as np
from core import Entity, Vec3, Cloud

# ---------- 无人机 ----------

class Drone(Entity):
    def __init__(self, id_: int, pos0: Vec3, vel: Vec3):
        self.id = id_
        self._pos = pos0.astype(float).copy()
        self._vel = vel.astype(float).copy()
        self.dead = False
        self.bombs: list[Bomb] = []

    def pos(self) -> Vec3:
        return self._pos

    def update(self, dt: float):
        self._pos += self._vel * dt
        for b in self.bombs:
            b.update(dt)

    def drop_bomb(self, delay: float, burst_pos: Vec3, scene):
        """delay 秒后在 burst_pos 起爆"""
        self.bombs.append(Bomb(delay, burst_pos, scene))


class Bomb:
    def __init__(self, delay: float, burst_pos: Vec3, scene):
        self.delay = delay
        self.burst_pos = burst_pos
        self.scene = scene
        self.dropped = False

    def update(self, dt: float):
        if self.dropped:
            return
        self.delay -= dt
        if self.delay <= 0:
            self.scene.clouds.append(Cloud(self.burst_pos))
            self.dropped = True