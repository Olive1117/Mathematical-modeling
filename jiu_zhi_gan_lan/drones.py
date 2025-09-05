import numpy as np
from core import Entity, Vec3, Scene
from cloud import Cloud


# ---------- 无人机 ----------

class Drone(Entity):
    def __init__(self, id_: int, pos0: Vec3, bomb_int):
        self.id = id_
        self._pos = pos0.astype(float).copy()
        self._vel = np.array([0, 0, 0]).astype(float).copy()
        self.dead = False
        self.bombs: int = bomb_int

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
    def __init__(self, delay: float, burst_pos: Vec3, scene: Scene):
        self.delay = delay          # 爆炸倒计时（秒）
        self.burst_pos = burst_pos  # 爆炸发生的三维坐标
        self.scene = scene          # 场景对象，里面至少有一个 clouds 列表
        self.dropped = False        # 标记是否已经爆炸过

    def update(self, dt: float):
        if self.dropped:
            return
        self.delay -= dt
        if self.delay <= 0:
            self.scene.clouds.append(Cloud(centre = self.burst_pos))
            self.dropped = True