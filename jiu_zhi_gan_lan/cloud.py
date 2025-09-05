from dataclasses import dataclass
from core import Vec3, norm, Entity, Scene


# ---------- 烟幕云团 ----------
class Cloud(Entity):
    def __init__(self, id_: int, pos0: Vec3, scene: Scene):
        self.id_ = id_
        self.centre: Vec3 = pos0.astype(float).copy()
        self.radius: float = 10.0        # 有效遮蔽半径
        self.duration: float = 20.0      # 持续时长
        self.effective_duration: float = 0 #对导弹的有效遮蔽时长
        self.scene= scene
        self.dead: bool = False

    def pos(self) -> Vec3:
        return self.centre

    def contains(self, p: Vec3) -> bool:
        return norm(p - self.centre) <= self.radius

    def update (self, dt: float):
        if self.dead:
            return
        # 云团以 3 m/s 匀速下降
        self.centre[2] -= 3.0 * dt

        # 持续时间递减
        self.duration -= dt
        if self.duration <= 0:
            self.dead = True