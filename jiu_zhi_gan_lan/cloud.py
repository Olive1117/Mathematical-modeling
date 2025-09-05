from dataclasses import dataclass
from core import Vec3, norm, Entity, Scene
from missile_search import missile_can_see_target


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

    def contains(self, p: Vec3) -> bool:
        return norm(p - self.centre) <= self.radius

    def update (self, dt: float):
        if self.dead:
            return

        if missile_can_see_target()