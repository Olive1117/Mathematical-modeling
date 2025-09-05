import numpy as np
from dataclasses import dataclass
from core import Entity, Vec3, Scene


# ---------- 目标 ----------

class BoxTarget(Entity):
    def __init__(self, id_: int, scene: Scene, pos0: Vec3 = np.array([0.0, 200.0, 0.0])):
        self.id = id_
        self.centre_bottom= pos0         # 底面圆心坐标
        self.scene = scene
        self.radius: float = 7.0          # 半径 (m)
        self.height: float = 10.0         # 高度 (m)
        self.dead: bool = False

    def pos(self) -> Vec3:
        return self.centre_bottom

    def is_inside(self, p: Vec3) -> bool:
        """判断点 p 是否在圆柱体内（含边界）"""
        # 高度检查
        if p[2] < self.centre_bottom[2] or p[2] > self.centre_bottom[2] + self.height:
            return False
        # 水平径向检查
        dx = p[0] - self.centre_bottom[0]
        dy = p[1] - self.centre_bottom[1]
        return (dx**2 + dy**2) <= self.radius**2