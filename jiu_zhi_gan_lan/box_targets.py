import numpy as np
from dataclasses import dataclass
from core import Entity, Vec3

# ---------- 目标 ----------

@dataclass
class BoxTarget(Entity):
    centre_bottom: Vec3 = np.array([0.0, 200.0, 0.0])         # 底面圆心坐标
    radius: float = 7.0          # 半径 (m)
    height: float = 10.0         # 高度 (m)
    dead: bool = False

    def is_inside(self, p: Vec3) -> bool:
        """判断点 p 是否在圆柱体内（含边界）"""
        # 高度检查
        if p[2] < self.centre_bottom[2] or p[2] > self.centre_bottom[2] + self.height:
            return False
        # 水平径向检查
        dx = p[0] - self.centre_bottom[0]
        dy = p[1] - self.centre_bottom[1]
        return (dx**2 + dy**2) <= self.radius**2