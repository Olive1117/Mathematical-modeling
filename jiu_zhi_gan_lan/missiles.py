import numpy as np
from core import Entity, Vec3, normalize, norm, Scene
from missile_search import missile_can_see_target

# ---------- 导弹 ----------

class Missile(Entity):
    def __init__(self, id_: int, pos0: Vec3, scene: Scene):
        self.id = id_
        self._pos = pos0.astype(float).copy()
        self._vel = normalize(-pos0) * V
        self.scene = scene
        self.guidance_on = False
        self.dead = False
        self.impact_flt = 7.0  # 7 m 圆靶

    # ---------- 常量 ----------
    V = 300.0  # 导弹常速 300 m/s
    MAX_G = 5.0  # 最大过载 5 g
    g = 9.81
    MAX_OMEGA = MAX_G * g / V  # 最大偏转角速度 (rad/s)
    IR_ON_RANGE = 6000.0  # 距假目标 6000 m 开启红外
    KILL_RANGE = 7.0  # 杀伤半径 7 m

    def pos(self) -> Vec3:
        return self._pos

    def update(self, dt: float) -> None:
        if self.dead:
            return

        # 1. 中段盲飞→6000 m 开启红外
        if not self.ir_on:
            fake_tgt = np.array([0.0, 0.0, 0.0])
            if norm(self._pos - fake_tgt) <= IR_ON_RANGE:
                self.ir_on = True
            else:
                self._vel = normalize(fake_tgt - self._pos) * V

        # 2. 红外阶段：未锁定前持续探测
        if self.ir_on and not self.locked:
            # 假设 scene 提供 nearest_cloud() 和 truth_target
            if missile_can_see_target(self, self.scene.nearest_cloud(self), self.scene.truth_target):
                self.locked = True

        # 3. 末段 5 g 限制转向真目标
        if self.locked:
            tgt = self.scene.truth_pos
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

        # 4. 积分 & 命中
        self._pos += self._vel * dt
        if norm(self._pos - self.scene.truth_pos) <= KILL_RANGE:
            self.dead = True