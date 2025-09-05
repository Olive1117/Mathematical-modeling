import numpy as np
from core import Entity, Vec3, normalize, norm, Scene
from missile_search import missile_can_see_target

# ---------- 导弹 ----------

# ---------- 常量 ----------
V = 300.0  # 导弹常速 300 m/s
MAX_G = 5.0  # 最大过载 5 g
g = 9.81
MAX_OMEGA = MAX_G * g / V  # 最大偏转角速度 (rad/s)
IR_ON_RANGE = 6000.0  # 距假目标 6000 m 开启红外
KILL_RANGE = 7.0  # 杀伤半径 7 m


class Missile(Entity):
    __slots__ = ("id", "_pos", "_vel", "scene", "ir_on", "locked", "dead",
                 "blocked_timer", "prev_blocked")

    def __init__(self, id_: int, pos0: Vec3, scene: Scene):
        self.id = id_
        self._pos = pos0.astype(float).copy()
        self._vel = normalize(-pos0) * V
        self.scene = scene
        self.ir_on = False
        self.locked = False
        self.dead = False
        # >>> 新增：遮挡计时
        self.blocked_timer = 0.0  # 累积被遮时长（s）
        self.prev_blocked = False  # 上一帧是否被遮


    def pos(self) -> Vec3:
        return self._pos

    # >>> 可选：外部读取被遮时长
    def get_blocked_time(self) -> float:
        return self.blocked_timer

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

        # 2. 红外阶段：未锁定前持续探测 + 遮挡计时
        if self.ir_on and not self.locked:
            cloud = self.scene.cloud[0]
            target = self.scene.targets[0]
            now_blocked = False if cloud is None else not missile_can_see_target(self, cloud, target)
            # 累积计时
            if now_blocked:
                self.blocked_timer += dt
            # 可选：刚进入/离开遮挡时可触发事件
            self.prev_blocked = now_blocked

            # 发现目标（未被遮）→ 锁定
            if not now_blocked:
                self.locked = True

        # 3. 末段 5 g 限制转向真目标
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

        # 4. 积分 & 命中
        self._pos += self._vel * dt
        if norm(self._pos - self.scene.targets[0].centre_bottom) <= KILL_RANGE:
            self.dead = True

