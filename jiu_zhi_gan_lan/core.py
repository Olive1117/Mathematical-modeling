from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Protocol

# ---------- 基础向量工具 ----------
Vec3 = np.ndarray   # shape (3,)

# 求向量长度
def norm(v: Vec3) -> float:
    return float(np.linalg.norm(v))

# 求单位向量
def normalize(v: Vec3) -> Vec3:
    n = norm(v)
    return v / n if n > 1e-8 else np.zeros(3)

# ---------- 实体协议 ----------
class Entity(Protocol):
    def update(self, dt: float): ...
    def pos(self) -> Vec3: ...
    def is_dead(self) -> bool: ...


# ---------- 场景 ----------
class Scene:
    def __init__(self):
        # entities数据结构 [[目标列表], [导弹列表], [无人机列表]]
        self.entities: List[Entity] = []
        self.log: List[dict] = []
        self.targets: Entity = None
        self.missile: List[Entity] = []
        self.drone: List[Entity] = []
        self.cloud: List[Entity] = []

    def add(self, e: Entity):
        self.entities.append(e)

    def step(self, t: float, dt: float):
        self.targets.update(dt)
        for e in self.missile:
            e.update(dt)
        for e in self.drone:
            e.update(dt)
        for e in self.cloud:
            e.update(dt)
        # TODO 待修改，改成图表可以读取的格式
        # 记录快照
        self.log.append(
            {
                "t": t,
                **{
                    f"{e.__class__.__name__}_{i}_x": e.pos[0]
                    for i, e in enumerate(self.entities)
                },
                **{
                    f"{e.__class__.__name__}_{i}_y": e.pos[1]
                    for i, e in enumerate(self.entities)
                },
                **{
                    f"{e.__class__.__name__}_{i}_z": e.pos[2]
                    for i, e in enumerate(self.entities)
                },
            }
        )

    def is_occluded(self, src: Vec3, dst: Vec3) -> bool:
        """视线段 src->dst 是否被任一 cloud 遮挡"""
        seg = dst - src
        seg_len = norm(seg)
        if seg_len == 0:
            return False
        dir = seg / seg_len
        for c in self.clouds:
            # 球与线段最近距离
            oc = c.centre - src
            t = np.clip(oc.dot(dir), 0, seg_len)
            closest = src + t * dir
            if norm(closest - c.centre) <= c.radius:
                return True
        return False

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)