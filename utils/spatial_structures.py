from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from typings.hit import Hit
from typings.ray import Ray
from utils.vector_operations import EPSILON


class AABB:
    """Axis-aligned bounding box."""

    __slots__ = ("min", "max")

    def __init__(self, min_point: np.ndarray, max_point: np.ndarray) -> None:
        self.min = np.asarray(min_point, dtype=float)
        self.max = np.asarray(max_point, dtype=float)

    @staticmethod
    def from_boxes(boxes: Sequence["AABB"]) -> "AABB":
        min_points = np.array([box.min for box in boxes])
        max_points = np.array([box.max for box in boxes])
        return AABB(np.min(min_points, axis=0), np.max(max_points, axis=0))

    @property
    def centroid(self) -> np.ndarray:
        return (self.min + self.max) * 0.5

    def hit(self, ray: Ray, t_min: float = EPSILON, t_max: float = float("inf")) -> Tuple[float, float] | None:
        ray_origin = ray.origin
        ray_direction = ray.direction
        for axis in range(3):
            direction_component = float(ray_direction[axis])
            origin_component = float(ray_origin[axis])
            if abs(direction_component) < EPSILON:
                if origin_component < self.min[axis] or origin_component > self.max[axis]:
                    return None
                continue

            inv_d = 1.0 / direction_component
            t0 = (self.min[axis] - origin_component) * inv_d
            t1 = (self.max[axis] - origin_component) * inv_d
            if t0 > t1:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_max < t_min:
                return None
        return t_min, t_max


@dataclass(slots=True)
class Primitive:
    surface: object
    bounds: AABB

    @property
    def centroid(self) -> np.ndarray:
        return self.bounds.centroid

