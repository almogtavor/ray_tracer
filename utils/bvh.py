from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from hit import Hit
from ray import Ray
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
class BVHPrimitive:
    surface: object
    bounds: AABB

    @property
    def centroid(self) -> np.ndarray:
        return self.bounds.centroid


class BVHNode:
    __slots__ = ("bounds", "left", "right", "primitives")

    def __init__(self, primitives: Sequence[BVHPrimitive], max_leaf_size: int = 2) -> None:
        if not primitives:
            raise ValueError("BVHNode requires at least one primitive")
        self.left: BVHNode | None = None
        self.right: BVHNode | None = None
        self.primitives: List[BVHPrimitive] = []
        self.bounds = AABB.from_boxes([primitive.bounds for primitive in primitives])

        if len(primitives) <= max_leaf_size:
            self.primitives = list(primitives)
            return

        extent = self.bounds.max - self.bounds.min
        split_axis = int(np.argmax(extent))
        sorted_primitives = sorted(primitives, key=lambda primitive: primitive.centroid[split_axis])
        mid = len(sorted_primitives) // 2
        self.left = BVHNode(sorted_primitives[:mid], max_leaf_size)
        self.right = BVHNode(sorted_primitives[mid:], max_leaf_size)

    def is_leaf(self) -> bool:
        return not self.left and not self.right

    def intersect(self, ray: Ray, t_max: float = float("inf"), best_hit: Hit | None = None) -> Hit | None:
        max_distance = min(t_max, best_hit.t if best_hit else t_max)
        hit_interval = self.bounds.hit(ray, EPSILON, max_distance)
        if hit_interval is None:
            return best_hit

        if self.is_leaf():
            for primitive in self.primitives:
                hit = primitive.surface.intersect(ray)
                if hit is None:
                    continue
                if hit.t <= EPSILON:
                    continue
                if hit.t >= max_distance:
                    continue
                if best_hit is None or hit.t < best_hit.t:
                    best_hit = hit
                    max_distance = hit.t
            return best_hit

        children: List[Tuple[float, BVHNode]] = []
        for child in (self.left, self.right):
            if child is None:
                continue
            child_interval = child.bounds.hit(ray, EPSILON, max_distance)
            if child_interval is not None:
                children.append((child_interval[0], child))

        children.sort(key=lambda entry: entry[0])
        for _, child in children:
            best_hit = child.intersect(ray, max_distance, best_hit)
            if best_hit is not None:
                max_distance = min(max_distance, best_hit.t)
        return best_hit


def build_bvh(primitives: Iterable[BVHPrimitive], max_leaf_size: int = 2) -> BVHNode | None:
    primitive_list = [primitive for primitive in primitives]
    if not primitive_list:
        return None
    return BVHNode(primitive_list, max_leaf_size)
