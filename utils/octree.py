from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from typings.hit import Hit
from typings.ray import Ray
from utils.bvh import AABB, BVHPrimitive
from utils.vector_operations import EPSILON


@dataclass(slots=True)
class OctreeConfig:
    max_depth: int = 8
    max_primitives_per_leaf: int = 4


def _bounds_overlap(a: AABB, b: AABB) -> bool:
    return not (
        (a.max[0] < b.min[0] or a.min[0] > b.max[0])
        or (a.max[1] < b.min[1] or a.min[1] > b.max[1])
        or (a.max[2] < b.min[2] or a.min[2] > b.max[2])
    )


class OctreeNode:
    __slots__ = ("bounds", "primitives", "children", "is_leaf")

    def __init__(
        self,
        bounds: AABB,
        primitives: Sequence[BVHPrimitive],
        depth: int,
        config: OctreeConfig,
    ) -> None:
        self.bounds = bounds
        self.primitives: List[BVHPrimitive] = []
        self.children: List[OctreeNode] = []

        should_split = (
            depth < config.max_depth
            and len(primitives) > config.max_primitives_per_leaf
        )
        if not should_split:
            self.primitives = list(primitives)
            self.is_leaf = True
            return

        child_bounds = self._subdivide(bounds)
        child_buckets: List[List[BVHPrimitive]] = [[] for _ in range(8)]

        for primitive in primitives:
            for idx, child_bound in enumerate(child_bounds):
                if _bounds_overlap(primitive.bounds, child_bound):
                    child_buckets[idx].append(primitive)

        for idx, bucket in enumerate(child_buckets):
            if not bucket:
                continue
            child = OctreeNode(child_bounds[idx], bucket, depth + 1, config)
            self.children.append(child)

        if not self.children:
            # Could not split further
            self.primitives = list(primitives)
            self.is_leaf = True
        else:
            self.is_leaf = False

    def intersect(self, ray: Ray, t_max: float = float("inf"), best_hit: Hit | None = None) -> Hit | None:
        max_distance = min(t_max, best_hit.t if best_hit else t_max)
        bounds_interval = self.bounds.hit(ray, EPSILON, max_distance)
        if bounds_interval is None:
            return best_hit

        if self.is_leaf:
            for primitive in self.primitives:
                hit = primitive.surface.intersect(ray)
                if hit is None:
                    continue
                if hit.t <= EPSILON:
                    continue
                if hit.t >= max_distance:
                    continue
                best_hit = hit
                max_distance = hit.t
            return best_hit

        children_hits: List[Tuple[float, OctreeNode]] = []
        for child in self.children:
            child_interval = child.bounds.hit(ray, EPSILON, max_distance)
            if child_interval is not None:
                children_hits.append((child_interval[0], child))

        children_hits.sort(key=lambda entry: entry[0])
        for _, child in children_hits:
            best_hit = child.intersect(ray, max_distance, best_hit)
            if best_hit is not None:
                max_distance = min(max_distance, best_hit.t)
        return best_hit

    def _subdivide(self, bounds: AABB) -> List[AABB]:
        min_point = bounds.min
        max_point = bounds.max
        center = (min_point + max_point) * 0.5
        children = []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    child_min = np.array(
                        [
                            min_point[0] if dx == 0 else center[0],
                            min_point[1] if dy == 0 else center[1],
                            min_point[2] if dz == 0 else center[2],
                        ],
                        dtype=float,
                    )
                    child_max = np.array(
                        [
                            center[0] if dx == 0 else max_point[0],
                            center[1] if dy == 0 else max_point[1],
                            center[2] if dz == 0 else max_point[2],
                        ],
                        dtype=float,
                    )
                    children.append(AABB(child_min, child_max))
        return children


class Octree:
    def __init__(self, primitives: Sequence[BVHPrimitive], config: OctreeConfig | None = None) -> None:
        if not primitives:
            raise ValueError("Octree requires at least one primitive")
        self.config = config or OctreeConfig()
        bounds = AABB.from_boxes([primitive.bounds for primitive in primitives])
        self.root = OctreeNode(bounds, primitives, 0, self.config)

    def intersect(self, ray: Ray, t_max: float = float("inf"), best_hit: Hit | None = None) -> Hit | None:
        return self.root.intersect(ray, t_max, best_hit)
