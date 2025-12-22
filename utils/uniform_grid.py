from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from typings.hit import Hit
from typings.ray import Ray
from utils.bvh import AABB, BVHPrimitive
from utils.vector_operations import EPSILON


@dataclass(slots=True)
class GridConfig:
    max_cells_per_axis: int = 64


class UniformGrid:
    """
    Regular grid (voxel) acceleration structure. Primitives are placed inside
    every voxel they overlap and rays advance using a 3D DDA to only test
    objects stored in the currently traversed voxel.
    """

    def __init__(self, primitives: Sequence[BVHPrimitive], config: GridConfig | None = None) -> None:
        if not primitives:
            raise ValueError("UniformGrid requires at least one primitive")
        self.config = config or GridConfig()
        self.bounds = AABB.from_boxes([primitive.bounds for primitive in primitives])
        self.resolution = self._compute_resolution(primitives)
        self.cell_size = self._compute_cell_size()
        self.cells: List[List[BVHPrimitive]] = [
            [] for _ in range(int(self.resolution[0] * self.resolution[1] * self.resolution[2]))
        ]
        self._populate(primitives)

    def intersect(self, ray: Ray, t_max: float = float("inf"), best_hit: Hit | None = None) -> Hit | None:
        max_distance = min(t_max, best_hit.t if best_hit else t_max)
        bounds_interval = self.bounds.hit(ray, EPSILON, max_distance)
        if bounds_interval is None:
            return best_hit

        t_enter, t_exit = bounds_interval
        if t_exit < 0.0:
            return best_hit

        start_t = max(t_enter, 0.0)
        current_point = ray.origin + ray.direction * start_t
        cell_indices = self._point_to_cell_indices(current_point)

        steps, t_next, delta_t = self._compute_traversal_parameters(ray.direction, current_point, start_t, cell_indices)
        if steps is None:
            # Degenerate grid: fall back to testing every primitive
            return self._brute_force(ray, max_distance, best_hit)

        visited_primitives: set[int] = set()
        current_t = start_t

        while self._inside_grid(cell_indices):
            cell_index = self._flatten_index(cell_indices)
            next_axis = int(np.argmin(t_next))
            cell_exit_t = float(t_next[next_axis])

            for primitive in self.cells[cell_index]:
                primitive_id = id(primitive)
                if primitive_id in visited_primitives:
                    continue
                visited_primitives.add(primitive_id)
                hit = primitive.surface.intersect(ray)
                if hit is None:
                    continue
                if hit.t <= EPSILON:
                    continue
                if hit.t >= max_distance:
                    continue
                if hit.t < current_t - EPSILON:
                    continue
                if hit.t > cell_exit_t + EPSILON:
                    continue
                best_hit = hit
                max_distance = hit.t

            next_t = cell_exit_t
            if best_hit is not None and best_hit.t <= next_t:
                break
            if next_t > t_exit:
                break

            cell_indices[next_axis] += steps[next_axis]
            if not self._inside_grid(cell_indices):
                break
            t_next[next_axis] += delta_t[next_axis]
            current_t = next_t

        return best_hit

    # === Build helpers ===

    def _compute_resolution(self, primitives: Sequence[BVHPrimitive]) -> np.ndarray:
        primitive_count = max(len(primitives), 1)
        base_resolution = max(int(round(primitive_count ** (1.0 / 3.0))), 1)
        max_axis_cells = max(1, int(self.config.max_cells_per_axis))
        extent = self.bounds.max - self.bounds.min
        longest_extent = float(np.max(extent))
        if longest_extent < EPSILON:
            return np.ones(3, dtype=int)
        axis_ratios = extent / longest_extent
        resolution = np.maximum(1, np.round(axis_ratios * base_resolution).astype(int))
        resolution = np.minimum(resolution, max_axis_cells)
        return resolution

    def _compute_cell_size(self) -> np.ndarray:
        extent = self.bounds.max - self.bounds.min
        resolution = np.maximum(self.resolution, 1)
        cell_size = np.divide(extent, resolution, out=np.zeros_like(extent), where=resolution != 0)
        cell_size[cell_size < EPSILON] = EPSILON
        return cell_size

    def _populate(self, primitives: Sequence[BVHPrimitive]) -> None:
        for primitive in primitives:
            prim_min = primitive.bounds.min
            prim_max = primitive.bounds.max
            min_indices = self._point_to_cell_indices(prim_min)
            max_indices = self._point_to_cell_indices(prim_max)
            for ix in range(min_indices[0], max_indices[0] + 1):
                for iy in range(min_indices[1], max_indices[1] + 1):
                    for iz in range(min_indices[2], max_indices[2] + 1):
                        flat_index = self._flatten_index((ix, iy, iz))
                        self.cells[flat_index].append(primitive)

    # === Traversal helpers ===

    def _point_to_cell_indices(self, point: np.ndarray) -> np.ndarray:
        indices = np.zeros(3, dtype=int)
        for axis in range(3):
            size = float(self.cell_size[axis])
            if size <= EPSILON:
                indices[axis] = 0
                continue
            normalized = (point[axis] - self.bounds.min[axis]) / size
            indices[axis] = int(np.clip(np.floor(normalized), 0, self.resolution[axis] - 1))
        return indices

    def _compute_traversal_parameters(
        self,
        direction: np.ndarray,
        start_point: np.ndarray,
        start_t: float,
        cell_indices: np.ndarray,
    ) -> Tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        steps = np.zeros(3, dtype=int)
        t_next = np.empty(3, dtype=float)
        delta_t = np.empty(3, dtype=float)

        for axis in range(3):
            dir_component = float(direction[axis])
            size = float(self.cell_size[axis])
            base_boundary = self.bounds.min[axis]
            if size <= EPSILON:
                steps[axis] = 0
                t_next[axis] = float("inf")
                delta_t[axis] = float("inf")
                continue
            if abs(dir_component) < EPSILON:
                steps[axis] = 0
                t_next[axis] = float("inf")
                delta_t[axis] = float("inf")
                continue

            if dir_component > 0.0:
                steps[axis] = 1
                next_boundary = base_boundary + (cell_indices[axis] + 1) * size
            else:
                steps[axis] = -1
                next_boundary = base_boundary + cell_indices[axis] * size

            t_to_boundary = (next_boundary - start_point[axis]) / dir_component
            t_next[axis] = start_t + t_to_boundary
            delta_t[axis] = size / abs(dir_component)

        if np.all(np.isinf(t_next)):
            # Ray direction is almost zero in every component -> skip grid traversal
            return None, t_next, delta_t
        return steps, t_next, delta_t

    def _flatten_index(self, indices: Tuple[int, int, int] | np.ndarray) -> int:
        ix, iy, iz = int(indices[0]), int(indices[1]), int(indices[2])
        return ix + self.resolution[0] * (iy + self.resolution[1] * iz)

    def _inside_grid(self, indices: np.ndarray) -> bool:
        return all(
            0 <= indices[axis] < self.resolution[axis]
            for axis in range(3)
        )

    def _brute_force(self, ray: Ray, t_max: float, best_hit: Hit | None) -> Hit | None:
        max_distance = min(t_max, best_hit.t if best_hit else t_max)
        visited_primitives: set[int] = set()
        for cell in self.cells:
            for primitive in cell:
                primitive_id = id(primitive)
                if primitive_id in visited_primitives:
                    continue
                visited_primitives.add(primitive_id)
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
