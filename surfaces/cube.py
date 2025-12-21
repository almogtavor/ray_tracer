from __future__ import annotations

import numpy as np

from typings.hit import Hit
from typings.ray import Ray
from utils.vector_operations import EPSILON, vector_dot


class Cube:
    def __init__(self, position: np.ndarray, scale: float, material_index: int) -> None:
        self.position: np.ndarray = np.asarray(position, dtype=float)
        self.scale: float = float(scale)
        self.material_index: int = int(material_index)

    def intersect(self, ray: Ray) -> Hit | None:
        ray_origin = ray.origin
        ray_direction = ray.direction
        half_scale = 0.5 * self.scale
        box_min = self.position - half_scale
        box_max = self.position + half_scale
        t_entry = -float("inf")
        t_exit = float("inf")

        for axis in range(3):
            if abs(ray_direction[axis]) < EPSILON:
                if ray_origin[axis] < box_min[axis] or ray_origin[axis] > box_max[axis]:
                    return None
                continue

            inverse_direction = 1.0 / float(ray_direction[axis])
            t_near = (float(box_min[axis]) - float(ray_origin[axis])) * inverse_direction
            t_far = (float(box_max[axis]) - float(ray_origin[axis])) * inverse_direction
            if t_near > t_far:
                t_near, t_far = t_far, t_near

            t_entry = max(t_entry, t_near)
            t_exit = min(t_exit, t_far)
            if t_exit < t_entry:
                return None

        if t_exit <= EPSILON:
            return None

        hit_distance = t_entry if t_entry > EPSILON else t_exit
        if hit_distance <= EPSILON:
            return None

        hit_point = ray_origin + hit_distance * ray_direction

        # Determine face normal by which coordinate is closest to a slab.
        local_position = hit_point - self.position
        face_distances = np.abs(np.abs(local_position) - half_scale)
        closest_axis = int(np.argmin(face_distances))

        surface_normal = np.zeros(3, dtype=float)
        surface_normal[closest_axis] = 1.0 if local_position[closest_axis] >= 0 else -1.0

        if vector_dot(surface_normal, ray_direction) > 0.0:
            surface_normal = -surface_normal

        return Hit(t=float(hit_distance), point=hit_point, normal=surface_normal, material_index=self.material_index)
