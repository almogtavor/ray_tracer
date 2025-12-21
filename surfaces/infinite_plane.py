import numpy as np

from typings.hit import Hit
from typings.ray import Ray
from utils.vector_operations import EPSILON, normalize_vector, vector_dot


class InfinitePlane:
    def __init__(self, normal: np.ndarray, offset: float, material_index: int) -> None:
        self.normal: np.ndarray = np.asarray(normal, dtype=float)
        self.offset: float = float(offset)
        self.material_index: int = int(material_index)

    def intersect(self, ray: Ray) -> Hit | None:
        ray_origin = ray.origin
        ray_direction = ray.direction
        plane_normal = normalize_vector(self.normal)

        direction_dot_normal = vector_dot(plane_normal, ray_direction)
        if abs(direction_dot_normal) < EPSILON:
            return None

        hit_distance = (self.offset - vector_dot(plane_normal, ray_origin)) / direction_dot_normal
        if hit_distance <= EPSILON:
            return None

        hit_point = ray_origin + hit_distance * ray_direction
        surface_normal = plane_normal
        if vector_dot(surface_normal, ray_direction) > 0.0:
            surface_normal = -surface_normal
        return Hit(t=float(hit_distance), point=hit_point, normal=surface_normal, material_index=self.material_index)
