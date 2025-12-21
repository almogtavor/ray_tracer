from __future__ import annotations

import numpy as np

from typings.hit import Hit
from typings.ray import Ray
from utils.vector_operations import EPSILON, normalize_vector, vector_dot


class Sphere:
    def __init__(self, position: np.ndarray, radius: float, material_index: int) -> None:
        self.position: np.ndarray = np.asarray(position, dtype=float)
        self.radius: float = float(radius)
        self.material_index: int = int(material_index)

    def intersect(self, ray: Ray) -> Hit | None:
        ray_origin = ray.origin
        ray_direction = ray.direction
        sphere_center = self.position

        origin_to_center = ray_origin - sphere_center
        quadratic_a = vector_dot(ray_direction, ray_direction)
        quadratic_b = 2.0 * vector_dot(origin_to_center, ray_direction)
        quadratic_c = vector_dot(origin_to_center, origin_to_center) - self.radius * self.radius

        discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
        if discriminant < 0.0:
            return None

        sqrt_discriminant = float(np.sqrt(discriminant))
        inverse_2a = 1.0 / (2.0 * quadratic_a)

        t_near = (-quadratic_b - sqrt_discriminant) * inverse_2a
        t_far = (-quadratic_b + sqrt_discriminant) * inverse_2a
        if t_near > t_far:
            t_near, t_far = t_far, t_near

        hit_distance = t_near if t_near > EPSILON else t_far
        if hit_distance <= EPSILON:
            return None

        hit_point = ray_origin + hit_distance * ray_direction
        surface_normal = normalize_vector(hit_point - sphere_center)
        if vector_dot(surface_normal, ray_direction) > 0.0:
            surface_normal = -surface_normal
        return Hit(t=float(hit_distance), point=hit_point, normal=surface_normal, material_index=self.material_index)
