import numpy as np

from typings.hit import Hit
from typings.ray import Ray
from utils.spatial_structures import AABB
from utils.vector_operations import EPSILON, normalize_vector, vector_dot


class Sphere:
    def __init__(self, position: np.ndarray, radius: float, material_index: int) -> None:
        self.position: np.ndarray = np.asarray(position, dtype=float)
        self.radius: float = float(radius)
        self.material_index: int = int(material_index)

    def aabb(self) -> AABB:
        radius_vec = np.full(3, self.radius, dtype=float)
        return AABB(self.position - radius_vec, self.position + radius_vec)

    def intersect(self, ray: Ray) -> Hit | None:
        ray_origin = ray.origin
        ray_direction = ray.direction
        sphere_center = self.position

        origin_to_center = ray_origin - sphere_center

        # Optimized for normalized ray direction (dot(d,d) = 1)
        # At^2 + Bt + C = 0 becomes t^2 + Bt + C = 0 when A=1
        b = vector_dot(origin_to_center, ray_direction)
        c = vector_dot(origin_to_center, origin_to_center) - self.radius * self.radius

        # Discriminant = b^2 - c (simplified from b^2 - 4ac when a=1)
        discriminant = b * b - c
        if discriminant < 0.0:
            return None

        sqrt_discriminant = float(np.sqrt(discriminant))

        # Solutions: t = -b ± sqrt(discriminant)
        t_near = -b - sqrt_discriminant
        t_far = -b + sqrt_discriminant

        hit_distance = t_near if t_near > EPSILON else t_far
        if hit_distance <= EPSILON:
            return None

        hit_point = ray_origin + hit_distance * ray_direction
        surface_normal = (hit_point - sphere_center) * (1.0 / self.radius)  # Normalize by dividing by radius
        if vector_dot(surface_normal, ray_direction) > 0.0:
            surface_normal = -surface_normal
        return Hit(t=float(hit_distance), point=hit_point, normal=surface_normal, material_index=self.material_index)
