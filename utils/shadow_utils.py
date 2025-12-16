import argparse
import time
from PIL import Image
import numpy as np
from typing import List, Tuple, Union

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from utils.vector_operations import clamp_color01, color_to_uint8
from utils.vector_operations import normalize_vector, vector_dot, reflect_vector, vector_cross, EPSILON

from typings.hit import Hit
from typings.ray import Ray


def find_closest_hit(
    ray: Ray,
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
) -> Hit | None:
    """Find the closest intersection of ray with any surface."""
    best_hit: Hit | None = None
    for surface in surfaces:
        hit = surface.intersect(ray)
        if hit is None:
            continue
        if best_hit is None or hit.t < best_hit.t:
            best_hit = hit
    return best_hit


def is_occluded(
    hit_point: np.ndarray,
    surface_normal: np.ndarray,
    light_position: np.ndarray,
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
) -> bool:
    """Check if a shadow ray from hit_point toward light is blocked by any surface."""
    shadow_origin = hit_point + surface_normal * EPSILON # to avoid shadow acne
    to_light = light_position - shadow_origin
    distance_to_light = float(np.linalg.norm(to_light))
    if distance_to_light < EPSILON: # we can immediately return not occluded if we're at the light source
        return False
    shadow_direction = to_light / distance_to_light
    shadow_ray = Ray(origin=shadow_origin, direction=shadow_direction)
    shadow_hit = find_closest_hit(shadow_ray, surfaces)
    return shadow_hit is not None and shadow_hit.t < distance_to_light


def compute_soft_shadow_factor(
    hit_point: np.ndarray,
    surface_normal: np.ndarray,
    light: Light,
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
    shadow_rays_root: int,
) -> float:
    """
    Computes the soft shadow fraction out of a NxN sampling grid on a plane that is perpendicular to the light direction.
    Returns the fraction of un-occluded rays (0.0 means fully shadowed wheares 1.0 means fully lit).
    """
    if shadow_rays_root <= 0:
        # No soft shadows, just check hard shadow
        if is_occluded(hit_point, surface_normal, light.position, surfaces):
            return 0.0
        return 1.0

    # Build plane perpendicular to direction from light to hit point
    light_to_point = hit_point - light.position
    light_distance = float(np.linalg.norm(light_to_point))
    if light_distance < EPSILON:
        return 1.0

    light_direction = light_to_point / light_distance

    # Build orthonormal basis on the plane perpendicular to light_direction
    # Find a vector not parallel to light_direction
    if abs(light_direction[0]) < 0.9:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])

    plane_u = normalize_vector(vector_cross(light_direction, up))
    plane_v = vector_cross(light_direction, plane_u)

    # Rectangle centered at light with width = light.radius
    half_width = light.radius / 2.0
    cell_size = light.radius / shadow_rays_root

    unoccluded_count = 0
    total_rays = shadow_rays_root * shadow_rays_root

    for row in range(shadow_rays_root):
        for col in range(shadow_rays_root):
            # Cell center offset from light center
            u_offset = -half_width + (col + 0.5) * cell_size
            v_offset = -half_width + (row + 0.5) * cell_size

            # Add random jitter within cell
            u_jitter = (np.random.random() - 0.5) * cell_size
            v_jitter = (np.random.random() - 0.5) * cell_size

            sample_position = light.position + plane_u * (u_offset + u_jitter) + plane_v * (v_offset + v_jitter)

            if not is_occluded(hit_point, surface_normal, sample_position, surfaces):
                unoccluded_count += 1

    return float(unoccluded_count) / float(total_rays)
