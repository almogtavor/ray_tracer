from __future__ import annotations
import argparse
import time
from PIL import Image
import numpy as np
from typing import List, Tuple, Union
from camera import Camera
from typings.light import Light
from typings.material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from enum import Enum
from dataclasses import dataclass

from utils.bvh import BVHNode, BVHPrimitive, build_bvh
from utils.uniform_grid import GridConfig, UniformGrid
from utils.octree import Octree, OctreeConfig
from utils.vector_operations import clamp_color01, color_to_uint8
from utils.vector_operations import normalize_vector, vector_dot, reflect_vector, vector_cross, EPSILON
from typings.hit import Hit
from typings.ray import Ray

class AccelerationType(str, Enum):
    BVH = "bvh"
    GRID = "grid"
    OCTREE = "octree"


@dataclass(slots=True)
class AccelerationSettings:
    structure: AccelerationType = AccelerationType.BVH
    grid_cells: int = 64
    octree_max_depth: int = 8
    octree_leaf_size: int = 4


_ACCELERATOR: BVHNode | UniformGrid | Octree | None = None
_PLANE_SURFACES: List[InfinitePlane] = []
_CACHED_SURFACES_ID: int | None = None
_CACHED_SETTINGS: AccelerationSettings | None = None
_ACCEL_SETTINGS = AccelerationSettings()


def configure_acceleration(settings: AccelerationSettings) -> None:
    global _ACCEL_SETTINGS
    _ACCEL_SETTINGS = settings


def build_surface_acceleration(
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
    settings: AccelerationSettings | None = None,
) -> None:
    """
    Build and cache the selected acceleration structure for bounded primitives.
    Infinite planes are excluded and handled separately.
    """
    global _ACCELERATOR, _PLANE_SURFACES, _CACHED_SURFACES_ID, _CACHED_SETTINGS

    if settings is not None:
        configure_acceleration(settings)

    applied_settings = _ACCEL_SETTINGS
    primitives: List[BVHPrimitive] = []
    planes: List[InfinitePlane] = []
    for surface in surfaces:
        if isinstance(surface, InfinitePlane):
            planes.append(surface)
            continue
        if isinstance(surface, (Sphere, Cube)):
            bounds = surface.aabb()
            primitives.append(BVHPrimitive(surface, bounds))

    if not primitives:
        _ACCELERATOR = None
    elif applied_settings.structure == AccelerationType.BVH:
        _ACCELERATOR = build_bvh(primitives)
    elif applied_settings.structure == AccelerationType.GRID:
        grid_config = GridConfig(max_cells_per_axis=applied_settings.grid_cells)
        _ACCELERATOR = UniformGrid(primitives, grid_config)
    elif applied_settings.structure == AccelerationType.OCTREE:
        octree_config = OctreeConfig(
            max_depth=applied_settings.octree_max_depth,
            max_primitives_per_leaf=applied_settings.octree_leaf_size,
        )
        _ACCELERATOR = Octree(primitives, octree_config)
    else:
        raise ValueError(f"Unknown acceleration structure: {applied_settings.structure}")

    _PLANE_SURFACES = planes
    _CACHED_SURFACES_ID = id(surfaces)
    _CACHED_SETTINGS = AccelerationSettings(
        structure=applied_settings.structure,
        grid_cells=applied_settings.grid_cells,
        octree_max_depth=applied_settings.octree_max_depth,
        octree_leaf_size=applied_settings.octree_leaf_size,
    )


def build_surface_bvh(surfaces: List[Union[Sphere, InfinitePlane, Cube]]) -> None:
    """
    Deprecated helper retained for backward compatibility.
    Uses the currently configured acceleration settings (default BVH).
    """
    build_surface_acceleration(surfaces)


def _ensure_acceleration(surfaces: List[Union[Sphere, InfinitePlane, Cube]]) -> None:
    global _CACHED_SURFACES_ID, _CACHED_SETTINGS
    if _CACHED_SURFACES_ID != id(surfaces):
        build_surface_acceleration(surfaces)
        return
    if _CACHED_SETTINGS != _ACCEL_SETTINGS:
        build_surface_acceleration(surfaces)


def find_closest_hit(
    ray: Ray,
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
    max_distance: float = float("inf"),
) -> Hit | None:
    """Find the closest intersection of ray with any surface using the cached BVH."""
    _ensure_acceleration(surfaces)

    best_hit: Hit | None = None
    if _ACCELERATOR is not None:
        best_hit = _ACCELERATOR.intersect(ray, t_max=max_distance)
        if best_hit is not None:
            max_distance = min(max_distance, best_hit.t)

    for plane in _PLANE_SURFACES:
        hit = plane.intersect(ray)
        if hit is None:
            continue
        if hit.t >= max_distance:
            continue
        best_hit = hit
        max_distance = hit.t
    return best_hit


def is_occluded(
    hit_point: np.ndarray,
    surface_normal: np.ndarray,
    light_position: np.ndarray,
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
) -> bool:
    """Check if a shadow ray from typings.hit_point toward light is blocked by any surface."""
    shadow_origin = hit_point + surface_normal * EPSILON # to avoid shadow acne
    to_light = light_position - shadow_origin
    distance_to_light = float(np.linalg.norm(to_light))
    if distance_to_light < EPSILON: # we can immediately return not occluded if we're at the light source
        return False
    shadow_direction = to_light / distance_to_light
    shadow_ray = Ray(origin=shadow_origin, direction=shadow_direction)
    shadow_hit = find_closest_hit(shadow_ray, surfaces, max_distance=distance_to_light)
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

    # Build plane perpendicular to direction from typings.light to hit point
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
            # Cell center offset from typings.light center
            u_offset = -half_width + (col + 0.5) * cell_size
            v_offset = -half_width + (row + 0.5) * cell_size

            # Add random jitter within cell
            u_jitter = (np.random.random() - 0.5) * cell_size
            v_jitter = (np.random.random() - 0.5) * cell_size

            sample_position = light.position + plane_u * (u_offset + u_jitter) + plane_v * (v_offset + v_jitter)

            if not is_occluded(hit_point, surface_normal, sample_position, surfaces):
                unoccluded_count += 1

    return float(unoccluded_count) / float(total_rays)
