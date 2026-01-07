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

from utils.bvh import BVHNode, BVHPrimitive
from utils.octree import Octree, OctreeConfig
from utils.light_buffer import LightBuffer, LightBufferConfig
from utils.vector_operations import clamp_color01, color_to_uint8
from utils.vector_operations import normalize_vector, vector_dot, reflect_vector, vector_cross, EPSILON
from typings.hit import Hit
from typings.ray import Ray

class AccelerationType(str, Enum):
    OCTREE = "octree"


@dataclass(slots=True)
class AccelerationSettings:
    structure: AccelerationType = AccelerationType.OCTREE
    octree_max_depth: int = 8
    octree_leaf_size: int = 4
    use_light_buffer: bool = True  # Enable light buffer for shadow ray acceleration
    light_buffer_cells_per_face: int = 16


_ACCELERATOR: BVHNode | Octree | None = None
_PLANE_SURFACES: List[InfinitePlane] = []
_LIGHT_BUFFERS: dict[int, LightBuffer] = {}  # Maps light id() to LightBuffer
_CACHED_SURFACES_ID: int | None = None
_CACHED_SETTINGS: AccelerationSettings | None = None
_ACCEL_SETTINGS = AccelerationSettings()
_PROFILE_COUNTERS = {
    "ray_intersections": 0,
    "shadow_rays": 0,
    "shadow_samples": 0,
    "light_buffer_hits": 0,
    "light_buffer_objects_tested": 0,
}


def reset_profile_counters() -> None:
    for key in _PROFILE_COUNTERS:
        _PROFILE_COUNTERS[key] = 0


def get_profile_counters() -> dict:
    return dict(_PROFILE_COUNTERS)


def configure_acceleration(settings: AccelerationSettings) -> None:
    global _ACCEL_SETTINGS
    _ACCEL_SETTINGS = settings


def build_surface_acceleration(
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
    settings: AccelerationSettings | None = None,
    lights: List[Light] | None = None,
) -> None:
    """
    Build and cache the selected acceleration structure for bounded primitives.
    Infinite planes are excluded and handled separately.

    If lights are provided and light buffers are enabled, also builds light buffers
    for shadow ray acceleration.
    """
    global _ACCELERATOR, _PLANE_SURFACES, _LIGHT_BUFFERS, _CACHED_SURFACES_ID, _CACHED_SETTINGS

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
    else:
        octree_config = OctreeConfig(
            max_depth=applied_settings.octree_max_depth,
            max_primitives_per_leaf=applied_settings.octree_leaf_size,
        )
        _ACCELERATOR = Octree(primitives, octree_config)

    _PLANE_SURFACES = planes

    # Build light buffers if enabled and lights provided
    _LIGHT_BUFFERS = {}
    if applied_settings.use_light_buffer and lights is not None:
        light_buffer_config = LightBufferConfig(
            cells_per_face=applied_settings.light_buffer_cells_per_face
        )
        for light in lights:
            _LIGHT_BUFFERS[id(light)] = LightBuffer(light, surfaces, light_buffer_config)

    _CACHED_SURFACES_ID = id(surfaces)
    _CACHED_SETTINGS = AccelerationSettings(
        structure=applied_settings.structure,
        octree_max_depth=applied_settings.octree_max_depth,
        octree_leaf_size=applied_settings.octree_leaf_size,
        use_light_buffer=applied_settings.use_light_buffer,
        light_buffer_cells_per_face=applied_settings.light_buffer_cells_per_face,
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
    _PROFILE_COUNTERS["ray_intersections"] += 1
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
    light: Light,
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
    sample_position: np.ndarray | None = None,
) -> bool:
    """Check if a shadow ray from typings.hit_point toward light is blocked by any surface."""
    _PROFILE_COUNTERS["shadow_rays"] += 1
    shadow_origin = hit_point + surface_normal * EPSILON # to avoid shadow acne
    light_position = light.position if sample_position is None else sample_position
    to_light = light_position - shadow_origin
    distance_to_light = float(np.linalg.norm(to_light))
    if distance_to_light < EPSILON: # we can immediately return not occluded if we're at the light source
        return False
    shadow_direction = to_light / distance_to_light
    shadow_ray = Ray(origin=shadow_origin, direction=shadow_direction)

    use_light_buffer = sample_position is None
    if use_light_buffer:
        light_id = id(light)
        if light_id in _LIGHT_BUFFERS:
            _PROFILE_COUNTERS["light_buffer_hits"] += 1
            light_buffer = _LIGHT_BUFFERS[light_id]
            candidate_surfaces = light_buffer.query(hit_point)
            _PROFILE_COUNTERS["light_buffer_objects_tested"] += len(candidate_surfaces)

            # Test only candidate surfaces from light buffer
            for surface in candidate_surfaces:
                hit = surface.intersect(shadow_ray)
                if hit is not None and EPSILON < hit.t < distance_to_light:
                    return True
            return False

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
    if shadow_rays_root <= 1 or light.radius <= EPSILON:
        # No soft shadows or point-light approximation.
        if is_occluded(hit_point, surface_normal, light, surfaces):
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
    from utils.vector_operations import UNIT_X, UNIT_Y
    if abs(light_direction[0]) < 0.9:
        up = UNIT_X
    else:
        up = UNIT_Y

    plane_u = normalize_vector(vector_cross(light_direction, up))
    plane_v = vector_cross(light_direction, plane_u)

    unoccluded_count = 0
    
    # We use Stratified Sampling (Jittering) on a grid to avoid banding artifacts.
    # We treat light.radius as the width of the square light source.
    # Grid size is shadow_rays_root x shadow_rays_root.
    
    grid_size = shadow_rays_root
    total_rays = grid_size * grid_size
    step_size = light.radius / grid_size
    
    # Start from bottom-left corner of the light square (centered at light.position)
    start_u = -light.radius / 2.0
    start_v = -light.radius / 2.0

    for i in range(grid_size):
        for j in range(grid_size):
            _PROFILE_COUNTERS["shadow_samples"] += 1
            
            # Random offset within the grid cell [0, 1)
            r1 = np.random.random()
            r2 = np.random.random()
            
            u_offset = start_u + (i + r1) * step_size
            v_offset = start_v + (j + r2) * step_size

            sample_position = light.position + plane_u * u_offset + plane_v * v_offset

            if not is_occluded(hit_point, surface_normal, light, surfaces, sample_position=sample_position):
                unoccluded_count += 1

    return float(unoccluded_count) / float(total_rays)
