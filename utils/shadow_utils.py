from __future__ import annotations
import numpy as np
from typing import List, Union, Dict
from enum import Enum
from dataclasses import dataclass

from utils.bvh import BVHPrimitive
from utils.octree import Octree, OctreeConfig
from utils.light_buffer import LightBuffer, LightBufferConfig
from utils.vector_operations import normalize_vector, vector_dot, vector_cross, reflect_vector, EPSILON
from typings.hit import Hit
from typings.ray import Ray
from typings.light import Light
from surfaces.sphere import Sphere
from surfaces.infinite_plane import InfinitePlane
from surfaces.cube import Cube

class AccelerationType(str, Enum):
    OCTREE = "octree"


@dataclass(slots=True)
class AccelerationSettings:
    structure: AccelerationType = AccelerationType.OCTREE
    octree_max_depth: int = 8
    octree_leaf_size: int = 4
    use_light_buffer: bool = True
    light_buffer_cells_per_face: int = 16


class SceneAccelerator:
    def __init__(
        self,
        surfaces: List[Union[Sphere, InfinitePlane, Cube]],
        settings: AccelerationSettings,
        lights: List[Light] | None = None,
    ) -> None:
        self.settings = settings
        self.planes: List[InfinitePlane] = []
        self.light_buffers: Dict[int, LightBuffer] = {}
        self.counters = {
            "ray_intersections": 0,
            "shadow_rays": 0,
            "shadow_samples": 0,
            "light_buffer_hits": 0,
            "light_buffer_objects_tested": 0,
        }

        primitives: List[BVHPrimitive] = []
        for surface in surfaces:
            if isinstance(surface, InfinitePlane):
                self.planes.append(surface)
            elif isinstance(surface, (Sphere, Cube)):
                primitives.append(BVHPrimitive(surface, surface.aabb()))

        if not primitives:
            self.spatial_accelerator = None
        else:
            config = OctreeConfig(
                max_depth=settings.octree_max_depth,
                max_primitives_per_leaf=settings.octree_leaf_size,
            )
            self.spatial_accelerator = Octree(primitives, config)

        if settings.use_light_buffer and lights:
            lb_config = LightBufferConfig(cells_per_face=settings.light_buffer_cells_per_face)
            for light in lights:
                self.light_buffers[id(light)] = LightBuffer(light, surfaces, lb_config)

    def find_closest_hit(self, ray: Ray, max_distance: float = float("inf")) -> Hit | None:
        self.counters["ray_intersections"] += 1
        best_hit: Hit | None = None

        if self.spatial_accelerator:
            best_hit = self.spatial_accelerator.intersect(ray, t_max=max_distance)
            if best_hit:
                max_distance = best_hit.t

        for plane in self.planes:
            hit = plane.intersect(ray)
            if hit and hit.t < max_distance:
                best_hit = hit
                max_distance = hit.t
        return best_hit

    def is_occluded(
        self,
        hit_point: np.ndarray,
        surface_normal: np.ndarray,
        light: Light,
        sample_position: np.ndarray | None = None,
    ) -> bool:
        self.counters["shadow_rays"] += 1
        shadow_origin = hit_point + surface_normal * EPSILON
        light_pos = light.position if sample_position is None else sample_position
        to_light = light_pos - shadow_origin
        dist = float(np.linalg.norm(to_light))

        if dist < EPSILON:
            return False

        shadow_ray = Ray(origin=shadow_origin, direction=to_light / dist)

        # Light Buffer check
        if sample_position is None:
            lb = self.light_buffers.get(id(light))
            if lb:
                self.counters["light_buffer_hits"] += 1
                candidates = lb.query(hit_point)
                self.counters["light_buffer_objects_tested"] += len(candidates)
                for surface in candidates:
                    hit = surface.intersect(shadow_ray)
                    if hit and EPSILON < hit.t < dist:
                        return True
                return False

        hit = self.find_closest_hit(shadow_ray, max_distance=dist)
        return hit is not None and hit.t < dist

    def compute_soft_shadow_factor(
        self,
        hit_point: np.ndarray,
        surface_normal: np.ndarray,
        light: Light,
        shadow_rays_root: int,
    ) -> float:
        if shadow_rays_root <= 1 or light.radius <= EPSILON:
            return 0.0 if self.is_occluded(hit_point, surface_normal, light) else 1.0

        light_to_point = hit_point - light.position
        dist = float(np.linalg.norm(light_to_point))
        if dist < EPSILON:
            return 1.0

        light_dir = light_to_point / dist

        # Orthonormal basis
        from utils.vector_operations import UNIT_X, UNIT_Y
        up = UNIT_X if abs(light_dir[0]) < 0.9 else UNIT_Y
        plane_u = normalize_vector(vector_cross(light_dir, up))
        plane_v = vector_cross(light_dir, plane_u)

        unoccluded = 0
        step = light.radius / shadow_rays_root
        start_u = start_v = -light.radius / 2.0

        for i in range(shadow_rays_root):
            for j in range(shadow_rays_root):
                self.counters["shadow_samples"] += 1
                u_off = start_u + (i + np.random.random()) * step
                v_off = start_v + (j + np.random.random()) * step
                sample_pos = light.position + plane_u * u_off + plane_v * v_off
                if not self.is_occluded(hit_point, surface_normal, light, sample_pos):
                    unoccluded += 1

        return float(unoccluded) / (shadow_rays_root * shadow_rays_root)

