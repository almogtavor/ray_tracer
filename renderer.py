from typing import List, Union
import numpy as np
from PIL import Image

from camera import Camera
from typings.light import Light
from typings.material import Material
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from typings.hit import Hit
from typings.ray import Ray
from utils.shadow_utils import (
    AccelerationSettings,
    build_surface_acceleration,
    compute_soft_shadow_factor,
    find_closest_hit,
)
from utils.vector_operations import (
    clamp_color01,
    color_to_uint8,
    normalize_vector,
    vector_dot,
    reflect_vector,
    EPSILON
)

MIN_RAY_WEIGHT: float = 1e-3

def save_image(image_array: np.ndarray, output_path: str) -> None:
    image = Image.fromarray(color_to_uint8(image_array))
    image.save(output_path)


# Recursive shading with reflection + transparency
def shade(
    ray: Ray,
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
    materials: List[Material],
    lights: List[Light],
    background_color: np.ndarray,
    camera_position: np.ndarray,
    shadow_rays_root: int,
    max_recursion: int,
    depth: int = 0,
    ray_weight: float = 1.0,
) -> np.ndarray:
    """
    Recursive shading function.
    Computes local lighting (diffuse + specular) with soft shadows,
    plus reflection and transparency.
    """
    if depth > max_recursion:
        return background_color

    best_hit = find_closest_hit(ray, surfaces)

    if best_hit is None:
        return background_color

    mat_idx = best_hit.material_index
    if not (1 <= mat_idx <= len(materials)):
        # Fallback: return normal as color
        return (best_hit.normal + 1.0) * 0.5

    material = materials[mat_idx - 1]
    hit_point = best_hit.point
    surface_normal = best_hit.normal
    view_direction = normalize_vector(camera_position - hit_point)
    effective_shadow_root = shadow_rays_root
    if depth > 0 and effective_shadow_root > 1:
        effective_shadow_root = max(1, effective_shadow_root // (2 ** depth))

    # Local lighting with shadows
    local_color = np.zeros(3, dtype=float)

    for light in lights:
        light_direction = normalize_vector(light.position - hit_point)

        # Diffuse component: kd * light_color * max(dot(N, L), 0)
        n_dot_l = vector_dot(surface_normal, light_direction)
        if n_dot_l <= 0.0:
            continue
        diffuse = material.diffuse_color * light.color * n_dot_l

        # Specular component (Phong): ks * light_color * spec_intensity * max(dot(R, V), 0)^shininess
        reflect_direction = reflect_vector(-light_direction, surface_normal)
        r_dot_v = vector_dot(reflect_direction, view_direction)
        specular = np.zeros(3, dtype=float)
        if r_dot_v > 0.0:
            specular = (
                material.specular_color
                * light.color
                * light.specular_intensity
                * (r_dot_v ** material.shininess)
            )

        light_contribution = diffuse + specular

        # Apply soft shadow
        if light.shadow_intensity > 0.0:
            shadow_factor = compute_soft_shadow_factor(
                hit_point, surface_normal, light, surfaces, effective_shadow_root
            )
            # Light intensity multiplier: (1 - shadow_intensity) + shadow_intensity * p
            shadow_multiplier = (1.0 - light.shadow_intensity) + light.shadow_intensity * shadow_factor
            light_contribution *= shadow_multiplier

        local_color += light_contribution

    reflection_term = np.zeros(3, dtype=float)
    reflection_color = material.reflection_color
    reflection_strength = float(np.max(reflection_color))
    if (
        depth < max_recursion
        and reflection_strength > EPSILON
        and ray_weight * reflection_strength > MIN_RAY_WEIGHT
    ):
        reflect_dir = reflect_vector(ray.direction, surface_normal)
        reflect_origin = hit_point + surface_normal * EPSILON
        reflect_ray = Ray(origin=reflect_origin, direction=reflect_dir)
        reflected_color = shade(
            reflect_ray,
            surfaces,
            materials,
            lights,
            background_color,
            camera_position,
            shadow_rays_root,
            max_recursion,
            depth + 1,
            ray_weight * reflection_strength,
        )
        reflection_term = reflected_color * reflection_color

    transparency = material.transparency
    behind_color = np.zeros(3, dtype=float)
    if transparency > EPSILON and depth < max_recursion and ray_weight * transparency > MIN_RAY_WEIGHT:
        # Continue ray forward through the surface
        transmit_origin = hit_point - surface_normal * EPSILON
        transmit_ray = Ray(origin=transmit_origin, direction=ray.direction)
        behind_color = shade(
            transmit_ray,
            surfaces,
            materials,
            lights,
            background_color,
            camera_position,
            shadow_rays_root,
            max_recursion,
            depth + 1,
            ray_weight * transparency,
        )

    # Final formula from spec:
    # out = behind * transparency + local * (1 - transparency) + reflection_term
    final_color = behind_color * transparency + local_color * (1.0 - transparency) + reflection_term

    return final_color


def render_with_full_shading(
    camera: Camera,
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
    materials: List[Material],
    lights: List[Light],
    background_color: np.ndarray,
    shadow_rays_root: int,
    max_recursion: int,
    width: int,
    height: int,
    accel_settings: AccelerationSettings | None = None,
    build_accel: bool = True,
) -> np.ndarray:
    """Render the scene with full ray tracing: Phong shading, soft shadows, reflection, transparency."""
    image = np.zeros((height, width, 3), dtype=float)
    bg = np.asarray(background_color, dtype=float)
    if build_accel:
        build_surface_acceleration(surfaces, accel_settings, lights)

    for i in range(height):
        for j in range(width):
            ray = camera.generate_ray(i, j, width, height)
            color = shade(
                ray,
                surfaces,
                materials,
                lights,
                bg,
                camera.position,
                shadow_rays_root,
                max_recursion,
                depth=0,
            )
            image[i, j, :] = color

    return clamp_color01(image)


def compute_phong_shading(
    hit: Hit,
    material: Material,
    lights: List[Light],
    camera_position: np.ndarray,
) -> np.ndarray:
    """Compute diffuse + specular (Phong) shading for a hit point. No shadows."""
    hit_point = hit.point
    surface_normal = hit.normal
    view_direction = normalize_vector(camera_position - hit_point)

    total_color = np.zeros(3, dtype=float)

    for light in lights:
        light_direction = normalize_vector(light.position - hit_point)

        # Diffuse component: kd * light_color * max(dot(N, L), 0)
        n_dot_l = max(vector_dot(surface_normal, light_direction), 0.0)
        diffuse = material.diffuse_color * light.color * n_dot_l

        # Specular component (Phong): ks * light_color * spec_intensity * max(dot(R, V), 0)^shininess
        # R = reflect(-L, N)
        reflect_direction = reflect_vector(-light_direction, surface_normal)
        r_dot_v = max(vector_dot(reflect_direction, view_direction), 0.0)
        specular = (
            material.specular_color
            * light.color
            * light.specular_intensity
            * (r_dot_v ** material.shininess)
        )

        total_color += diffuse + specular

    return total_color


def render_with_phong_shading(
    camera: Camera,
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
    materials: List[Material],
    lights: List[Light],
    background_color: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Render the scene with Phong shading (diffuse + specular). No shadows yet."""
    image = np.zeros((height, width, 3), dtype=float)
    bg = np.asarray(background_color, dtype=float)

    for i in range(height):
        for j in range(width):
            ray = camera.generate_ray(i, j, width, height)

            best_hit: Hit | None = None
            for surface in surfaces:
                hit = surface.intersect(ray)
                if hit is None:
                    continue
                if best_hit is None or hit.t < best_hit.t:
                    best_hit = hit

            if best_hit is None:
                image[i, j, :] = bg
                continue

            mat_idx = best_hit.material_index
            if 1 <= mat_idx <= len(materials):
                material = materials[mat_idx - 1]
                image[i, j, :] = compute_phong_shading(
                    best_hit, material, lights, camera.position
                )
            else:
                # Fallback: show normal as color
                image[i, j, :] = (best_hit.normal + 1.0) * 0.5

    return clamp_color01(image)
