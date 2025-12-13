import argparse
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
from utils import clamp_color01, color_to_uint8
from utils import normalize_vector, vector_dot, reflect_vector

from hit import Hit

SceneObject = Union[Material, Sphere, InfinitePlane, Cube, Light]


def parse_scene_file(file_path: str) -> Tuple[Camera | None, SceneSettings | None, List[SceneObject]]:
    objects: List[SceneObject] = []
    camera: Camera | None = None
    scene_settings: SceneSettings | None = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(
                    np.asarray(params[:3], dtype=float),
                    np.asarray(params[3:6], dtype=float),
                    np.asarray(params[6:9], dtype=float),
                    params[9],
                    params[10],
                )
            elif obj_type == "set":
                scene_settings = SceneSettings(np.asarray(params[:3], dtype=float), params[3], params[4])
            elif obj_type == "mtl":
                material = Material(
                    np.asarray(params[:3], dtype=float),
                    np.asarray(params[3:6], dtype=float),
                    np.asarray(params[6:9], dtype=float),
                    params[9],
                    params[10],
                )
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(np.asarray(params[:3], dtype=float), params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(np.asarray(params[:3], dtype=float), params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(np.asarray(params[:3], dtype=float), params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(
                    np.asarray(params[:3], dtype=float),
                    np.asarray(params[3:6], dtype=float),
                    params[6],
                    params[7],
                    params[8],
                )
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array: np.ndarray, output_path: str) -> None:
    image = Image.fromarray(color_to_uint8(image_array))
    image.save(output_path)


def render_intersections_debug(
    camera: Camera,
    surfaces: List[Union[Sphere, InfinitePlane, Cube]],
    materials: List[Material],
    background_color: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
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

            # Prefer material diffuse color if available & otherwise show normal as color.
            mat_idx = best_hit.material_index
            if 1 <= mat_idx <= len(materials):
                image[i, j, :] = materials[mat_idx - 1].diffuse_color
            else:
                image[i, j, :] = (best_hit.normal + 1.0) * 0.5

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


def main() -> None:
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    if camera is None:
        raise ValueError("Scene file is missing a camera ('cam' line)")
    if scene_settings is None:
        raise ValueError("Scene file is missing scene settings ('set' line)")

    materials: List[Material] = [obj for obj in objects if isinstance(obj, Material)]
    surfaces: List[Union[Sphere, InfinitePlane, Cube]] = [
        obj for obj in objects if isinstance(obj, (Sphere, InfinitePlane, Cube))
    ]
    lights: List[Light] = [obj for obj in objects if isinstance(obj, Light)]

    # Render with Phong shading (diffuse + specular). No shadows yet.
    image_array = render_with_phong_shading(
        camera,
        surfaces,
        materials,
        lights,
        scene_settings.background_color,
        args.width,
        args.height,
    )
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
