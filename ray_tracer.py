import argparse
import time
from typing import List, Union

from typings.material import Material
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from typings.light import Light
from utils.shadow_utils import (
    AccelerationSettings,
    SceneAccelerator,
)
from scene_parser import parse_scene_file
from renderer import render_with_full_shading, save_image


def main() -> None:
    parser = argparse.ArgumentParser(description='Almog\'s and Tal\'s great Ray Tracer!')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    parser.add_argument('--octree-depth', type=int, default=8, help='Acceleration Trick Param: Max octree depth')
    parser.add_argument('--octree-leaf-size', type=int, default=4, help='Acceleration Trick Param: Max leaf size')
    parser.add_argument('--no-light-buffer', action='store_true', help='Acceleration Trick Param: Disable light buffer')
    parser.add_argument('--light-buffer-resolution', type=int, default=16, help='Acceleration Trick Param: Light buffer res')
    args = parser.parse_args()

    parse_start = time.perf_counter()
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    print(f"[phase] parse_scene: {time.perf_counter() - parse_start:.2f}s")
    materials: List[Material] = [obj for obj in objects if isinstance(obj, Material)]
    surfaces: List[Union[Sphere, InfinitePlane, Cube]] = [
        obj for obj in objects if isinstance(obj, (Sphere, InfinitePlane, Cube))
    ]
    lights: List[Light] = [obj for obj in objects if isinstance(obj, Light)]
    accel_settings = AccelerationSettings(
        octree_max_depth=args.octree_depth,
        octree_leaf_size=args.octree_leaf_size,
        use_light_buffer=not args.no_light_buffer,
        light_buffer_cells_per_face=args.light_buffer_resolution,
    )

    accel_start = time.perf_counter()
    accelerator = SceneAccelerator(surfaces, accel_settings, lights)
    print(f"[phase] build_accel: {time.perf_counter() - accel_start:.2f}s")

    render_start = time.perf_counter()
    image = render_with_full_shading(
        camera, surfaces, materials, lights,
        scene_settings.background_color,
        scene_settings.root_number_shadow_rays,
        scene_settings.max_recursions,
        args.width, args.height,
        accelerator=accelerator,
        accel_settings=accel_settings,
        build_accel=False,
    )
    print(f"[phase] render: {time.perf_counter() - render_start:.2f}s")

    save_image(image, args.output_image)

    cnt = accelerator.counters
    msg = f"[stats] rays={cnt['ray_intersections']}, shadow_rays={cnt.get('shadow_rays',0)}, samples={cnt.get('shadow_samples',0)}"
    if cnt.get("light_buffer_hits", 0) > 0:
        avg = cnt["light_buffer_objects_tested"] / cnt["light_buffer_hits"]
        msg += f", light_buff_queries={cnt['light_buffer_hits']}, avg_obj/query={avg:.1f}"
    print(msg)


if __name__ == '__main__':
    start = time.time()
    print(f"[timer] Start: {time.strftime('%H:%M:%S', time.localtime(start))}")
    try:
        main()
    finally:
        end = time.time()
        print(f"[timer] End: {time.strftime('%H:%M:%S', time.localtime(end))} (elapsed {end - start:.2f}s)")
