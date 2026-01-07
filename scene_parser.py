from typing import List, Tuple, Union
import numpy as np

from camera import Camera
from typings.light import Light
from typings.material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

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
                raise ValueError("unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects
