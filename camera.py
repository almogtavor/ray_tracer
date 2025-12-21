import numpy as np

from typings.ray import Ray
from utils.vector_operations import vector_cross, normalize_vector


class Camera:
    def __init__(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        up_vector: np.ndarray,
        screen_distance: float,
        screen_width: float,
    ) -> None:
        self.position = np.asarray(position, dtype=float)
        self.look_at = np.asarray(look_at, dtype=float)
        self.up_vector = np.asarray(up_vector, dtype=float)
        self.screen_distance = float(screen_distance)
        self.screen_width = float(screen_width)

        self._recompute_basis()

    def _recompute_basis(self) -> None:
        """Ray directions and camera axes must have unit length so that distances, projections, and scaling behave correctly.
        right (perpendicular to) forward (perpendicular to) true_up"""
        forward = normalize_vector(self.look_at - self.position) # distance from camera position to look_at point
        right = normalize_vector(vector_cross(forward, self.up_vector)) # horizontal axis
        true_up = vector_cross(right, forward) # vertical axis

        self.forward: np.ndarray = forward
        self.right: np.ndarray = right
        self.true_up: np.ndarray = true_up

    def generate_ray(self, i: int, j: int, W: int, H: int) -> Ray:
        # starts at the camera's position and moves along the camera's forward (or view) direction by the screen_distance
        screen_center = self.position + self.forward * self.screen_distance
        screen_height = self.screen_width * (float(H) / float(W))

        # coordinates on the camera’s image plane (u is horizontal left&right, and v vertical for up&down)
        u = ((float(j) + 0.5) / float(W) - 0.5) * self.screen_width
        v = (0.5 - (float(i) + 0.5) / float(H)) * screen_height

        pixel_point = screen_center + self.right * u + self.true_up * v
        direction = normalize_vector(pixel_point - self.position)
        return Ray(origin=self.position, direction=direction)
