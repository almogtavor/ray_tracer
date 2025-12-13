import numpy as np

from ray import Ray
from utils import vector_cross, normalize_vector


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
        forward = normalize_vector(self.look_at - self.position)
        right = normalize_vector(vector_cross(forward, self.up_vector))
        true_up = vector_cross(right, forward)

        self.forward: np.ndarray = forward
        self.right: np.ndarray = right
        self.true_up: np.ndarray = true_up

    def generate_ray(self, i: int, j: int, W: int, H: int) -> Ray:
        screen_center = self.position + self.forward * self.screen_distance
        screen_height = self.screen_width * (float(H) / float(W))

        u = ((float(j) + 0.5) / float(W) - 0.5) * self.screen_width
        v = (0.5 - (float(i) + 0.5) / float(H)) * screen_height

        pixel_point = screen_center + self.right * u + self.true_up * v
        direction = normalize_vector(pixel_point - self.position)
        return Ray(origin=self.position, direction=direction)
