import numpy as np


class Light:
    def __init__(
        self,
        position: np.ndarray,
        color: np.ndarray,
        specular_intensity: float,
        shadow_intensity: float,
        radius: float,
    ) -> None:
        self.position: np.ndarray = np.asarray(position, dtype=float)
        self.color: np.ndarray = np.asarray(color, dtype=float)
        self.specular_intensity: float = float(specular_intensity)
        self.shadow_intensity: float = float(shadow_intensity)
        self.radius: float = float(radius)
