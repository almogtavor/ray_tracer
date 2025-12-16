import numpy as np


class Material:
    def __init__(
        self,
        diffuse_color: np.ndarray,
        specular_color: np.ndarray,
        reflection_color: np.ndarray,
        shininess: float,
        transparency: float,
    ) -> None:
        self.diffuse_color: np.ndarray = np.asarray(diffuse_color, dtype=float)
        self.specular_color: np.ndarray = np.asarray(specular_color, dtype=float)
        self.reflection_color: np.ndarray = np.asarray(reflection_color, dtype=float)
        self.shininess: float = float(shininess)
        self.transparency: float = float(transparency)
