import numpy as np


class Sphere:
    def __init__(self, position: np.ndarray, radius: float, material_index: int) -> None:
        self.position: np.ndarray = np.asarray(position, dtype=float)
        self.radius: float = float(radius)
        self.material_index: int = int(material_index)
