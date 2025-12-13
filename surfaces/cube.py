import numpy as np


class Cube:
    def __init__(self, position: np.ndarray, scale: float, material_index: int) -> None:
        self.position: np.ndarray = np.asarray(position, dtype=float)
        self.scale: float = float(scale)
        self.material_index: int = int(material_index)
