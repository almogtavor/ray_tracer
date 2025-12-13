import numpy as np


class InfinitePlane:
    def __init__(self, normal: np.ndarray, offset: float, material_index: int) -> None:
        self.normal: np.ndarray = np.asarray(normal, dtype=float)
        self.offset: float = float(offset)
        self.material_index: int = int(material_index)
