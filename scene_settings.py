import numpy as np


class SceneSettings:
    def __init__(self, background_color: np.ndarray, root_number_shadow_rays: float, max_recursions: float) -> None:
        self.background_color: np.ndarray = np.asarray(background_color, dtype=float)
        self.root_number_shadow_rays: int = int(root_number_shadow_rays)
        self.max_recursions: int = int(max_recursions)
