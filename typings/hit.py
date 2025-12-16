from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Hit:
    t: float
    point: np.ndarray
    normal: np.ndarray
    material_index: int
