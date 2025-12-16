from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray
