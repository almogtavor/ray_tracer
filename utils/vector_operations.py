from __future__ import annotations

import numpy as np

EPSILON: float = 1e-5 # As disscussed in the lecture - small epsilon value for floating point comparisons (to handle edge cases in vector normalization)


def vector_length(v: np.ndarray) -> float: #Euclidean length (magnitude) of a vector
    vector_array = np.asarray(v, dtype=float)
    return float(np.linalg.norm(vector_array))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    vector_array = np.asarray(v, dtype=float)
    magnitude = np.linalg.norm(vector_array)
    if magnitude < EPSILON:
        raise ValueError("Cannot normalize near-zero vector")
    return vector_array / magnitude


def vector_dot(a: np.ndarray, b: np.ndarray) -> float:
    vector_a = np.asarray(a, dtype=float)
    vector_b = np.asarray(b, dtype=float)
    return float(np.dot(vector_a, vector_b))


def vector_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray: #cross product of two vectors (3D)
    vector_a = np.asarray(a, dtype=float)
    vector_b = np.asarray(b, dtype=float)
    return np.cross(vector_a, vector_b)


def reflect_vector(I: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Calculates the reflection vector R given the incident vector I and surface normal N.
       Assumes I points toward the surface"""
    vector_I = np.asarray(I, dtype=float)
    vector_N = np.asarray(N, dtype=float)
    return vector_I - 2.0 * vector_dot(vector_I, vector_N) * vector_N


def clamp_color01(color_rgb: np.ndarray) -> np.ndarray:
    """Clamps an RGB color array to the range [0.0, 1.0]."""
    color_array = np.asarray(color_rgb, dtype=float)
    return np.clip(color_array, 0.0, 1.0)


def color_to_uint8(color_rgb: np.ndarray) -> np.ndarray:
    """Converts a floating-point RGB color array (clamped to [0, 1]) to 8-bit integer [0, 255]."""
    clamped_color = clamp_color01(color_rgb)
    return (clamped_color * 255.0 + 0.5).astype(np.uint8) # 0.5 before conversion ensures correct rounding
