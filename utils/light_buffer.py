"""
Light Buffer acceleration structure for shadow rays.

Implements the "Light Buffer" technique by discretizing the space of directions
around each light source using a direction cube (like a cubemap). Each cell
stores a sorted list of objects visible from the light source through that cell.

This accelerates shadow ray queries by only testing against objects that could
potentially occlude the light from a given direction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np

from typings.light import Light
from typings.ray import Ray
from surfaces.sphere import Sphere
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from utils.vector_operations import normalize_vector, EPSILON


@dataclass(slots=True)
class LightBufferConfig:
    """Configuration for light buffer construction."""
    cells_per_face: int = 16  # Number of cells along each axis of a cube face
    max_objects_per_cell: int = 10  # Limit objects stored per cell


@dataclass(frozen=True, slots=True)
class OccluderEntry:
    """An object that can occlude light, with its distance from the light."""
    surface: Union[Sphere, Cube, InfinitePlane]
    distance: float  # Distance from light to object (for sorting)


class LightBuffer:
    """
    Direction cube around a light source for accelerating shadow rays.

    The cube has 6 faces (±X, ±Y, ±Z), each subdivided into cells.
    Each cell stores objects visible from the light in that direction,
    sorted by distance from the light.
    """

    def __init__(
        self,
        light: Light,
        surfaces: List[Union[Sphere, Cube, InfinitePlane]],
        config: LightBufferConfig | None = None,
    ) -> None:
        self.light = light
        self.config = config or LightBufferConfig()
        self.cells_per_face = self.config.cells_per_face

        # 6 faces, each with cells_per_face x cells_per_face cells
        total_cells = 6 * self.cells_per_face * self.cells_per_face
        self.cells: List[List[OccluderEntry]] = [[] for _ in range(total_cells)]

        self._build(surfaces)

    def _build(self, surfaces: List[Union[Sphere, Cube, InfinitePlane]]) -> None:
        """Build the light buffer by sampling directions and finding visible objects."""
        # For each cell in the direction cube, sample multiple rays from the light
        # to ensure we capture all objects visible through any part of the cell

        # Sample a 3x3 grid within each cell for better coverage
        samples_per_cell = 3
        for face_idx in range(6):
            for u_idx in range(self.cells_per_face):
                for v_idx in range(self.cells_per_face):
                    cell_idx = self._get_cell_index(face_idx, u_idx, v_idx)

                    # Collect all unique objects visible through this cell
                    visible_object_ids = set()
                    visible_objects_map = {}
                    # Sample multiple directions within the cell
                    for su in range(samples_per_cell):
                        for sv in range(samples_per_cell):
                            # Offset within cell: [-0.4, 0, 0.4] for 3 samples
                            offset_u = (su / (samples_per_cell - 1) - 0.5) if samples_per_cell > 1 else 0.0
                            offset_v = (sv / (samples_per_cell - 1) - 0.5) if samples_per_cell > 1 else 0.0

                            direction = self._cell_to_direction_offset(face_idx, u_idx, v_idx, offset_u, offset_v)

                            # Find objects visible in this direction
                            for obj_entry in self._find_visible_objects(direction, surfaces):
                                obj_id = id(obj_entry.surface)
                                if obj_id not in visible_object_ids:
                                    visible_object_ids.add(obj_id)
                                    visible_objects_map[obj_id] = obj_entry
                                else:
                                    # Keep the closest distance for this object
                                    if obj_entry.distance < visible_objects_map[obj_id].distance:
                                        visible_objects_map[obj_id] = obj_entry

                    # Store sorted by distance (closest first)
                    visible_objects = list(visible_objects_map.values())
                    self.cells[cell_idx] = sorted(visible_objects, key=lambda e: e.distance)
                    # Limit to max_objects_per_cell
                    if len(self.cells[cell_idx]) > self.config.max_objects_per_cell:
                        self.cells[cell_idx] = self.cells[cell_idx][:self.config.max_objects_per_cell]

    def _get_cell_index(self, face_idx: int, u_idx: int, v_idx: int) -> int:
        """Convert (face, u, v) coordinates to flat cell index."""
        return face_idx * (self.cells_per_face * self.cells_per_face) + \
               u_idx * self.cells_per_face + v_idx

    def _cell_to_direction(self, face_idx: int, u_idx: int, v_idx: int) -> np.ndarray:
        """
        Convert cell coordinates to a direction vector (center of cell).

        Maps (face_idx, u_idx, v_idx) to a direction on the unit cube.
        Each face is in [-1, 1] x [-1, 1] space.
        """
        return self._cell_to_direction_offset(face_idx, u_idx, v_idx, 0.0, 0.0)

    def _cell_to_direction_offset(self, face_idx: int, u_idx: int, v_idx: int,
                                  offset_u: float, offset_v: float) -> np.ndarray:
        """
        Convert cell coordinates with offset to a direction vector.

        Args:
            face_idx: Cube face index (0-5)
            u_idx, v_idx: Cell coordinates within face
            offset_u, offset_v: Offset within cell in range [-0.5, 0.5]
        """
        # Normalize u, v to [-1, 1] with offset
        cell_size = 2.0 / self.cells_per_face
        u = (u_idx + 0.5 + offset_u) / self.cells_per_face * 2.0 - 1.0
        v = (v_idx + 0.5 + offset_v) / self.cells_per_face * 2.0 - 1.0

        # Map to cube face
        # Faces: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
        if face_idx == 0:    # +X face
            direction = np.array([1.0, v, -u])
        elif face_idx == 1:  # -X face
            direction = np.array([-1.0, v, u])
        elif face_idx == 2:  # +Y face
            direction = np.array([u, 1.0, -v])
        elif face_idx == 3:  # -Y face
            direction = np.array([u, -1.0, v])
        elif face_idx == 4:  # +Z face
            direction = np.array([u, v, 1.0])
        else:                # -Z face (face_idx == 5)
            direction = np.array([-u, v, -1.0])

        return normalize_vector(direction)

    def _direction_to_cell(self, direction: np.ndarray) -> int:
        """
        Convert a direction vector to a cell index.

        Given a direction from the light, find which cell of the direction cube
        it corresponds to.
        """
        # Find dominant axis (face)
        abs_dir = np.abs(direction)
        max_axis = int(np.argmax(abs_dir))

        if max_axis == 0:  # X dominant
            face_idx = 0 if direction[0] > 0 else 1
            u = -direction[2] / abs_dir[0]
            v = direction[1] / abs_dir[0]
        elif max_axis == 1:  # Y dominant
            face_idx = 2 if direction[1] > 0 else 3
            u = direction[0] / abs_dir[1]
            v = -direction[2] / abs_dir[1] if direction[1] > 0 else direction[2] / abs_dir[1]
        else:  # Z dominant
            face_idx = 4 if direction[2] > 0 else 5
            u = direction[0] / abs_dir[2] if direction[2] > 0 else -direction[0] / abs_dir[2]
            v = direction[1] / abs_dir[2]

        # Convert u, v from [-1, 1] to cell indices
        u_idx = int(np.clip((u + 1.0) * 0.5 * self.cells_per_face, 0, self.cells_per_face - 1))
        v_idx = int(np.clip((v + 1.0) * 0.5 * self.cells_per_face, 0, self.cells_per_face - 1))

        return self._get_cell_index(face_idx, u_idx, v_idx)

    def _find_visible_objects(
        self,
        direction: np.ndarray,
        surfaces: List[Union[Sphere, Cube, InfinitePlane]],
    ) -> List[OccluderEntry]:
        """
        Find objects visible from the light in the given direction.

        Cast a ray from the light in the given direction and collect all objects
        that could potentially occlude light in this direction.
        """
        ray = Ray(origin=self.light.position, direction=direction)
        visible = []

        for surface in surfaces:
            hit = surface.intersect(ray)
            if hit is not None and hit.t > EPSILON:
                visible.append(OccluderEntry(surface=surface, distance=hit.t))

        return visible

    def get_potential_occluders(
        self,
        direction: np.ndarray,
    ) -> List[Union[Sphere, Cube, InfinitePlane]]:
        """
        Get list of objects that could occlude the light in the given direction.

        Args:
            direction: Normalized direction from some point toward the light

        Returns:
            List of surfaces that could potentially occlude the light
        """
        # Direction from point to light, but we need direction from light to point
        light_to_point_direction = -direction

        cell_idx = self._direction_to_cell(light_to_point_direction)

        # Return surfaces from this cell
        return [entry.surface for entry in self.cells[cell_idx]]

    def query(
        self,
        point: np.ndarray,
    ) -> List[Union[Sphere, Cube, InfinitePlane]]:
        """
        Query the light buffer for objects that could occlude the light from point.

        Args:
            point: 3D point from which we're checking visibility to the light

        Returns:
            List of surfaces that could potentially occlude the light from this point
        """
        to_light = self.light.position - point
        distance_to_light = float(np.linalg.norm(to_light))

        if distance_to_light < EPSILON:
            return []

        direction = to_light / distance_to_light
        return self.get_potential_occluders(direction)
