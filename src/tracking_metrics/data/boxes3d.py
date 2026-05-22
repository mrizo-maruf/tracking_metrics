from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box3D:
    """Axis-aligned 3D bounding box with an optional yaw angle.

    center: (x, y, z) in world coordinates
    size: (dx, dy, dz) — all components must be non-negative
    yaw: rotation around the vertical axis in radians (stored, not used by axis-aligned IoU)
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float | None = None

    def volume(self) -> float:
        dx, dy, dz = self.size
        return dx * dy * dz

    def is_valid(self) -> bool:
        return all(s >= 0 for s in self.size)

    def min_corner(self) -> np.ndarray:
        cx, cy, cz = self.center
        dx, dy, dz = self.size
        return np.array([cx - dx / 2, cy - dy / 2, cz - dz / 2], dtype=np.float64)

    def max_corner(self) -> np.ndarray:
        cx, cy, cz = self.center
        dx, dy, dz = self.size
        return np.array([cx + dx / 2, cy + dy / 2, cz + dz / 2], dtype=np.float64)
