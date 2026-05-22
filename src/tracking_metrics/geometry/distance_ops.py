from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tracking_metrics.data.boxes3d import Box3D


def center_distance_3d(a: Box3D, b: Box3D) -> float:
    """Euclidean distance between 3D box centers."""
    ac = np.array(a.center, dtype=np.float64)
    bc = np.array(b.center, dtype=np.float64)
    return float(np.linalg.norm(ac - bc))


def center_distance_matrix_3d(
    boxes_a: Sequence[Box3D],
    boxes_b: Sequence[Box3D],
) -> np.ndarray:
    n, m = len(boxes_a), len(boxes_b)
    matrix = np.zeros((n, m), dtype=np.float64)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            matrix[i, j] = center_distance_3d(a, b)
    return matrix
