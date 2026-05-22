from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tracking_metrics.data.boxes3d import Box3D


def box3d_intersection_axis_aligned(a: Box3D, b: Box3D) -> float:
    """Compute intersection volume of two axis-aligned 3D boxes (yaw is ignored)."""
    if not a.is_valid() or not b.is_valid():
        return 0.0

    a_min = a.min_corner()
    a_max = a.max_corner()
    b_min = b.min_corner()
    b_max = b.max_corner()

    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    dims = np.maximum(0.0, inter_max - inter_min)
    return float(dims[0] * dims[1] * dims[2])


def box3d_iou_axis_aligned(a: Box3D, b: Box3D) -> float:
    """Compute axis-aligned 3D IoU between two boxes. Yaw is not used."""
    if not a.is_valid() or not b.is_valid():
        return 0.0

    intersection = box3d_intersection_axis_aligned(a, b)
    if intersection == 0.0:
        return 0.0

    union = a.volume() + b.volume() - intersection
    return float(intersection / union) if union > 0 else 0.0


def box3d_iou_matrix_axis_aligned(
    boxes_a: Sequence[Box3D],
    boxes_b: Sequence[Box3D],
) -> np.ndarray:
    n, m = len(boxes_a), len(boxes_b)
    matrix = np.zeros((n, m), dtype=np.float64)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            matrix[i, j] = box3d_iou_axis_aligned(a, b)
    return matrix
