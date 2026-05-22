from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tracking_metrics.data.boxes import Box2D


def box2d_iou(a: Box2D, b: Box2D) -> float:
    if not a.is_valid() or not b.is_valid():
        return 0.0

    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    if intersection == 0.0:
        return 0.0

    union = a.area() + b.area() - intersection
    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def box2d_iou_matrix(
    boxes_a: Sequence[Box2D],
    boxes_b: Sequence[Box2D],
) -> np.ndarray:
    n, m = len(boxes_a), len(boxes_b)
    matrix = np.zeros((n, m), dtype=np.float64)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            matrix[i, j] = box2d_iou(a, b)
    return matrix
