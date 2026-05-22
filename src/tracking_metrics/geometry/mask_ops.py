from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tracking_metrics.data.masks import Mask2D


def _to_binary(m: Mask2D | np.ndarray) -> np.ndarray:
    if isinstance(m, np.ndarray):
        return m.astype(bool)
    return m.to_binary()


def _check_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Mask shapes must be identical for IoU computation, "
            f"got {a.shape} and {b.shape}."
        )


def mask_area(mask: Mask2D | np.ndarray) -> int:
    return int(_to_binary(mask).sum())


def mask_intersection(mask_a: Mask2D | np.ndarray, mask_b: Mask2D | np.ndarray) -> int:
    a = _to_binary(mask_a)
    b = _to_binary(mask_b)
    _check_shapes(a, b)
    return int((a & b).sum())


def mask_union(mask_a: Mask2D | np.ndarray, mask_b: Mask2D | np.ndarray) -> int:
    a = _to_binary(mask_a)
    b = _to_binary(mask_b)
    _check_shapes(a, b)
    return int((a | b).sum())


def mask_iou(mask_a: Mask2D | np.ndarray, mask_b: Mask2D | np.ndarray) -> float:
    a = _to_binary(mask_a)
    b = _to_binary(mask_b)
    _check_shapes(a, b)
    inter = int((a & b).sum())
    if inter == 0:
        return 0.0
    union = int((a | b).sum())
    return float(inter / union) if union > 0 else 0.0


def mask_dice(mask_a: Mask2D | np.ndarray, mask_b: Mask2D | np.ndarray) -> float:
    a = _to_binary(mask_a)
    b = _to_binary(mask_b)
    _check_shapes(a, b)
    inter = int((a & b).sum())
    denom = int(a.sum()) + int(b.sum())
    return float(2 * inter / denom) if denom > 0 else 0.0


def mask_iou_matrix(
    masks_a: Sequence[Mask2D],
    masks_b: Sequence[Mask2D],
) -> np.ndarray:
    n, m = len(masks_a), len(masks_b)
    matrix = np.zeros((n, m), dtype=np.float64)
    for i, a in enumerate(masks_a):
        for j, b in enumerate(masks_b):
            matrix[i, j] = mask_iou(a, b)
    return matrix
