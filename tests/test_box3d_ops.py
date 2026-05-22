import numpy as np
import pytest

from tracking_metrics.data.boxes3d import Box3D
from tracking_metrics.geometry.box3d_ops import (
    box3d_iou_axis_aligned,
    box3d_iou_matrix_axis_aligned,
)
from tracking_metrics.geometry.distance_ops import center_distance_3d, center_distance_matrix_3d

# --- Box3D model ---

def test_volume():
    b = Box3D(center=(0, 0, 0), size=(2, 3, 4))
    assert b.volume() == pytest.approx(24.0)


def test_volume_zero():
    b = Box3D(center=(0, 0, 0), size=(0, 3, 4))
    assert b.volume() == pytest.approx(0.0)


def test_is_valid_positive_size():
    assert Box3D(center=(0, 0, 0), size=(1, 1, 1)).is_valid()


def test_is_valid_zero_size():
    assert Box3D(center=(0, 0, 0), size=(0, 1, 1)).is_valid()


def test_is_invalid_negative_size():
    assert not Box3D(center=(0, 0, 0), size=(-1, 1, 1)).is_valid()


def test_min_max_corners():
    b = Box3D(center=(0, 0, 0), size=(2, 4, 6))
    np.testing.assert_allclose(b.min_corner(), [-1, -2, -3])
    np.testing.assert_allclose(b.max_corner(), [1, 2, 3])


def test_yaw_stored_not_used():
    b1 = Box3D(center=(0, 0, 0), size=(1, 1, 1), yaw=1.57)
    b2 = Box3D(center=(0, 0, 0), size=(1, 1, 1), yaw=None)
    # IoU should be identical regardless of yaw for axis-aligned ops
    assert box3d_iou_axis_aligned(b1, b2) == pytest.approx(1.0)


# --- Axis-aligned IoU ---

def test_iou_identical():
    b = Box3D(center=(0, 0, 0), size=(2, 2, 2))
    assert box3d_iou_axis_aligned(b, b) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = Box3D(center=(0, 0, 0), size=(1, 1, 1))
    b = Box3D(center=(5, 0, 0), size=(1, 1, 1))
    assert box3d_iou_axis_aligned(a, b) == pytest.approx(0.0)


def test_iou_touching_edges():
    a = Box3D(center=(0, 0, 0), size=(2, 2, 2))
    b = Box3D(center=(2, 0, 0), size=(2, 2, 2))
    # Touching at x=1 plane — intersection volume = 0
    assert box3d_iou_axis_aligned(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    # a: x in [-1,1], y in [-1,1], z in [-1,1]  (vol=8)
    # b: x in [0, 2], y in [-1,1], z in [-1,1]  (vol=8)
    # intersection: x in [0,1], y in [-1,1], z in [-1,1]  => vol=4
    # union = 8+8-4 = 12
    a = Box3D(center=(0, 0, 0), size=(2, 2, 2))
    b = Box3D(center=(1, 0, 0), size=(2, 2, 2))
    assert box3d_iou_axis_aligned(a, b) == pytest.approx(4 / 12)


def test_iou_one_inside_other():
    outer = Box3D(center=(0, 0, 0), size=(4, 4, 4))
    inner = Box3D(center=(0, 0, 0), size=(2, 2, 2))
    # intersection = 8, union = 64+8-8 = 64
    assert box3d_iou_axis_aligned(outer, inner) == pytest.approx(8 / 64)


def test_iou_invalid_box_returns_zero():
    valid = Box3D(center=(0, 0, 0), size=(1, 1, 1))
    invalid = Box3D(center=(0, 0, 0), size=(-1, 1, 1))
    assert box3d_iou_axis_aligned(valid, invalid) == pytest.approx(0.0)


def test_iou_matrix_shape():
    a = [Box3D(center=(0, 0, 0), size=(1, 1, 1))]
    b = [Box3D(center=(0, 0, 0), size=(1, 1, 1)), Box3D(center=(5, 0, 0), size=(1, 1, 1))]
    mat = box3d_iou_matrix_axis_aligned(a, b)
    assert mat.shape == (1, 2)


def test_iou_matrix_empty():
    mat = box3d_iou_matrix_axis_aligned([], [])
    assert mat.shape == (0, 0)


# --- Center distance ---

def test_center_distance_345():
    a = Box3D(center=(0, 0, 0), size=(1, 1, 1))
    b = Box3D(center=(3, 4, 0), size=(1, 1, 1))
    assert center_distance_3d(a, b) == pytest.approx(5.0)


def test_center_distance_same_center():
    a = Box3D(center=(1, 2, 3), size=(1, 1, 1))
    assert center_distance_3d(a, a) == pytest.approx(0.0)


def test_center_distance_matrix_shape():
    a = [Box3D(center=(0, 0, 0), size=(1, 1, 1)), Box3D(center=(1, 0, 0), size=(1, 1, 1))]
    b = [Box3D(center=(0, 0, 0), size=(1, 1, 1))]
    mat = center_distance_matrix_3d(a, b)
    assert mat.shape == (2, 1)


def test_center_distance_matrix_values():
    a = Box3D(center=(0, 0, 0), size=(1, 1, 1))
    b = Box3D(center=(3, 4, 0), size=(1, 1, 1))
    mat = center_distance_matrix_3d([a], [b])
    assert mat[0, 0] == pytest.approx(5.0)
