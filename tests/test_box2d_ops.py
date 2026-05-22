import pytest

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.geometry.box2d_ops import box2d_iou, box2d_iou_matrix


def test_iou_identical_boxes():
    b = Box2D(0, 0, 10, 10)
    assert box2d_iou(b, b) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = Box2D(0, 0, 10, 10)
    b = Box2D(20, 20, 30, 30)
    assert box2d_iou(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    a = Box2D(0, 0, 10, 10)
    b = Box2D(5, 5, 15, 15)
    # intersection = 5*5 = 25, union = 100+100-25 = 175
    assert box2d_iou(a, b) == pytest.approx(25 / 175)


def test_iou_one_inside_other():
    outer = Box2D(0, 0, 10, 10)
    inner = Box2D(2, 2, 8, 8)
    # intersection = 6*6 = 36, union = 100+36-36 = 100
    assert box2d_iou(outer, inner) == pytest.approx(36 / 100)


def test_iou_invalid_box_returns_zero():
    valid = Box2D(0, 0, 10, 10)
    invalid = Box2D(10, 10, 5, 5)
    assert box2d_iou(valid, invalid) == 0.0
    assert box2d_iou(invalid, valid) == 0.0


def test_iou_matrix_shape():
    boxes_a = [Box2D(0, 0, 10, 10), Box2D(20, 20, 30, 30)]
    boxes_b = [Box2D(5, 5, 15, 15), Box2D(25, 25, 35, 35), Box2D(50, 50, 60, 60)]
    mat = box2d_iou_matrix(boxes_a, boxes_b)
    assert mat.shape == (2, 3)


def test_iou_matrix_empty_a():
    mat = box2d_iou_matrix([], [Box2D(0, 0, 10, 10)])
    assert mat.shape == (0, 1)


def test_iou_matrix_empty_b():
    mat = box2d_iou_matrix([Box2D(0, 0, 10, 10)], [])
    assert mat.shape == (1, 0)


def test_iou_matrix_both_empty():
    mat = box2d_iou_matrix([], [])
    assert mat.shape == (0, 0)


def test_iou_matrix_values():
    a = Box2D(0, 0, 10, 10)
    b = Box2D(0, 0, 10, 10)
    mat = box2d_iou_matrix([a], [b])
    assert mat[0, 0] == pytest.approx(1.0)
