import numpy as np
import pytest

from tracking_metrics.data.masks import Mask2D
from tracking_metrics.geometry.mask_ops import (
    mask_area,
    mask_dice,
    mask_intersection,
    mask_iou,
    mask_iou_matrix,
    mask_union,
)


def _m(arr: list) -> Mask2D:
    return Mask2D(data=np.array(arr, dtype=bool))


def test_area_2x2():
    m = _m([[1, 0], [0, 1]])
    assert mask_area(m) == 2


def test_area_all_zeros():
    m = _m([[0, 0], [0, 0]])
    assert mask_area(m) == 0


def test_iou_identical():
    m = _m([[1, 1], [0, 0]])
    assert mask_iou(m, m) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = _m([[1, 0], [0, 0]])
    b = _m([[0, 1], [0, 0]])
    assert mask_iou(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    # mask_a: 4 pixels active (left 2x2), mask_b: 4 pixels active (right 2x2 in 2x4 grid)
    # Let's use a simpler case: area=4, area=4, intersection=2, union=6
    a = _m([[1, 1, 0, 0], [1, 1, 0, 0]])
    b = _m([[0, 0, 1, 1], [0, 1, 1, 0]])
    inter = mask_intersection(a, b)
    union = mask_union(a, b)
    assert inter == 1  # only b[1,1] overlaps (b has [0,1] row)
    expected_iou = inter / union
    assert mask_iou(a, b) == pytest.approx(expected_iou)


def test_iou_spec_case():
    # spec: area_a=4, area_b=4, intersection=2, union=6 => IoU=2/6
    a = _m([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    b = _m([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
    assert mask_intersection(a, b) == 2
    assert mask_union(a, b) == 6
    assert mask_iou(a, b) == pytest.approx(2 / 6)


def test_iou_zero_mask_returns_zero():
    a = _m([[0, 0], [0, 0]])
    b = _m([[1, 1], [1, 1]])
    assert mask_iou(a, b) == pytest.approx(0.0)


def test_shape_mismatch_raises():
    a = _m([[1, 0]])
    b = _m([[1, 0, 0]])
    with pytest.raises(ValueError, match="shapes must be identical"):
        mask_iou(a, b)


def test_intersection_shape_mismatch_raises():
    a = _m([[1, 0]])
    b = _m([[1, 0, 0]])
    with pytest.raises(ValueError, match="shapes must be identical"):
        mask_intersection(a, b)


def test_dice_identical():
    m = _m([[1, 1], [0, 0]])
    assert mask_dice(m, m) == pytest.approx(1.0)


def test_dice_no_overlap():
    a = _m([[1, 0], [0, 0]])
    b = _m([[0, 0], [0, 1]])
    assert mask_dice(a, b) == pytest.approx(0.0)


def test_dice_partial():
    # inter=2, denom=4+4=8 => dice=4/8=0.5
    a = _m([[1, 1, 0, 0]])
    b = _m([[0, 1, 1, 0]])
    assert mask_dice(a, b) == pytest.approx(2 * 1 / (2 + 2))


def test_iou_matrix_shape():
    masks_a = [_m([[1, 0]]), _m([[0, 1]])]
    masks_b = [_m([[1, 0]]), _m([[0, 1]]), _m([[1, 1]])]
    mat = mask_iou_matrix(masks_a, masks_b)
    assert mat.shape == (2, 3)


def test_iou_matrix_empty():
    mat = mask_iou_matrix([], [])
    assert mat.shape == (0, 0)


def test_iou_matrix_correct_values():
    m = _m([[1, 1]])
    mat = mask_iou_matrix([m], [m])
    assert mat[0, 0] == pytest.approx(1.0)


def test_mask2d_to_binary_from_data():
    data = np.array([[True, False], [False, True]])
    m = Mask2D(data=data)
    result = m.to_binary()
    assert result.dtype == bool
    np.testing.assert_array_equal(result, data)


def test_mask2d_area():
    m = Mask2D(data=np.array([[1, 0], [0, 1]], dtype=bool))
    assert m.area() == 2


def test_mask2d_no_data_or_rle_raises():
    m = Mask2D()
    with pytest.raises(ValueError, match="neither data nor rle"):
        m.to_binary()


def test_mask2d_rle_without_pycocotools():
    m = Mask2D(rle={"size": [4, 4], "counts": "abc"})
    import sys
    # Temporarily hide pycocotools if present
    original = sys.modules.get("pycocotools")
    sys.modules["pycocotools"] = None  # type: ignore[assignment]
    sys.modules["pycocotools.mask"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError, match="pycocotools"):
            m.to_binary()
    finally:
        if original is None:
            sys.modules.pop("pycocotools", None)
            sys.modules.pop("pycocotools.mask", None)
        else:
            sys.modules["pycocotools"] = original
