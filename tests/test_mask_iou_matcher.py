from __future__ import annotations

import numpy as np

from tracking_metrics.data.detection import Detection
from tracking_metrics.data.masks import Mask2D
from tracking_metrics.matching.mask_iou_matcher import MaskIoUMatcher


def _mask(arr: list) -> Mask2D:
    return Mask2D(data=np.array(arr, dtype=bool))


def _det(
    track_id: str,
    mask: Mask2D | None = None,
    class_id: str | None = None,
) -> Detection:
    return Detection(frame_id=0, track_id=track_id, mask=mask, class_id=class_id)


_FULL = _mask([[1, 1], [1, 1]])
_EMPTY = _mask([[0, 0], [0, 0]])
_HALF_L = _mask([[1, 0], [1, 0]])
_HALF_R = _mask([[0, 1], [0, 1]])


def test_identical_mask_matches():
    gt = [_det("1", _FULL)]
    pred = [_det("10", _FULL)]
    result = MaskIoUMatcher(threshold=0.5).match(gt, pred)
    assert result.matches == [(0, 0)]
    assert result.unmatched_gt == []
    assert result.unmatched_pred == []


def test_no_overlap_no_match():
    gt = [_det("1", _HALF_L)]
    pred = [_det("10", _HALF_R)]
    result = MaskIoUMatcher(threshold=0.5).match(gt, pred)
    assert result.matches == []
    assert 0 in result.unmatched_gt
    assert 0 in result.unmatched_pred


def test_no_mask_no_match():
    gt = [_det("1")]
    pred = [_det("10")]
    result = MaskIoUMatcher(threshold=0.5).match(gt, pred)
    assert result.matches == []


def test_below_threshold_no_match():
    # _HALF_L and _FULL have IoU = 2/4 = 0.5 — exactly at threshold
    # use a stricter threshold
    gt = [_det("1", _HALF_L)]
    pred = [_det("10", _FULL)]
    result = MaskIoUMatcher(threshold=0.6).match(gt, pred)
    assert result.matches == []


def test_class_aware_same_class_matches():
    gt = [_det("1", _FULL, class_id="obj")]
    pred = [_det("10", _FULL, class_id="obj")]
    result = MaskIoUMatcher(threshold=0.5, class_aware=True).match(gt, pred)
    assert result.matches == [(0, 0)]


def test_class_aware_different_class_no_match():
    gt = [_det("1", _FULL, class_id="car")]
    pred = [_det("10", _FULL, class_id="person")]
    result = MaskIoUMatcher(threshold=0.5, class_aware=True).match(gt, pred)
    assert result.matches == []


def test_class_aware_null_class_allows_match():
    gt = [_det("1", _FULL, class_id=None)]
    pred = [_det("10", _FULL, class_id="car")]
    result = MaskIoUMatcher(threshold=0.5, class_aware=True).match(gt, pred)
    assert result.matches == [(0, 0)]


def test_empty_inputs():
    result = MaskIoUMatcher().match([], [])
    assert result.matches == []
    assert result.unmatched_gt == []
    assert result.unmatched_pred == []
