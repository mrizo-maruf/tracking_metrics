from __future__ import annotations

from tracking_metrics.data.boxes3d import Box3D
from tracking_metrics.data.detection import Detection
from tracking_metrics.matching.box3d_iou_matcher import Box3DIoUMatcher


def _det(
    track_id: str,
    center: tuple,
    size: tuple,
    class_id: str | None = None,
) -> Detection:
    return Detection(
        frame_id=0,
        track_id=track_id,
        bbox3d=Box3D(center=center, size=size),
        class_id=class_id,
    )


_BOX_A = ((0, 0, 0), (2, 2, 2))   # centered at origin, 2x2x2
_BOX_B = ((0, 0, 0), (2, 2, 2))   # same
_FAR = ((10, 10, 10), (2, 2, 2))  # no overlap


def test_identical_box_matches():
    gt = [_det("1", *_BOX_A)]
    pred = [_det("10", *_BOX_B)]
    result = Box3DIoUMatcher(threshold=0.25).match(gt, pred)
    assert result.matches == [(0, 0)]


def test_no_overlap_no_match():
    gt = [_det("1", *_BOX_A)]
    pred = [_det("10", *_FAR)]
    result = Box3DIoUMatcher(threshold=0.25).match(gt, pred)
    assert result.matches == []
    assert 0 in result.unmatched_gt
    assert 0 in result.unmatched_pred


def test_no_bbox3d_no_match():
    gt = [Detection(frame_id=0, track_id="1")]
    pred = [Detection(frame_id=0, track_id="10")]
    result = Box3DIoUMatcher(threshold=0.25).match(gt, pred)
    assert result.matches == []


def test_below_threshold_no_match():
    # IoU = 4/12 ≈ 0.33 — above 0.25, but let's use 0.5 threshold
    a = Box3D(center=(0, 0, 0), size=(2, 2, 2))
    b = Box3D(center=(1, 0, 0), size=(2, 2, 2))
    gt = [Detection(frame_id=0, track_id="1", bbox3d=a)]
    pred = [Detection(frame_id=0, track_id="10", bbox3d=b)]
    result = Box3DIoUMatcher(threshold=0.5).match(gt, pred)
    assert result.matches == []


def test_class_aware_same_class_matches():
    gt = [_det("1", *_BOX_A, class_id="car")]
    pred = [_det("10", *_BOX_B, class_id="car")]
    result = Box3DIoUMatcher(threshold=0.25, class_aware=True).match(gt, pred)
    assert result.matches == [(0, 0)]


def test_class_aware_different_class_no_match():
    gt = [_det("1", *_BOX_A, class_id="car")]
    pred = [_det("10", *_BOX_B, class_id="person")]
    result = Box3DIoUMatcher(threshold=0.25, class_aware=True).match(gt, pred)
    assert result.matches == []


def test_empty_inputs():
    result = Box3DIoUMatcher().match([], [])
    assert result.matches == []
