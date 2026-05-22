
from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher


def _det(track_id: str, x1: float, y1: float, x2: float, y2: float, class_id: str | None = None) -> Detection:
    return Detection(frame_id=0, track_id=track_id, bbox2d=Box2D(x1, y1, x2, y2), class_id=class_id)


def test_perfect_match():
    gt = [_det("1", 0, 0, 10, 10)]
    pred = [_det("10", 0, 0, 10, 10)]
    matcher = Box2DIoUMatcher(threshold=0.5)
    result = matcher.match(gt, pred)
    assert result.matches == [(0, 0)]
    assert result.unmatched_gt == []
    assert result.unmatched_pred == []


def test_below_threshold_no_match():
    gt = [_det("1", 0, 0, 10, 10)]
    pred = [_det("10", 8, 8, 18, 18)]
    matcher = Box2DIoUMatcher(threshold=0.5)
    result = matcher.match(gt, pred)
    assert result.matches == []
    assert 0 in result.unmatched_gt
    assert 0 in result.unmatched_pred


def test_no_bbox_no_match():
    gt = [Detection(frame_id=0, track_id="1")]
    pred = [Detection(frame_id=0, track_id="10")]
    matcher = Box2DIoUMatcher(threshold=0.5)
    result = matcher.match(gt, pred)
    assert result.matches == []


def test_class_aware_same_class():
    gt = [_det("1", 0, 0, 10, 10, class_id="car")]
    pred = [_det("10", 0, 0, 10, 10, class_id="car")]
    matcher = Box2DIoUMatcher(threshold=0.5, class_aware=True)
    result = matcher.match(gt, pred)
    assert result.matches == [(0, 0)]


def test_class_aware_different_class():
    gt = [_det("1", 0, 0, 10, 10, class_id="car")]
    pred = [_det("10", 0, 0, 10, 10, class_id="person")]
    matcher = Box2DIoUMatcher(threshold=0.5, class_aware=True)
    result = matcher.match(gt, pred)
    assert result.matches == []


def test_class_aware_null_class_allows_match():
    gt = [_det("1", 0, 0, 10, 10, class_id=None)]
    pred = [_det("10", 0, 0, 10, 10, class_id="car")]
    matcher = Box2DIoUMatcher(threshold=0.5, class_aware=True)
    result = matcher.match(gt, pred)
    assert result.matches == [(0, 0)]


def test_empty_inputs():
    matcher = Box2DIoUMatcher(threshold=0.5)
    result = matcher.match([], [])
    assert result.matches == []
    assert result.unmatched_gt == []
    assert result.unmatched_pred == []
