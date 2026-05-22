from __future__ import annotations

import pytest

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.track_coverage import TrackCoverageMetric


def _box() -> Box2D:
    return Box2D(x1=0, y1=0, x2=2, y2=2)


def _det(track_id: str, frame_id: int) -> Detection:
    return Detection(frame_id=frame_id, track_id=track_id, bbox2d=_box())


def _seq(*frames_data: tuple) -> Sequence:
    return Sequence(
        name="s",
        frames=[Frame(frame_id=fid, detections=dets) for fid, dets in frames_data],
    )


def _run(gt: Sequence, pred: Sequence) -> dict:
    evaluator = TrackingEvaluator(
        matcher=Box2DIoUMatcher(threshold=0.5),
        metrics=[TrackCoverageMetric()],
    )
    return evaluator.evaluate(gt, pred)


def test_all_matched_is_mt():
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]))
    pred = _seq((0, [_det("A", 0)]), (1, [_det("A", 1)]))
    scores = _run(gt, pred)
    assert scores["MT"] == 1
    assert scores["PT"] == 0
    assert scores["ML"] == 0
    assert scores["MT%"] == pytest.approx(1.0)


def test_none_matched_is_ml():
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]))
    pred = _seq((0, []), (1, []))
    scores = _run(gt, pred)
    assert scores["ML"] == 1
    assert scores["MT"] == 0
    assert scores["ML%"] == pytest.approx(1.0)


def test_partial_match_is_pt():
    # 5 frames, 2 matched (40% coverage) → PT (0.2 <= 0.4 < 0.8)
    gt = _seq(
        (0, [_det("1", 0)]),
        (1, [_det("1", 1)]),
        (2, [_det("1", 2)]),
        (3, [_det("1", 3)]),
        (4, [_det("1", 4)]),
    )
    pred = _seq(
        (0, [_det("A", 0)]),
        (1, [_det("A", 1)]),
        (2, []),
        (3, []),
        (4, []),
    )
    scores = _run(gt, pred)
    assert scores["PT"] == 1
    assert scores["MT"] == 0
    assert scores["ML"] == 0


def test_mt_pl_ml_percentages_sum_to_one():
    gt = _seq(
        (0, [_det("1", 0), _det("2", 0)]),
        (1, [_det("1", 1)]),
    )
    pred = _seq((0, [_det("A", 0)]), (1, [_det("A", 1)]))
    scores = _run(gt, pred)
    total = scores["MT%"] + scores["PT%"] + scores["ML%"]
    assert total == pytest.approx(1.0)


def test_custom_thresholds():
    # With mt_threshold=0.5, a track matched in 1/2 frames should be MT
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]))
    pred = _seq((0, [_det("A", 0)]), (1, []))
    evaluator = TrackingEvaluator(
        matcher=Box2DIoUMatcher(threshold=0.5),
        metrics=[TrackCoverageMetric(mt_threshold=0.5, ml_threshold=0.1)],
    )
    scores = evaluator.evaluate(gt, pred)
    assert scores["MT"] == 1
