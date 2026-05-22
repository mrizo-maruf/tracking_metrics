from __future__ import annotations

import pytest

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.track_survival import TrackSurvivalRateMetric


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
        metrics=[TrackSurvivalRateMetric()],
    )
    return evaluator.evaluate(gt, pred)


def test_all_survive():
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]))
    pred = _seq((0, [_det("A", 0)]), (1, [_det("A", 1)]))
    scores = _run(gt, pred)
    assert scores["T-SR"] == pytest.approx(1.0)
    assert scores["SurvivedTracks"] == 1
    assert scores["TotalGTTracks"] == 1


def test_none_survive():
    gt = _seq((0, [_det("1", 0)]))
    pred = _seq((0, []))
    scores = _run(gt, pred)
    assert scores["T-SR"] == pytest.approx(0.0)
    assert scores["SurvivedTracks"] == 0
    assert scores["TotalGTTracks"] == 1


def test_partial_survival():
    gt = _seq((0, [_det("1", 0), _det("2", 0)]))
    # Only track "1" has a matching pred
    pred = _seq((0, [_det("A", 0)]))
    scores = _run(gt, pred)
    assert scores["T-SR"] == pytest.approx(0.5)
    assert scores["SurvivedTracks"] == 1
    assert scores["TotalGTTracks"] == 2


def test_empty_gt_returns_zero():
    gt = _seq((0, []))
    pred = _seq((0, []))
    scores = _run(gt, pred)
    assert scores["T-SR"] == pytest.approx(0.0)
    assert scores["TotalGTTracks"] == 0
