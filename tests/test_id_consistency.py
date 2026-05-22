from __future__ import annotations

import pytest

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.id_consistency import IDConsistencyMetric


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
        metrics=[IDConsistencyMetric()],
    )
    return evaluator.evaluate(gt, pred)


def test_perfect_consistency():
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]), (2, [_det("1", 2)]))
    pred = _seq((0, [_det("A", 0)]), (1, [_det("A", 1)]), (2, [_det("A", 2)]))
    scores = _run(gt, pred)
    assert scores["IDCons"] == pytest.approx(1.0)


def test_half_consistency_two_switches():
    # GT track "1" matched to "A" in frames 0,1 and to "B" in frames 2,3
    # Most common = "A" (2/4 frames) or "B" (2/4) → consistency = 0.5
    gt = _seq(
        (0, [_det("1", 0)]),
        (1, [_det("1", 1)]),
        (2, [_det("1", 2)]),
        (3, [_det("1", 3)]),
    )
    pred = _seq(
        (0, [_det("A", 0)]),
        (1, [_det("A", 1)]),
        (2, [_det("B", 2)]),
        (3, [_det("B", 3)]),
    )
    scores = _run(gt, pred)
    assert scores["IDCons"] == pytest.approx(0.5)


def test_no_matches_returns_zero():
    gt = _seq((0, [_det("1", 0)]))
    pred = _seq((0, []))
    scores = _run(gt, pred)
    assert scores["IDCons"] == pytest.approx(0.0)


def test_multiple_tracks_averaged():
    # Track "1": always matched to "A" → consistency = 1.0
    # Track "2": matched to "B" twice and "C" once → consistency = 2/3
    # Mean = (1.0 + 2/3) / 2
    gt = _seq(
        (0, [_det("1", 0), _det("2", 0)]),
        (1, [_det("1", 1), _det("2", 1)]),
        (2, [_det("2", 2)]),
    )
    pred = _seq(
        (0, [_det("A", 0), _det("B", 0)]),
        (1, [_det("A", 1), _det("B", 1)]),
        (2, [_det("C", 2)]),
    )
    scores = _run(gt, pred)
    expected = (1.0 + 2 / 3) / 2
    assert scores["IDCons"] == pytest.approx(expected)
