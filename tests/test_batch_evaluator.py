from __future__ import annotations

import pytest

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.batch import BatchEvaluator, SequencePair
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric
from tracking_metrics.metrics.id_switches import IDSwitchesMetric
from tracking_metrics.metrics.mota import MOTAMetric


def _box() -> Box2D:
    return Box2D(x1=0, y1=0, x2=2, y2=2)


def _det(track_id: str, frame_id: int) -> Detection:
    return Detection(frame_id=frame_id, track_id=track_id, bbox2d=_box())


def _seq(name: str, *frames_data: tuple) -> Sequence:
    return Sequence(
        name=name,
        frames=[Frame(frame_id=fid, detections=dets) for fid, dets in frames_data],
    )


def _make_evaluator():
    return TrackingEvaluator(
        matcher=Box2DIoUMatcher(threshold=0.5),
        metrics=[DetectionCountsMetric(), IDSwitchesMetric(), MOTAMetric()],
    )


def test_batch_output_structure():
    gt1 = _seq("s1", (0, [_det("1", 0)]))
    pred1 = _seq("s1", (0, [_det("A", 0)]))
    gt2 = _seq("s2", (0, [_det("2", 0)]))
    pred2 = _seq("s2", (0, [_det("B", 0)]))

    evaluator = _make_evaluator()
    batch = BatchEvaluator(evaluator)
    results = batch.evaluate([
        SequencePair("s1", gt1, pred1),
        SequencePair("s2", gt2, pred2),
    ])

    assert "sequences" in results
    assert "average" in results
    assert "global" in results
    assert "s1" in results["sequences"]
    assert "s2" in results["sequences"]


def test_batch_per_sequence_scores():
    gt1 = _seq("s1", (0, [_det("1", 0)]), (1, [_det("1", 1)]))
    pred1 = _seq("s1", (0, [_det("A", 0)]), (1, [_det("A", 1)]))

    evaluator = _make_evaluator()
    batch = BatchEvaluator(evaluator)
    results = batch.evaluate([SequencePair("s1", gt1, pred1)])

    scores = results["sequences"]["s1"]
    assert scores["TP"] == 2
    assert scores["MOTA"] == pytest.approx(1.0)


def test_batch_average_is_arithmetic_mean():
    # s1: MOTA=1.0, s2: MOTA=0.5 → average MOTA = 0.75
    gt1 = _seq("s1", (0, [_det("1", 0)]))
    pred1 = _seq("s1", (0, [_det("A", 0)]))

    gt2 = _seq("s2", (0, [_det("1", 0)]), (1, [_det("1", 1)]))
    pred2 = _seq("s2", (0, [_det("A", 0)]), (1, []))  # one FN → MOTA=0.5

    evaluator = _make_evaluator()
    batch = BatchEvaluator(evaluator)
    results = batch.evaluate([
        SequencePair("s1", gt1, pred1),
        SequencePair("s2", gt2, pred2),
    ])

    assert results["average"]["MOTA"] == pytest.approx(0.75)


def test_batch_global_tp_is_sum():
    gt1 = _seq("s1", (0, [_det("1", 0)]))
    pred1 = _seq("s1", (0, [_det("A", 0)]))
    gt2 = _seq("s2", (0, [_det("2", 0)]))
    pred2 = _seq("s2", (0, [_det("B", 0)]))

    evaluator = _make_evaluator()
    batch = BatchEvaluator(evaluator)
    results = batch.evaluate([
        SequencePair("s1", gt1, pred1),
        SequencePair("s2", gt2, pred2),
    ])

    # Each sequence has 1 TP → global TP should be 2
    assert results["global"]["TP"] == 2


def test_batch_empty_sequences():
    gt1 = _seq("s1", (0, []))
    pred1 = _seq("s1", (0, []))

    evaluator = _make_evaluator()
    batch = BatchEvaluator(evaluator)
    results = batch.evaluate([SequencePair("s1", gt1, pred1)])

    assert results["sequences"]["s1"]["TP"] == 0
    assert "average" in results
    assert "global" in results


def test_batch_empty_list():
    evaluator = _make_evaluator()
    batch = BatchEvaluator(evaluator)
    results = batch.evaluate([])

    assert results["sequences"] == {}
    assert results["average"] == {}
    assert results["global"] == {}
