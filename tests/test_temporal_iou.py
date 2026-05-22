from __future__ import annotations

import numpy as np
import pytest

from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.masks import Mask2D
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.mask_iou_matcher import MaskIoUMatcher
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric
from tracking_metrics.metrics.id_switches import IDSwitchesMetric
from tracking_metrics.metrics.mota import MOTAMetric
from tracking_metrics.metrics.motp import MOTPMetric
from tracking_metrics.metrics.temporal_iou import TemporalDiceMetric, TemporalIoUMetric


def _mask(arr: list) -> Mask2D:
    return Mask2D(data=np.array(arr, dtype=bool))


_FULL = _mask([[1, 1], [1, 1]])
_HALF = _mask([[1, 0], [1, 0]])


def _det(track_id: str, frame_id: int, mask: Mask2D) -> Detection:
    return Detection(frame_id=frame_id, track_id=track_id, mask=mask)


def _seq(name: str, frames_data: list[tuple[int, list[Detection]]]) -> Sequence:
    return Sequence(
        name=name,
        frames=[Frame(frame_id=fid, detections=dets) for fid, dets in frames_data],
    )


def _run(gt: Sequence, pred: Sequence) -> tuple:
    metrics = [
        DetectionCountsMetric(),
        IDSwitchesMetric(),
        MOTAMetric(),
        MOTPMetric(),
        TemporalIoUMetric(),
        TemporalDiceMetric(),
    ]
    evaluator = TrackingEvaluator(matcher=MaskIoUMatcher(threshold=0.5), metrics=metrics)
    result = evaluator.evaluate_events(gt, pred)
    scores = evaluator.evaluate(gt, pred)
    return result, scores


def test_perfect_mask_tracking():
    gt = _seq("s", [(0, [_det("1", 0, _FULL)]), (1, [_det("1", 1, _FULL)])])
    pred = _seq("s", [(0, [_det("10", 0, _FULL)]), (1, [_det("10", 1, _FULL)])])
    _, scores = _run(gt, pred)
    assert scores["TP"] == 2
    assert scores["MOTA"] == pytest.approx(1.0)
    assert scores["MOTP"] == pytest.approx(1.0)
    assert scores["IDSW"] == 0
    assert scores["T-mIoU"] == pytest.approx(1.0)
    assert scores["T-Dice"] == pytest.approx(1.0)


def test_id_switch_with_masks():
    gt = _seq("s", [(0, [_det("1", 0, _FULL)]), (1, [_det("1", 1, _FULL)])])
    pred = _seq("s", [(0, [_det("10", 0, _FULL)]), (1, [_det("20", 1, _FULL)])])
    _, scores = _run(gt, pred)
    assert scores["IDSW"] == 1


def test_iou_threshold_rejection():
    # _HALF vs _HALF_OTHER has IoU < 0.5
    half_l = _mask([[1, 0], [1, 0]])
    half_r = _mask([[0, 1], [0, 1]])
    gt = _seq("s", [(0, [_det("1", 0, half_l)])])
    pred = _seq("s", [(0, [_det("10", 0, half_r)])])
    _, scores = _run(gt, pred)
    assert scores["TP"] == 0
    assert scores["FP"] == 1
    assert scores["FN"] == 1
    assert scores["T-mIoU"] == pytest.approx(0.0)


def test_t_miou_no_matches():
    gt = _seq("s", [(0, [_det("1", 0, _FULL)])])
    pred = _seq("s", [(0, [])])
    _, scores = _run(gt, pred)
    assert scores["T-mIoU"] == pytest.approx(0.0)


def test_t_miou_partial_overlap():
    # _HALF has area=2, _FULL has area=4, intersection=2, union=4 => IoU=0.5
    gt = _seq("s", [(0, [_det("1", 0, _HALF)])])
    pred = _seq("s", [(0, [_det("10", 0, _FULL)])])
    _, scores = _run(gt, pred)
    # IoU=0.5 is exactly at threshold => match accepted
    assert scores["TP"] == 1
    assert scores["T-mIoU"] == pytest.approx(0.5)


def test_t_miou_skips_detections_without_masks():
    from tracking_metrics.data.boxes import Box2D
    # Use box matcher — masks missing, T-mIoU should be 0.0
    box = Box2D(0, 0, 10, 10)
    gt_det = Detection(frame_id=0, track_id="1", bbox2d=box)
    pred_det = Detection(frame_id=0, track_id="10", bbox2d=box)
    gt = Sequence(name="s", frames=[Frame(frame_id=0, detections=[gt_det])])
    pred_s = Sequence(name="s", frames=[Frame(frame_id=0, detections=[pred_det])])

    from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher

    metrics = [TemporalIoUMetric()]
    evaluator = TrackingEvaluator(matcher=Box2DIoUMatcher(threshold=0.5), metrics=metrics)
    scores = evaluator.evaluate(gt, pred_s)
    # match exists (IoU=1.0) but no masks -> T-mIoU = 0.0
    assert scores["T-mIoU"] == pytest.approx(0.0)
