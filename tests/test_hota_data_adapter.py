from __future__ import annotations

import pytest

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.association import evaluation_result_to_hota_data
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher


def _box(x: float = 0.0) -> Box2D:
    return Box2D(x1=x, y1=0, x2=x + 2, y2=2)


def _det(track_id: str, frame_id: int, x: float = 0.0) -> Detection:
    return Detection(frame_id=frame_id, track_id=track_id, bbox2d=_box(x))


def _seq(*frames_data: tuple) -> Sequence:
    return Sequence(
        name="s",
        frames=[Frame(frame_id=fid, detections=dets) for fid, dets in frames_data],
    )


def _eval_result(gt: Sequence, pred: Sequence):
    evaluator = TrackingEvaluator(
        matcher=Box2DIoUMatcher(threshold=0.5), metrics=[]
    )
    return evaluator.evaluate_events(gt, pred)


def test_frame_result_stores_gt_and_pred():
    gt = _seq((0, [_det("1", 0)]))
    pred = _seq((0, [_det("10", 0)]))
    result = _eval_result(gt, pred)
    fr = result.frame_results[0]
    assert len(fr.gt_detections) == 1
    assert len(fr.pred_detections) == 1
    assert fr.gt_detections[0].track_id == "1"
    assert fr.pred_detections[0].track_id == "10"


def test_frame_result_stores_similarity_matrix():
    gt = _seq((0, [_det("1", 0)]))
    pred = _seq((0, [_det("10", 0)]))
    result = _eval_result(gt, pred)
    fr = result.frame_results[0]
    assert fr.similarity_matrix is not None
    assert fr.similarity_matrix.shape == (1, 1)
    assert fr.similarity_matrix[0, 0] == pytest.approx(1.0)


def test_frame_result_empty_gt():
    gt = _seq((0, []))
    pred = _seq((0, [_det("10", 0)]))
    result = _eval_result(gt, pred)
    fr = result.frame_results[0]
    assert fr.gt_detections == []
    assert len(fr.pred_detections) == 1


def test_hota_data_gt_track_frame_counts():
    gt = _seq(
        (0, [_det("1", 0)]),
        (1, [_det("1", 1), _det("2", 1)]),
    )
    pred = _seq(
        (0, [_det("10", 0)]),
        (1, [_det("10", 1), _det("20", 1)]),
    )
    result = _eval_result(gt, pred)
    hd = evaluation_result_to_hota_data(result)
    assert hd.gt_track_frame_counts["1"] == 2
    assert hd.gt_track_frame_counts["2"] == 1
    assert hd.pred_track_frame_counts["10"] == 2
    assert hd.pred_track_frame_counts["20"] == 1


def test_hota_data_frame_count():
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]))
    pred = _seq((0, [_det("10", 0)]), (1, [_det("10", 1)]))
    result = _eval_result(gt, pred)
    hd = evaluation_result_to_hota_data(result)
    assert len(hd.frames) == 2


def test_hota_data_similarity_matrix_shape():
    gt = _seq((0, [_det("1", 0), _det("2", 0, x=5.0)]))
    pred = _seq((0, [_det("10", 0)]))
    result = _eval_result(gt, pred)
    hd = evaluation_result_to_hota_data(result)
    assert hd.frames[0].similarity_matrix.shape == (2, 1)


def test_hota_data_empty_frame_gets_zero_matrix():
    gt = _seq((0, []))
    pred = _seq((0, []))
    result = _eval_result(gt, pred)
    hd = evaluation_result_to_hota_data(result)
    assert hd.frames[0].similarity_matrix.shape == (0, 0)
