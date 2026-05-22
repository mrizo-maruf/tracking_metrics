from __future__ import annotations

import pytest

from tracking_metrics.data.boxes3d import Box3D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box3d_iou_matcher import Box3DIoUMatcher
from tracking_metrics.matching.center_distance_matcher import CenterDistanceMatcher
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric
from tracking_metrics.metrics.id_switches import IDSwitchesMetric
from tracking_metrics.metrics.localization3d import MeanBox3DIoU, MeanCenterDistance3D
from tracking_metrics.metrics.mota import MOTAMetric
from tracking_metrics.metrics.motp import MOTPMetric


def _box(center: tuple) -> Box3D:
    return Box3D(center=center, size=(2, 2, 2))


def _det(track_id: str, frame_id: int, center: tuple) -> Detection:
    return Detection(frame_id=frame_id, track_id=track_id, bbox3d=_box(center))


def _seq(name: str, *frames_data: tuple) -> Sequence:
    return Sequence(
        name=name,
        frames=[Frame(frame_id=fid, detections=dets) for fid, dets in frames_data],
    )


def _run_iou(gt: Sequence, pred: Sequence, threshold: float = 0.25) -> dict:
    metrics = [
        DetectionCountsMetric(), IDSwitchesMetric(), MOTAMetric(), MOTPMetric(),
        MeanBox3DIoU(), MeanCenterDistance3D(),
    ]
    evaluator = TrackingEvaluator(matcher=Box3DIoUMatcher(threshold=threshold), metrics=metrics)
    return evaluator.evaluate(gt, pred)


def _run_dist(gt: Sequence, pred: Sequence, max_distance: float = 1.0) -> dict:
    metrics = [
        DetectionCountsMetric(), IDSwitchesMetric(), MOTAMetric(), MOTPMetric(),
        MeanBox3DIoU(), MeanCenterDistance3D(),
    ]
    evaluator = TrackingEvaluator(
        matcher=CenterDistanceMatcher(max_distance=max_distance), metrics=metrics
    )
    return evaluator.evaluate(gt, pred)


# --- IoU matcher: full 3D tracking pipeline ---

def test_perfect_3d_tracking_iou():
    gt = _seq("s", (0, [_det("1", 0, (0, 0, 0))]), (1, [_det("1", 1, (0, 0, 0))]))
    pred = _seq("s", (0, [_det("10", 0, (0, 0, 0))]), (1, [_det("10", 1, (0, 0, 0))]))
    scores = _run_iou(gt, pred)
    assert scores["TP"] == 2
    assert scores["MOTA"] == pytest.approx(1.0)
    assert scores["MOTP"] == pytest.approx(1.0)
    assert scores["IDSW"] == 0
    assert scores["MeanBox3DIoU"] == pytest.approx(1.0)
    assert scores["MeanCenterDist3D"] == pytest.approx(0.0)


def test_id_switch_3d():
    gt = _seq("s", (0, [_det("1", 0, (0, 0, 0))]), (1, [_det("1", 1, (0, 0, 0))]))
    pred = _seq("s", (0, [_det("10", 0, (0, 0, 0))]), (1, [_det("20", 1, (0, 0, 0))]))
    scores = _run_iou(gt, pred)
    assert scores["IDSW"] == 1
    assert scores["MOTA"] == pytest.approx(0.5)


def test_3d_false_positive():
    gt = _seq("s", (0, []))
    pred = _seq("s", (0, [_det("10", 0, (0, 0, 0))]))
    scores = _run_iou(gt, pred)
    assert scores["FP"] == 1
    assert scores["TP"] == 0


def test_3d_false_negative():
    gt = _seq("s", (0, [_det("1", 0, (0, 0, 0))]))
    pred = _seq("s", (0, []))
    scores = _run_iou(gt, pred)
    assert scores["FN"] == 1
    assert scores["TP"] == 0


def test_mean_box3d_iou_partial():
    # Overlap: 4/12 ≈ 0.333
    a = Box3D(center=(0, 0, 0), size=(2, 2, 2))
    b = Box3D(center=(1, 0, 0), size=(2, 2, 2))
    gt = Sequence(name="s", frames=[Frame(frame_id=0, detections=[
        Detection(frame_id=0, track_id="1", bbox3d=a),
    ])])
    pred = Sequence(name="s", frames=[Frame(frame_id=0, detections=[
        Detection(frame_id=0, track_id="10", bbox3d=b),
    ])])
    scores = _run_iou(gt, pred, threshold=0.25)
    assert scores["MeanBox3DIoU"] == pytest.approx(4 / 12)


def test_mean_center_dist_3d():
    gt = _seq("s", (0, [_det("1", 0, (0, 0, 0))]))
    pred = _seq("s", (0, [_det("10", 0, (3, 4, 0))]))
    # IoU=0 for non-overlapping boxes; use distance matcher to get a real match
    scores2 = _run_dist(gt, pred, max_distance=10.0)
    assert scores2["MeanCenterDist3D"] == pytest.approx(5.0)


# --- Center distance matcher ---

def test_perfect_3d_tracking_dist():
    gt = _seq("s", (0, [_det("1", 0, (0, 0, 0))]), (1, [_det("1", 1, (0, 0, 0))]))
    pred = _seq("s", (0, [_det("10", 0, (0, 0, 0))]), (1, [_det("10", 1, (0, 0, 0))]))
    scores = _run_dist(gt, pred)
    assert scores["TP"] == 2
    assert scores["MOTA"] == pytest.approx(1.0)
    assert scores["IDSW"] == 0
    assert scores["MeanCenterDist3D"] == pytest.approx(0.0)


def test_center_dist_above_max_rejected():
    gt = _seq("s", (0, [_det("1", 0, (0, 0, 0))]))
    pred = _seq("s", (0, [_det("10", 0, (2, 0, 0))]))
    scores = _run_dist(gt, pred, max_distance=1.0)
    assert scores["TP"] == 0
    assert scores["FP"] == 1
    assert scores["FN"] == 1


def test_id_switch_center_dist():
    gt = _seq("s", (0, [_det("1", 0, (0, 0, 0))]), (1, [_det("1", 1, (0, 0, 0))]))
    pred = _seq("s", (0, [_det("10", 0, (0, 0, 0))]), (1, [_det("20", 1, (0, 0, 0))]))
    scores = _run_dist(gt, pred)
    assert scores["IDSW"] == 1


def test_mean_center_dist_no_matches_returns_zero():
    gt = _seq("s", (0, [_det("1", 0, (0, 0, 0))]))
    pred = _seq("s", (0, []))
    scores = _run_dist(gt, pred)
    assert scores["MeanCenterDist3D"] == pytest.approx(0.0)
    assert scores["MeanBox3DIoU"] == pytest.approx(0.0)
