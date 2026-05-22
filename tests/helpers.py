"""Helper utilities shared across metric tests."""

from __future__ import annotations

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import EvaluationResult, TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric
from tracking_metrics.metrics.id_switches import IDSwitchesMetric
from tracking_metrics.metrics.idf1 import IDF1Metric
from tracking_metrics.metrics.mota import MOTAMetric
from tracking_metrics.metrics.motp import MOTPMetric

_BOX = Box2D(0, 0, 100, 100)
_BOX_OTHER = Box2D(200, 200, 300, 300)
_BOX_LOW_IOU = Box2D(90, 90, 190, 190)  # IoU with _BOX is tiny


def det(track_id: str, frame_id: int, box: Box2D = _BOX) -> Detection:
    return Detection(frame_id=frame_id, track_id=track_id, bbox2d=box)


def seq(name: str, *frames: tuple[int, list[Detection]]) -> Sequence:
    return Sequence(name=name, frames=[Frame(frame_id=fid, detections=dets) for fid, dets in frames])


def run_eval(gt: Sequence, pred: Sequence) -> tuple[EvaluationResult, dict]:
    all_metrics = [
        DetectionCountsMetric(),
        IDSwitchesMetric(),
        MOTAMetric(),
        MOTPMetric(),
        IDF1Metric(),
    ]
    matcher = Box2DIoUMatcher(threshold=0.5)
    evaluator = TrackingEvaluator(matcher=matcher, metrics=all_metrics)
    result = evaluator.evaluate_events(gt, pred)
    scores = evaluator.evaluate(gt, pred)
    return result, scores
