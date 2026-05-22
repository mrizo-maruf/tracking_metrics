"""Example: evaluate mask tracking with the Python API."""

from __future__ import annotations

import numpy as np

from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.masks import Mask2D
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.mask_iou_matcher import MaskIoUMatcher
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric
from tracking_metrics.metrics.id_switches import IDSwitchesMetric
from tracking_metrics.metrics.idf1 import IDF1Metric
from tracking_metrics.metrics.mota import MOTAMetric
from tracking_metrics.metrics.motp import MOTPMetric
from tracking_metrics.metrics.temporal_iou import TemporalIoUMetric
from tracking_metrics.report.terminal import print_results_table


def make_mask(arr: list) -> Mask2D:
    return Mask2D(data=np.array(arr, dtype=bool))


mask_a = make_mask([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
mask_b = make_mask([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

gt = Sequence(
    name="toy_scene",
    frames=[
        Frame(
            frame_id=0,
            detections=[
                Detection(frame_id=0, track_id="1", mask=mask_a),
                Detection(frame_id=0, track_id="2", mask=mask_b),
            ],
        ),
        Frame(
            frame_id=1,
            detections=[
                Detection(frame_id=1, track_id="1", mask=mask_a),
                Detection(frame_id=1, track_id="2", mask=mask_b),
            ],
        ),
    ],
)

pred = Sequence(
    name="toy_scene",
    frames=[
        Frame(
            frame_id=0,
            detections=[
                Detection(frame_id=0, track_id="10", mask=mask_a),
                Detection(frame_id=0, track_id="20", mask=mask_b),
            ],
        ),
        Frame(
            frame_id=1,
            detections=[
                Detection(frame_id=1, track_id="10", mask=mask_a),
                Detection(frame_id=1, track_id="30", mask=mask_b),  # ID switch on track 2
            ],
        ),
    ],
)

matcher = MaskIoUMatcher(threshold=0.5)
metrics = [
    DetectionCountsMetric(),
    IDSwitchesMetric(),
    MOTAMetric(),
    MOTPMetric(),
    IDF1Metric(),
    TemporalIoUMetric(),
]

evaluator = TrackingEvaluator(matcher=matcher, metrics=metrics)
results = evaluator.evaluate(gt, pred)

print_results_table(results)
