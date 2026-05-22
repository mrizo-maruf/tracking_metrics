"""Example: evaluate 3D tracking with Box3DIoUMatcher and CenterDistanceMatcher."""

from __future__ import annotations

from pathlib import Path

from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box3d_iou_matcher import Box3DIoUMatcher
from tracking_metrics.matching.center_distance_matcher import CenterDistanceMatcher
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric
from tracking_metrics.metrics.id_switches import IDSwitchesMetric
from tracking_metrics.metrics.idf1 import IDF1Metric
from tracking_metrics.metrics.localization3d import MeanBox3DIoU, MeanCenterDistance3D
from tracking_metrics.metrics.mota import MOTAMetric
from tracking_metrics.metrics.motp import MOTPMetric
from tracking_metrics.report.terminal import print_results_table

here = Path(__file__).parent
gt = Sequence.from_json(here / "gt_3d.json")
pred = Sequence.from_json(here / "pred_3d.json")

all_metrics = [
    DetectionCountsMetric(),
    IDSwitchesMetric(),
    MOTAMetric(),
    MOTPMetric(),
    IDF1Metric(),
    MeanBox3DIoU(),
    MeanCenterDistance3D(),
]

print("=== Box3D IoU Matcher (threshold=0.25) ===")
evaluator_iou = TrackingEvaluator(
    matcher=Box3DIoUMatcher(threshold=0.25),
    metrics=all_metrics,
)
print_results_table(evaluator_iou.evaluate(gt, pred))

print()
print("=== Center Distance Matcher (max_distance=0.5 m) ===")
evaluator_dist = TrackingEvaluator(
    matcher=CenterDistanceMatcher(max_distance=0.5),
    metrics=all_metrics,
)
print_results_table(evaluator_dist.evaluate(gt, pred))
