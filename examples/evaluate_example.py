"""Example: evaluate tracking with the Python API."""

from pathlib import Path

from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric
from tracking_metrics.metrics.id_switches import IDSwitchesMetric
from tracking_metrics.metrics.idf1 import IDF1Metric
from tracking_metrics.metrics.mota import MOTAMetric
from tracking_metrics.metrics.motp import MOTPMetric
from tracking_metrics.report.terminal import print_results_table

here = Path(__file__).parent

gt = Sequence.from_json(here / "gt.json")
pred = Sequence.from_json(here / "pred.json")

matcher = Box2DIoUMatcher(threshold=0.5)
metrics = [
    DetectionCountsMetric(),
    IDSwitchesMetric(),
    MOTAMetric(),
    MOTPMetric(),
    IDF1Metric(),
]

evaluator = TrackingEvaluator(matcher=matcher, metrics=metrics)
results = evaluator.evaluate(gt, pred)

print_results_table(results)
