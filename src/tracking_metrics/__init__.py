"""tracking-metrics: dataset-independent Python library for multi-object tracking evaluation."""

from __future__ import annotations

__version__ = "0.5.0"

# Core data types
from tracking_metrics.data import Box2D, Box3D, Detection, Frame, Mask2D, Sequence

# Evaluation
from tracking_metrics.evaluation import EvaluationResult, FrameResult, TrackingEvaluator

# Matchers
from tracking_metrics.matching import (
    Box2DIoUMatcher,
    Box3DIoUMatcher,
    CenterDistanceMatcher,
    MaskIoUMatcher,
)

# Metrics — canonical short names
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric as DetectionCounts
from tracking_metrics.metrics.fragmentation import FragmentationsMetric as Fragmentations
from tracking_metrics.metrics.hota import HOTAMetric as HOTA
from tracking_metrics.metrics.id_consistency import IDConsistencyMetric as IDConsistency
from tracking_metrics.metrics.id_switches import IDSwitchesMetric as IDSwitches
from tracking_metrics.metrics.idf1 import IDF1Metric as IDF1
from tracking_metrics.metrics.localization3d import MeanBox3DIoU, MeanCenterDistance3D
from tracking_metrics.metrics.mota import MOTAMetric as MOTA
from tracking_metrics.metrics.motp import MOTPMetric as MOTP
from tracking_metrics.metrics.temporal_iou import TemporalIoUMetric as TemporalIoU
from tracking_metrics.metrics.track_coverage import TrackCoverageMetric as TrackCoverage
from tracking_metrics.metrics.track_survival import TrackSurvivalRateMetric as TrackSurvivalRate

__all__ = [
    "__version__",
    # data
    "Box2D",
    "Box3D",
    "Detection",
    "Frame",
    "Mask2D",
    "Sequence",
    # evaluation
    "EvaluationResult",
    "FrameResult",
    "TrackingEvaluator",
    # matchers
    "Box2DIoUMatcher",
    "Box3DIoUMatcher",
    "CenterDistanceMatcher",
    "MaskIoUMatcher",
    # metrics (short names)
    "DetectionCounts",
    "Fragmentations",
    "HOTA",
    "IDConsistency",
    "IDSwitches",
    "IDF1",
    "MeanBox3DIoU",
    "MeanCenterDistance3D",
    "MOTA",
    "MOTP",
    "TemporalIoU",
    "TrackCoverage",
    "TrackSurvivalRate",
]
