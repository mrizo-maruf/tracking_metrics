from __future__ import annotations

from dataclasses import dataclass

from tracking_metrics.evaluation.evaluator import EvaluationResult
from tracking_metrics.geometry.box3d_ops import box3d_iou_axis_aligned
from tracking_metrics.geometry.distance_ops import center_distance_3d


@dataclass
class MeanCenterDistance3D:
    """Average Euclidean center distance over matched pairs that have 3D boxes."""

    name: str = "MeanCenterDist3D"

    def compute(self, result: EvaluationResult) -> dict[str, float | int]:
        distances: list[float] = []
        for match in result.all_matches():
            if match.gt.bbox3d is None or match.pred.bbox3d is None:
                continue
            distances.append(center_distance_3d(match.gt.bbox3d, match.pred.bbox3d))
        return {"MeanCenterDist3D": sum(distances) / len(distances) if distances else 0.0}


@dataclass
class MeanBox3DIoU:
    """Average axis-aligned 3D IoU over matched pairs that have 3D boxes."""

    name: str = "MeanBox3DIoU"

    def compute(self, result: EvaluationResult) -> dict[str, float | int]:
        ious: list[float] = []
        for match in result.all_matches():
            if match.gt.bbox3d is None or match.pred.bbox3d is None:
                continue
            ious.append(box3d_iou_axis_aligned(match.gt.bbox3d, match.pred.bbox3d))
        return {"MeanBox3DIoU": sum(ious) / len(ious) if ious else 0.0}
