from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking_metrics.data.detection import Detection
from tracking_metrics.geometry.box3d_ops import box3d_iou_axis_aligned
from tracking_metrics.matching.base import MatchResult
from tracking_metrics.matching.hungarian import hungarian_match_from_similarity


@dataclass
class Box3DIoUMatcher:
    """Match detections using axis-aligned 3D IoU. Yaw is ignored."""

    threshold: float = 0.25
    class_aware: bool = False

    def match(self, gt: list[Detection], pred: list[Detection]) -> MatchResult:
        n, m = len(gt), len(pred)
        similarity_matrix = np.zeros((n, m), dtype=np.float64)

        for i, g in enumerate(gt):
            for j, p in enumerate(pred):
                if g.bbox3d is None or p.bbox3d is None:
                    continue
                if (
                    self.class_aware
                    and g.class_id is not None
                    and p.class_id is not None
                    and g.class_id != p.class_id
                ):
                    continue
                similarity_matrix[i, j] = box3d_iou_axis_aligned(g.bbox3d, p.bbox3d)

        return hungarian_match_from_similarity(similarity_matrix, self.threshold)
