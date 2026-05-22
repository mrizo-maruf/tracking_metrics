from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking_metrics.data.detection import Detection
from tracking_metrics.geometry.box2d_ops import box2d_iou
from tracking_metrics.matching.base import MatchResult
from tracking_metrics.matching.hungarian import hungarian_match_from_similarity


@dataclass
class Box2DIoUMatcher:
    threshold: float = 0.5
    class_aware: bool = False

    def match(self, gt: list[Detection], pred: list[Detection]) -> MatchResult:
        n, m = len(gt), len(pred)
        similarity_matrix = np.zeros((n, m), dtype=np.float64)

        for i, g in enumerate(gt):
            for j, p in enumerate(pred):
                if g.bbox2d is None or p.bbox2d is None:
                    continue
                if (
                    self.class_aware
                    and g.class_id is not None
                    and p.class_id is not None
                    and g.class_id != p.class_id
                ):
                    continue
                similarity_matrix[i, j] = box2d_iou(g.bbox2d, p.bbox2d)

        return hungarian_match_from_similarity(similarity_matrix, self.threshold)
