from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking_metrics.data.detection import Detection
from tracking_metrics.geometry.distance_ops import center_distance_3d
from tracking_metrics.matching.base import MatchResult


@dataclass
class CenterDistanceMatcher:
    """Match 3D detections by Euclidean center distance.

    similarity = max(0, 1 - distance / max_distance)

    Matches with distance > max_distance are rejected.
    MOTP will reflect the normalized similarity (not raw distance);
    use MeanCenterDist3D metric for the raw average distance.
    """

    max_distance: float = 0.5
    class_aware: bool = False

    def match(self, gt: list[Detection], pred: list[Detection]) -> MatchResult:
        n, m = len(gt), len(pred)
        distance_matrix = np.full((n, m), fill_value=np.inf, dtype=np.float64)

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
                distance_matrix[i, j] = center_distance_3d(g.bbox3d, p.bbox3d)

        # similarity = max(0, 1 - dist / max_distance); inf distance => 0
        with np.errstate(invalid="ignore"):
            similarity_matrix = np.where(
                np.isfinite(distance_matrix),
                np.maximum(0.0, 1.0 - distance_matrix / self.max_distance),
                0.0,
            )

        if n == 0 or m == 0:
            return MatchResult(
                matches=[],
                unmatched_gt=list(range(n)),
                unmatched_pred=list(range(m)),
                similarity_matrix=similarity_matrix,
                distance_matrix=distance_matrix,
            )

        # Minimize distance (treat inf as a large finite cost)
        cost = np.where(np.isfinite(distance_matrix), distance_matrix, 1e9)
        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        for r, c in zip(row_ind, col_ind, strict=True):
            dist = distance_matrix[r, c]
            if np.isfinite(dist) and dist <= self.max_distance:
                matches.append((int(r), int(c)))
                matched_gt.add(int(r))
                matched_pred.add(int(c))

        unmatched_gt = [i for i in range(n) if i not in matched_gt]
        unmatched_pred = [j for j in range(m) if j not in matched_pred]

        return MatchResult(
            matches=matches,
            unmatched_gt=unmatched_gt,
            unmatched_pred=unmatched_pred,
            similarity_matrix=similarity_matrix,
            distance_matrix=distance_matrix,
        )
