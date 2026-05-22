from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking_metrics.matching.base import MatchResult


def hungarian_match_from_similarity(
    similarity_matrix: np.ndarray,
    threshold: float,
) -> MatchResult:
    n, m = similarity_matrix.shape

    if n == 0 or m == 0:
        return MatchResult(
            matches=[],
            unmatched_gt=list(range(n)),
            unmatched_pred=list(range(m)),
            similarity_matrix=similarity_matrix,
        )

    cost_matrix = 1.0 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for r, c in zip(row_ind, col_ind, strict=True):
        if similarity_matrix[r, c] >= threshold:
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
    )
