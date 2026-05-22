from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking_metrics.evaluation.evaluator import EvaluationResult


class IDF1Metric:
    name = "IDF1"

    def compute(self, result: EvaluationResult) -> dict[str, float | int]:
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        total_gt = 0
        total_pred = 0

        for fr in result.frame_results:
            total_gt += len(fr.matches) + len(fr.false_negatives)
            total_pred += len(fr.matches) + len(fr.false_positives)
            for match in fr.matches:
                pair_counts[(match.gt.track_id, match.pred.track_id)] += 1

        if not pair_counts:
            return {"IDF1": 0.0, "IDP": 0.0, "IDR": 0.0, "IDTP": 0, "IDFP": total_pred, "IDFN": total_gt}

        gt_ids = sorted({g for g, _ in pair_counts})
        pred_ids = sorted({p for _, p in pair_counts})

        gt_index = {g: i for i, g in enumerate(gt_ids)}
        pred_index = {p: j for j, p in enumerate(pred_ids)}

        cost_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
        for (g, p), count in pair_counts.items():
            cost_matrix[gt_index[g], pred_index[p]] = count

        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        idtp = int(cost_matrix[row_ind, col_ind].sum())

        idfp = total_pred - idtp
        idfn = total_gt - idtp

        idp = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0
        idr = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
        idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0

        return {
            "IDF1": idf1,
            "IDP": idp,
            "IDR": idr,
            "IDTP": idtp,
            "IDFP": idfp,
            "IDFN": idfn,
        }
