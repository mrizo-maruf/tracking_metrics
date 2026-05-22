from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from tracking_metrics.evaluation.evaluator import EvaluationResult


class IDConsistencyMetric:
    """Average fraction of matched frames where each GT track uses its most-common pred ID."""

    def compute(self, result: EvaluationResult) -> dict[str, Any]:
        gt_to_preds: dict[str, list[str]] = defaultdict(list)

        for fr in result.frame_results:
            for m in fr.matches:
                gt_to_preds[m.gt.track_id].append(m.pred.track_id)

        if not gt_to_preds:
            return {"IDCons": 0.0}

        consistencies = []
        for pred_ids in gt_to_preds.values():
            most_common_count = Counter(pred_ids).most_common(1)[0][1]
            consistencies.append(most_common_count / len(pred_ids))

        return {"IDCons": float(np.mean(consistencies))}
