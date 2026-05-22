from __future__ import annotations

from collections import defaultdict
from typing import Any

from tracking_metrics.evaluation.evaluator import EvaluationResult


class TrackCoverageMetric:
    """Classify GT tracks as Mostly Tracked / Partially Tracked / Mostly Lost.

    MT: track coverage >= mt_threshold (default 0.8)
    ML: track coverage <  ml_threshold (default 0.2)
    PT: otherwise
    """

    def __init__(self, mt_threshold: float = 0.8, ml_threshold: float = 0.2) -> None:
        self.mt_threshold = mt_threshold
        self.ml_threshold = ml_threshold

    def compute(self, result: EvaluationResult) -> dict[str, Any]:
        gt_total: dict[str, int] = defaultdict(int)
        gt_matched: dict[str, int] = defaultdict(int)

        for fr in result.frame_results:
            for m in fr.matches:
                gt_total[m.gt.track_id] += 1
                gt_matched[m.gt.track_id] += 1
            for fn in fr.false_negatives:
                gt_total[fn.track_id] += 1

        mt = pt = ml = 0
        for gt_id, total in gt_total.items():
            ratio = gt_matched.get(gt_id, 0) / total if total > 0 else 0.0
            if ratio >= self.mt_threshold:
                mt += 1
            elif ratio >= self.ml_threshold:
                pt += 1
            else:
                ml += 1

        n_tracks = mt + pt + ml
        return {
            "MT": mt,
            "PT": pt,
            "ML": ml,
            "MT%": mt / n_tracks if n_tracks > 0 else 0.0,
            "PT%": pt / n_tracks if n_tracks > 0 else 0.0,
            "ML%": ml / n_tracks if n_tracks > 0 else 0.0,
        }
