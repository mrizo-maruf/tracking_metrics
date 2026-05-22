from __future__ import annotations

from tracking_metrics.evaluation.evaluator import EvaluationResult


class MOTAMetric:
    name = "MOTA"

    def compute(self, result: EvaluationResult) -> dict[str, float | int]:
        tp = len(result.all_matches())
        fp = len(result.all_false_positives())
        fn = len(result.all_false_negatives())
        idsw = len(result.all_id_switches())
        gt = tp + fn

        if gt == 0:
            return {"MOTA": 0.0}

        mota = 1.0 - (fn + fp + idsw) / gt
        return {"MOTA": mota}
