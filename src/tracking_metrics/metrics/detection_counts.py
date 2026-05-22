from __future__ import annotations

from tracking_metrics.evaluation.evaluator import EvaluationResult


class DetectionCountsMetric:
    name = "DetectionCounts"

    def compute(self, result: EvaluationResult) -> dict[str, float | int]:
        tp = len(result.all_matches())
        fp = len(result.all_false_positives())
        fn = len(result.all_false_negatives())
        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "GT": tp + fn,
            "Pred": tp + fp,
        }
