from __future__ import annotations

from tracking_metrics.evaluation.evaluator import EvaluationResult


class IDSwitchesMetric:
    name = "IDSW"

    def compute(self, result: EvaluationResult) -> dict[str, float | int]:
        return {"IDSW": len(result.all_id_switches())}
