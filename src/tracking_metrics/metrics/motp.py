from __future__ import annotations

from tracking_metrics.evaluation.evaluator import EvaluationResult


class MOTPMetric:
    name = "MOTP"

    def compute(self, result: EvaluationResult) -> dict[str, float | int]:
        matches = result.all_matches()
        if not matches:
            return {"MOTP": 0.0}
        avg_sim = sum(m.similarity for m in matches) / len(matches)
        return {"MOTP": avg_sim}
