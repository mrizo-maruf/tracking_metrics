from __future__ import annotations

from typing import Any

from tracking_metrics.evaluation.evaluator import EvaluationResult


class TrackSurvivalRateMetric:
    """Fraction of GT tracks that appear in at least one matched detection."""

    def compute(self, result: EvaluationResult) -> dict[str, Any]:
        gt_tracks: set[str] = set()
        survived: set[str] = set()

        for fr in result.frame_results:
            for m in fr.matches:
                gt_tracks.add(m.gt.track_id)
                survived.add(m.gt.track_id)
            for fn in fr.false_negatives:
                gt_tracks.add(fn.track_id)

        n = len(gt_tracks)
        return {
            "T-SR": len(survived) / n if n > 0 else 0.0,
            "SurvivedTracks": len(survived),
            "TotalGTTracks": n,
        }
