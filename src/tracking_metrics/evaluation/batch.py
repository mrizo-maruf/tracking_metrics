from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tracking_metrics.data.detection import Detection
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import EvaluationResult, TrackingEvaluator
from tracking_metrics.evaluation.events import FrameResult, IDSwitchEvent, Match

if TYPE_CHECKING:
    pass


@dataclass
class SequencePair:
    name: str
    gt: Sequence
    pred: Sequence


def _prefix_detection(d: Detection, prefix: str) -> Detection:
    return Detection(
        frame_id=d.frame_id,
        track_id=f"{prefix}/{d.track_id}",
        class_id=d.class_id,
        score=d.score,
        bbox2d=d.bbox2d,
        mask=d.mask,
        bbox3d=d.bbox3d,
        attributes=d.attributes,
    )


def _prefix_frame_result(fr: FrameResult, prefix: str, frame_offset: int) -> FrameResult:
    """Return a copy of fr with track IDs prefixed and frame_id offset for uniqueness."""
    fid = fr.frame_id + frame_offset

    def pdet(d: Detection) -> Detection:
        return _prefix_detection(d, prefix)

    return FrameResult(
        frame_id=fid,
        gt_detections=[pdet(d) for d in fr.gt_detections],
        pred_detections=[pdet(d) for d in fr.pred_detections],
        matches=[
            Match(
                frame_id=fid,
                gt=pdet(m.gt),
                pred=pdet(m.pred),
                similarity=m.similarity,
            )
            for m in fr.matches
        ],
        false_positives=[pdet(d) for d in fr.false_positives],
        false_negatives=[pdet(d) for d in fr.false_negatives],
        id_switches=[
            IDSwitchEvent(
                frame_id=fid,
                gt_track_id=f"{prefix}/{e.gt_track_id}",
                previous_pred_track_id=f"{prefix}/{e.previous_pred_track_id}",
                current_pred_track_id=f"{prefix}/{e.current_pred_track_id}",
            )
            for e in fr.id_switches
        ],
        similarity_matrix=fr.similarity_matrix,
        distance_matrix=fr.distance_matrix,
    )


class BatchEvaluator:
    """Evaluate a list of sequence pairs and aggregate results."""

    def __init__(self, evaluator: TrackingEvaluator) -> None:
        self._evaluator = evaluator

    def evaluate(self, sequence_pairs: list[SequencePair]) -> dict[str, Any]:
        seq_scores: dict[str, dict[str, Any]] = {}
        merged_frame_results: list[FrameResult] = []
        frame_offset = 0

        for pair in sequence_pairs:
            ev_result = self._evaluator.evaluate_events(pair.gt, pair.pred)

            scores: dict[str, Any] = {}
            for metric in self._evaluator._metrics:
                scores.update(metric.compute(ev_result))
            seq_scores[pair.name] = scores

            max_fid = max((fr.frame_id for fr in ev_result.frame_results), default=-1)
            for fr in ev_result.frame_results:
                merged_frame_results.append(
                    _prefix_frame_result(fr, pair.name, frame_offset)
                )
            frame_offset += max_fid + 1

        # Average: arithmetic mean of per-sequence scalar metrics
        average: dict[str, Any] = {}
        if seq_scores:
            all_keys = {
                k
                for scores in seq_scores.values()
                for k, v in scores.items()
                if isinstance(v, (int, float))
            }
            for key in sorted(all_keys):
                vals = [
                    seq_scores[name][key]
                    for name in seq_scores
                    if key in seq_scores[name] and isinstance(seq_scores[name][key], (int, float))
                ]
                if vals:
                    average[key] = sum(vals) / len(vals)

        # Global: metrics run on merged EvaluationResult (track IDs are prefixed to ensure
        # cross-sequence independence while still computing correct aggregated scores)
        global_scores: dict[str, Any] = {}
        if merged_frame_results:
            merged_result = EvaluationResult(
                sequence_name="__global__",
                frame_results=merged_frame_results,
            )
            for metric in self._evaluator._metrics:
                global_scores.update(metric.compute(merged_result))

        return {
            "sequences": seq_scores,
            "average": average,
            "global": global_scores,
        }
