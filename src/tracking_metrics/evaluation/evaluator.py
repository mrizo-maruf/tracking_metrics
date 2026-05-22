from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tracking_metrics.data.detection import Detection
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.events import FrameResult, IDSwitchEvent, Match
from tracking_metrics.evaluation.identity_history import IdentityHistory
from tracking_metrics.matching.base import Matcher

if TYPE_CHECKING:
    from tracking_metrics.metrics.base import Metric


@dataclass
class EvaluationResult:
    sequence_name: str
    frame_results: list[FrameResult] = field(default_factory=list)

    def all_matches(self) -> list[Match]:
        return [m for fr in self.frame_results for m in fr.matches]

    def all_false_positives(self) -> list[Detection]:
        return [fp for fr in self.frame_results for fp in fr.false_positives]

    def all_false_negatives(self) -> list[Detection]:
        return [fn for fr in self.frame_results for fn in fr.false_negatives]

    def all_id_switches(self) -> list[IDSwitchEvent]:
        return [idsw for fr in self.frame_results for idsw in fr.id_switches]


class TrackingEvaluator:
    def __init__(self, matcher: Matcher, metrics: list[Metric]) -> None:
        self._matcher = matcher
        self._metrics = metrics

    def evaluate_events(
        self,
        gt_sequence: Sequence,
        pred_sequence: Sequence,
    ) -> EvaluationResult:
        all_frame_ids = sorted(
            set(gt_sequence.frame_ids()) | set(pred_sequence.frame_ids())
        )
        history = IdentityHistory()
        frame_results: list[FrameResult] = []

        for fid in all_frame_ids:
            gt_frame = gt_sequence.get_frame(fid)
            pred_frame = pred_sequence.get_frame(fid)

            gt_dets = gt_frame.detections
            pred_dets = pred_frame.detections

            match_result = self._matcher.match(gt_dets, pred_dets)

            matches: list[Match] = []
            for gt_idx, pred_idx in match_result.matches:
                sim = float(match_result.similarity_matrix[gt_idx, pred_idx])
                matches.append(
                    Match(
                        frame_id=fid,
                        gt=gt_dets[gt_idx],
                        pred=pred_dets[pred_idx],
                        similarity=sim,
                    )
                )

            false_negatives = [gt_dets[i] for i in match_result.unmatched_gt]
            false_positives = [pred_dets[j] for j in match_result.unmatched_pred]

            id_switches = history.update(fid, matches)

            frame_results.append(
                FrameResult(
                    frame_id=fid,
                    gt_detections=gt_dets,
                    pred_detections=pred_dets,
                    matches=matches,
                    false_positives=false_positives,
                    false_negatives=false_negatives,
                    id_switches=id_switches,
                    similarity_matrix=match_result.similarity_matrix,
                    distance_matrix=match_result.distance_matrix,
                )
            )

        return EvaluationResult(
            sequence_name=gt_sequence.name,
            frame_results=frame_results,
        )

    def evaluate(
        self,
        gt_sequence: Sequence,
        pred_sequence: Sequence,
    ) -> dict[str, Any]:
        result = self.evaluate_events(gt_sequence, pred_sequence)
        output: dict[str, Any] = {}
        for metric in self._metrics:
            output.update(metric.compute(result))
        return output
