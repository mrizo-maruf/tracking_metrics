from __future__ import annotations

from dataclasses import dataclass

from tracking_metrics.evaluation.evaluator import EvaluationResult
from tracking_metrics.geometry.mask_ops import mask_dice, mask_iou


@dataclass
class TemporalIoUMetric:
    name: str = "T-mIoU"

    def compute(self, result: EvaluationResult) -> dict[str, float | int]:
        ious: list[float] = []
        for match in result.all_matches():
            gt_mask = match.gt.mask
            pred_mask = match.pred.mask
            if gt_mask is None or pred_mask is None:
                continue
            ious.append(mask_iou(gt_mask, pred_mask))

        return {"T-mIoU": sum(ious) / len(ious) if ious else 0.0}


@dataclass
class TemporalDiceMetric:
    name: str = "T-Dice"

    def compute(self, result: EvaluationResult) -> dict[str, float | int]:
        scores: list[float] = []
        for match in result.all_matches():
            gt_mask = match.gt.mask
            pred_mask = match.pred.mask
            if gt_mask is None or pred_mask is None:
                continue
            scores.append(mask_dice(gt_mask, pred_mask))

        return {"T-Dice": sum(scores) / len(scores) if scores else 0.0}
