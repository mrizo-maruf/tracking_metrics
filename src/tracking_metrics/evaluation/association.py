from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from tracking_metrics.evaluation.evaluator import EvaluationResult


@dataclass
class FrameHOTAData:
    frame_id: int
    gt_track_ids: list[str]
    pred_track_ids: list[str]
    similarity_matrix: np.ndarray  # shape (n_gt, n_pred)


@dataclass
class HOTAData:
    frames: list[FrameHOTAData] = field(default_factory=list)
    gt_track_frame_counts: dict[str, int] = field(default_factory=dict)
    pred_track_frame_counts: dict[str, int] = field(default_factory=dict)


def evaluation_result_to_hota_data(result: EvaluationResult) -> HOTAData:
    frames: list[FrameHOTAData] = []
    gt_counts: dict[str, int] = defaultdict(int)
    pred_counts: dict[str, int] = defaultdict(int)

    for fr in result.frame_results:
        gt_ids = [d.track_id for d in fr.gt_detections]
        pred_ids = [d.track_id for d in fr.pred_detections]

        for tid in gt_ids:
            gt_counts[tid] += 1
        for tid in pred_ids:
            pred_counts[tid] += 1

        if fr.similarity_matrix is not None:
            sim = fr.similarity_matrix
        else:
            sim = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)

        frames.append(
            FrameHOTAData(
                frame_id=fr.frame_id,
                gt_track_ids=gt_ids,
                pred_track_ids=pred_ids,
                similarity_matrix=sim,
            )
        )

    return HOTAData(
        frames=frames,
        gt_track_frame_counts=dict(gt_counts),
        pred_track_frame_counts=dict(pred_counts),
    )
