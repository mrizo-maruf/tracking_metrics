from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from tracking_metrics.evaluation.association import HOTAData, evaluation_result_to_hota_data
from tracking_metrics.evaluation.evaluator import EvaluationResult
from tracking_metrics.matching.hungarian import hungarian_match_from_similarity

_EPS = np.finfo(float).eps

_DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


class HOTAMetric:
    def __init__(
        self,
        thresholds: np.ndarray | list[float] | None = None,
        return_curves: bool = False,
    ) -> None:
        self._thresholds = (
            np.asarray(thresholds, dtype=float)
            if thresholds is not None
            else _DEFAULT_THRESHOLDS
        )
        self._return_curves = return_curves

    def compute(self, result: EvaluationResult) -> dict[str, Any]:
        hota_data = evaluation_result_to_hota_data(result)

        curves: dict[str, list[float]] = defaultdict(list)

        for alpha in self._thresholds:
            vals = self._compute_at_threshold(hota_data, float(alpha))
            for k, v in vals.items():
                curves[k].append(v)

        out: dict[str, Any] = {
            "HOTA": float(np.mean(curves["hota"])),
            "DetA": float(np.mean(curves["det_a"])),
            "AssA": float(np.mean(curves["ass_a"])),
            "LocA": float(np.mean(curves["loc_a"])),
            "OWTA": float(np.mean(curves["owta"])),
            "DetRe": float(np.mean(curves["det_re"])),
            "DetPr": float(np.mean(curves["det_pr"])),
            "AssRe": float(np.mean(curves["ass_re"])),
            "AssPr": float(np.mean(curves["ass_pr"])),
        }

        if self._return_curves:
            out["thresholds"] = self._thresholds.tolist()
            out["HOTA_curve"] = curves["hota"]
            out["DetA_curve"] = curves["det_a"]
            out["AssA_curve"] = curves["ass_a"]
            out["LocA_curve"] = curves["loc_a"]
            out["OWTA_curve"] = curves["owta"]
            out["DetRe_curve"] = curves["det_re"]
            out["DetPr_curve"] = curves["det_pr"]
            out["AssRe_curve"] = curves["ass_re"]
            out["AssPr_curve"] = curves["ass_pr"]

        return out

    def _compute_at_threshold(
        self, hota_data: HOTAData, alpha: float
    ) -> dict[str, float]:
        # A[gt_id][pred_id] = number of matched frames at this threshold
        A: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        tp = 0
        fp = 0
        fn = 0
        total_sim = 0.0
        tp_pairs: list[tuple[str, str]] = []

        for frame in hota_data.frames:
            n_gt = len(frame.gt_track_ids)
            n_pred = len(frame.pred_track_ids)

            if n_gt == 0 and n_pred == 0:
                continue
            if n_gt == 0:
                fp += n_pred
                continue
            if n_pred == 0:
                fn += n_gt
                continue

            match_result = hungarian_match_from_similarity(
                frame.similarity_matrix, alpha
            )

            for gi, pj in match_result.matches:
                gt_id = frame.gt_track_ids[gi]
                pred_id = frame.pred_track_ids[pj]
                A[gt_id][pred_id] += 1
                total_sim += float(frame.similarity_matrix[gi, pj])
                tp_pairs.append((gt_id, pred_id))
                tp += 1

            fn += len(match_result.unmatched_gt)
            fp += len(match_result.unmatched_pred)

        # Detection scores
        det_re = tp / (tp + fn + _EPS)
        det_pr = tp / (tp + fp + _EPS)
        det_a = np.sqrt(det_re * det_pr)
        loc_a = total_sim / (tp + _EPS)

        # Association scores
        if tp > 0:
            ass_sum = ass_re_sum = ass_pr_sum = 0.0
            for gt_id, _pred_id in tp_pairs:
                best_pred = max(A[gt_id], key=lambda p: A[gt_id][p])
                tpa = A[gt_id][best_pred]
                n_gt_t = hota_data.gt_track_frame_counts.get(gt_id, 0)
                n_pred_t = hota_data.pred_track_frame_counts.get(best_pred, 0)
                ass_sum += tpa / (n_gt_t + n_pred_t - tpa + _EPS)
                ass_re_sum += tpa / (n_gt_t + _EPS)
                ass_pr_sum += tpa / (n_pred_t + _EPS)
            ass_a = ass_sum / tp
            ass_re = ass_re_sum / tp
            ass_pr = ass_pr_sum / tp
        else:
            ass_a = ass_re = ass_pr = 0.0

        hota = float(np.sqrt(det_a * ass_a))
        owta = float(np.sqrt(det_re * ass_a))

        return {
            "hota": hota,
            "det_a": float(det_a),
            "ass_a": float(ass_a),
            "loc_a": float(loc_a),
            "owta": owta,
            "det_re": float(det_re),
            "det_pr": float(det_pr),
            "ass_re": float(ass_re),
            "ass_pr": float(ass_pr),
        }
