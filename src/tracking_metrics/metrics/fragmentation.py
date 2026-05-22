from __future__ import annotations

from collections import defaultdict
from typing import Any

from tracking_metrics.evaluation.evaluator import EvaluationResult


class FragmentationsMetric:
    """Count of times a GT track breaks (disappears then reappears)."""

    def compute(self, result: EvaluationResult) -> dict[str, Any]:
        gt_all_frames: dict[str, list[int]] = defaultdict(list)
        gt_matched_frames: dict[str, set[int]] = defaultdict(set)

        for fr in result.frame_results:
            for m in fr.matches:
                gt_all_frames[m.gt.track_id].append(fr.frame_id)
                gt_matched_frames[m.gt.track_id].add(fr.frame_id)
            for fn in fr.false_negatives:
                gt_all_frames[fn.track_id].append(fr.frame_id)

        total_frags = 0
        for gt_id, all_frames in gt_all_frames.items():
            matched = gt_matched_frames.get(gt_id, set())
            sorted_frames = sorted(set(all_frames))
            segments = 0
            in_segment = False
            for fid in sorted_frames:
                if fid in matched:
                    if not in_segment:
                        segments += 1
                        in_segment = True
                else:
                    in_segment = False
            total_frags += max(0, segments - 1)

        return {"Frag": total_frags}
