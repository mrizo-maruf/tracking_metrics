"""Example: run evaluation and generate all visualization plots.

Run from the repo root:
    conda activate da3
    python examples/visualize_results.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.hota import HOTAMetric
from tracking_metrics.metrics.mota import MOTAMetric
from tracking_metrics.metrics.motp import MOTPMetric
from tracking_metrics.visualization import (
    get_id_switch_table,
    plot_association_heatmap,
    plot_frame_overlay,
    plot_frame_timeline,
    plot_hota_curves,
    plot_id_switch_timeline,
    plot_metric_summary,
    plot_track_coverage,
    plot_track_survival,
)


def _box(x1, y1, x2, y2):
    return Box2D(x1=x1, y1=y1, x2=x2, y2=y2)


def _det(frame_id, track_id, box):
    return Detection(frame_id=frame_id, track_id=track_id, bbox2d=box)


# --- Build a small synthetic sequence ---
gt = Sequence(
    name="demo",
    frames=[
        Frame(0, [_det(0, "1", _box(0, 0, 10, 10)), _det(0, "2", _box(50, 50, 60, 60))]),
        Frame(1, [_det(1, "1", _box(2, 2, 12, 12)), _det(1, "2", _box(52, 52, 62, 62))]),
        Frame(2, [_det(2, "1", _box(4, 4, 14, 14))]),
        Frame(3, [_det(3, "1", _box(6, 6, 16, 16))]),
        Frame(4, [_det(4, "1", _box(8, 8, 18, 18)), _det(4, "3", _box(80, 80, 90, 90))]),
    ],
)

pred = Sequence(
    name="demo",
    frames=[
        Frame(0, [_det(0, "a", _box(0, 0, 10, 10)), _det(0, "b", _box(50, 50, 60, 60))]),
        Frame(1, [_det(1, "a", _box(2, 2, 12, 12))]),             # track 2 missed
        Frame(2, [_det(2, "a", _box(4, 4, 14, 14))]),
        Frame(3, [_det(3, "c", _box(6, 6, 16, 16))]),             # ID switch on track 1
        Frame(4, [_det(4, "c", _box(8, 8, 18, 18))]),
    ],
)

# --- Run evaluation ---
evaluator = TrackingEvaluator(
    matcher=Box2DIoUMatcher(threshold=0.3),
    metrics=[MOTAMetric(), MOTPMetric(), HOTAMetric(return_curves=True)],
)
eval_result = evaluator.evaluate_events(gt, pred)
metric_results = evaluator.compute_metrics(eval_result)

print("=== Metrics ===")
for k, v in metric_results.items():
    if isinstance(v, (int, float)):
        print(f"  {k}: {v:.4f}")

# --- Save plots ---
out = Path("plots_demo")
out.mkdir(exist_ok=True)

plot_metric_summary(metric_results, save_path=out / "summary.png", title="Demo Summary")
print(f"Saved {out}/summary.png")

plot_hota_curves(metric_results, save_path=out / "hota_curves.png", title="Demo HOTA Curves")
print(f"Saved {out}/hota_curves.png")

plot_frame_timeline(eval_result, save_path=out / "frame_timeline.png")
print(f"Saved {out}/frame_timeline.png")

plot_track_coverage(eval_result, save_path=out / "track_coverage.png")
print(f"Saved {out}/track_coverage.png")

plot_track_survival(eval_result, save_path=out / "track_survival.png")
print(f"Saved {out}/track_survival.png")

plot_id_switch_timeline(eval_result, save_path=out / "id_switches.png")
print(f"Saved {out}/id_switches.png")

plot_association_heatmap(eval_result, save_path=out / "association.png")
print(f"Saved {out}/association.png")

plot_frame_overlay(eval_result.frame_results[3], save_path=out / "frame_3_overlay.png")
print(f"Saved {out}/frame_3_overlay.png")

switches = get_id_switch_table(eval_result)
print(f"\n=== ID switch table ({len(switches)} events) ===")
for row in switches:
    print(f"  frame {row['frame_id']}: GT {row['gt_track_id']} — {row['prev_pred_id']} → {row['new_pred_id']}")
