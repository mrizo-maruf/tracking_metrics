from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from tracking_metrics.evaluation.evaluator import EvaluationResult
from tracking_metrics.visualization.utils import require_matplotlib, save_figure


def plot_association_heatmap(
    result: EvaluationResult,
    save_path: str | Path | None = None,
    title: str = "GT–Pred Association Heatmap",
    normalize: bool = True,
) -> Any:
    """Heatmap of matched frames between each (GT track, Pred track) pair.

    Parameters
    ----------
    result:
        Output of ``TrackingEvaluator.evaluate_events``.
    save_path:
        Optional path to save the figure.
    normalize:
        If True, values are divided by GT track length (fraction of frames matched).

    Returns
    -------
    matplotlib.figure.Figure
    """
    mpl, plt = require_matplotlib()
    import numpy as np

    # Count co-occurrence (matched frames) per (gt_id, pred_id) pair
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    gt_total: dict[str, int] = defaultdict(int)

    for fr in result.frame_results:
        for det in fr.gt_detections:
            gt_total[det.track_id] += 1
        for match in fr.matches:
            counts[match.gt.track_id][match.pred.track_id] += 1

    gt_ids = sorted(counts.keys())
    pred_ids = sorted({pid for row in counts.values() for pid in row})

    if not gt_ids or not pred_ids:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No associations found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        save_figure(fig, save_path)
        return fig

    matrix = np.zeros((len(gt_ids), len(pred_ids)))
    for i, gid in enumerate(gt_ids):
        for j, pid in enumerate(pred_ids):
            val = counts[gid].get(pid, 0)
            if normalize and gt_total[gid] > 0:
                val = val / gt_total[gid]
            matrix[i, j] = val

    fig_w = max(6, len(pred_ids) * 0.6 + 2)
    fig_h = max(4, len(gt_ids) * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, label="Fraction matched" if normalize else "Matched frames")

    ax.set_xticks(range(len(pred_ids)))
    ax.set_xticklabels(pred_ids, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(gt_ids)))
    ax.set_yticklabels(gt_ids, fontsize=7)
    ax.set_xlabel("Predicted track ID")
    ax.set_ylabel("GT track ID")
    ax.set_title(title)
    fig.tight_layout()

    save_figure(fig, save_path)
    return fig
