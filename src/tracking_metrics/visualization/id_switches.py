from __future__ import annotations

from pathlib import Path
from typing import Any

from tracking_metrics.evaluation.evaluator import EvaluationResult
from tracking_metrics.visualization.utils import require_matplotlib, save_figure


def plot_id_switch_timeline(
    result: EvaluationResult,
    save_path: str | Path | None = None,
    title: str = "ID Switch Timeline",
) -> Any:
    """Scatter plot marking ID switch events across frames.

    Parameters
    ----------
    result:
        Output of ``TrackingEvaluator.evaluate_events``.
    save_path:
        Optional path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mpl, plt = require_matplotlib()

    switch_frames: list[int] = []
    switch_gt_ids: list[str] = []

    for fr in result.frame_results:
        for sw in fr.id_switches:
            switch_frames.append(fr.frame_id)
            switch_gt_ids.append(sw.gt_track_id)

    all_frame_ids = [fr.frame_id for fr in result.frame_results]

    fig, ax = plt.subplots(figsize=(max(8, len(all_frame_ids) * 0.3 + 2), 3))

    if switch_frames:
        unique_gt = sorted(set(switch_gt_ids))
        gt_to_y = {tid: i for i, tid in enumerate(unique_gt)}
        ys = [gt_to_y[tid] for tid in switch_gt_ids]

        ax.scatter(switch_frames, ys, marker="x", s=80, color="#d62728", linewidths=1.5, zorder=5)
        ax.set_yticks(list(range(len(unique_gt))))
        ax.set_yticklabels(unique_gt, fontsize=7)
        ax.set_ylabel("GT track ID")
    else:
        ax.text(0.5, 0.5, "No ID switches", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Frame ID")
    ax.set_title(f"{title} ({len(switch_frames)} total)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    save_figure(fig, save_path)
    return fig


def get_id_switch_table(result: EvaluationResult) -> list[dict[str, Any]]:
    """Return a list of ID switch events as plain dicts.

    Each dict has keys: ``frame_id``, ``gt_track_id``, ``prev_pred_id``, ``new_pred_id``.

    Parameters
    ----------
    result:
        Output of ``TrackingEvaluator.evaluate_events``.

    Returns
    -------
    list[dict]
    """
    rows: list[dict[str, Any]] = []
    for fr in result.frame_results:
        for sw in fr.id_switches:
            rows.append(
                {
                    "frame_id": fr.frame_id,
                    "gt_track_id": sw.gt_track_id,
                    "prev_pred_id": sw.previous_pred_track_id,
                    "new_pred_id": sw.current_pred_track_id,
                }
            )
    return rows
