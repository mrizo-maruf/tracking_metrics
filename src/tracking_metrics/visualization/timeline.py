from __future__ import annotations

from pathlib import Path
from typing import Any

from tracking_metrics.evaluation.evaluator import EvaluationResult
from tracking_metrics.visualization.utils import require_matplotlib, save_figure


def plot_frame_timeline(
    result: EvaluationResult,
    save_path: str | Path | None = None,
    title: str = "Per-Frame Tracking Events",
) -> Any:
    """Stacked bar chart of TP / FP / FN / ID-switch counts per frame.

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

    frame_ids = [fr.frame_id for fr in result.frame_results]
    tps = [len(fr.matches) for fr in result.frame_results]
    fps = [len(fr.false_positives) for fr in result.frame_results]
    fns = [len(fr.false_negatives) for fr in result.frame_results]
    idsws = [len(fr.id_switches) for fr in result.frame_results]

    x = list(range(len(frame_ids)))
    width = 0.8

    fig, ax = plt.subplots(figsize=(max(8, len(frame_ids) * 0.3 + 2), 4))

    ax.bar(x, tps, width, label="TP", color="#2ca02c", alpha=0.85)
    ax.bar(x, fps, width, bottom=tps, label="FP", color="#d62728", alpha=0.85)
    bottom_fn = [t + f for t, f in zip(tps, fps)]
    ax.bar(x, fns, width, bottom=bottom_fn, label="FN", color="#ff7f0e", alpha=0.85)
    bottom_idsw = [b + fn for b, fn in zip(bottom_fn, fns)]
    ax.bar(x, idsws, width, bottom=bottom_idsw, label="IDSW", color="#9467bd", alpha=0.85)

    tick_step = max(1, len(frame_ids) // 20)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([str(frame_ids[i]) for i in range(0, len(frame_ids), tick_step)], rotation=45)
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_figure(fig, save_path)
    return fig


def plot_track_coverage(
    result: EvaluationResult,
    save_path: str | Path | None = None,
    title: str = "Track Coverage",
) -> Any:
    """Horizontal bar showing which frames each GT track is active in.

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

    track_frames: dict[str, list[int]] = {}
    for fr in result.frame_results:
        for det in fr.gt_detections:
            track_frames.setdefault(det.track_id, []).append(fr.frame_id)

    if not track_frames:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No tracks found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        save_figure(fig, save_path)
        return fig

    all_frames = sorted({fr.frame_id for fr in result.frame_results})
    frame_to_idx = {fid: i for i, fid in enumerate(all_frames)}

    track_ids = sorted(track_frames.keys())
    fig_height = max(3, len(track_ids) * 0.35 + 1)
    fig, ax = plt.subplots(figsize=(max(8, len(all_frames) * 0.15 + 2), fig_height))

    for row, tid in enumerate(track_ids):
        xs = [frame_to_idx[f] for f in track_frames[tid]]
        ax.scatter(xs, [row] * len(xs), marker="|", s=200, linewidths=1.5, color="#1f77b4")

    ax.set_yticks(range(len(track_ids)))
    ax.set_yticklabels(track_ids, fontsize=7)
    ax.set_xlabel("Frame index")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    save_figure(fig, save_path)
    return fig


def plot_track_survival(
    result: EvaluationResult,
    save_path: str | Path | None = None,
    title: str = "Track Survival Rate",
) -> Any:
    """Line plot of the fraction of GT tracks still alive at each frame offset.

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

    track_start: dict[str, int] = {}
    track_last: dict[str, int] = {}

    for fr in result.frame_results:
        for det in fr.gt_detections:
            tid = det.track_id
            if tid not in track_start:
                track_start[tid] = fr.frame_id
            track_last[tid] = fr.frame_id

    if not track_start:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No tracks found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        save_figure(fig, save_path)
        return fig

    track_lengths = {tid: track_last[tid] - track_start[tid] + 1 for tid in track_start}
    max_len = max(track_lengths.values())
    n_tracks = len(track_start)

    offsets = list(range(max_len + 1))
    survival = [
        sum(1 for l in track_lengths.values() if l > offset) / n_tracks
        for offset in offsets
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(offsets, survival, color="#1f77b4", linewidth=2)
    ax.fill_between(offsets, survival, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Frame offset from track start")
    ax.set_ylabel("Fraction of tracks alive")
    ax.set_title(title)
    ax.set_xlim(0, max_len)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save_figure(fig, save_path)
    return fig
