from __future__ import annotations

from pathlib import Path
from typing import Any

from tracking_metrics.evaluation.events import FrameResult
from tracking_metrics.visualization.utils import require_matplotlib, save_figure


def plot_frame_overlay(
    frame_result: FrameResult,
    save_path: str | Path | None = None,
    title: str | None = None,
    image: Any | None = None,
) -> Any:
    """2-D scatter overlay of GT and predicted detections for a single frame.

    Draws GT detections in green, predictions in blue, and marks ID switches
    with a red ring. When detections carry 2-D bounding boxes, the box centre
    is plotted; otherwise the raw ``(x, y)`` position attribute is used if
    present.

    Parameters
    ----------
    frame_result:
        A single ``FrameResult`` from ``EvaluationResult.frame_results``.
    save_path:
        Optional path to save the figure.
    title:
        Axes title. Defaults to ``"Frame <frame_id>"``.
    image:
        Optional background image (H×W or H×W×C numpy array).

    Returns
    -------
    matplotlib.figure.Figure
    """
    mpl, plt = require_matplotlib()
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 6))

    if image is not None:
        ax.imshow(image, origin="upper")

    def _center(det: Any) -> tuple[float, float] | None:
        box = getattr(det, "box", None)
        if box is not None:
            # BoundingBox2D or similar with x1,y1,x2,y2
            x = (getattr(box, "x1", 0) + getattr(box, "x2", 0)) / 2
            y = (getattr(box, "y1", 0) + getattr(box, "y2", 0)) / 2
            return float(x), float(y)
        # Fall back to direct attributes
        x = getattr(det, "x", None)
        y = getattr(det, "y", None)
        if x is not None and y is not None:
            return float(x), float(y)
        return None

    switch_gt_ids = {sw.gt_track_id for sw in frame_result.id_switches}

    for det in frame_result.gt_detections:
        c = _center(det)
        if c is None:
            continue
        ax.scatter(*c, marker="s", s=80, color="#2ca02c", zorder=3, label="GT" if det is frame_result.gt_detections[0] else "")
        ax.annotate(str(det.track_id), c, fontsize=6, color="#2ca02c", xytext=(3, 3), textcoords="offset points")
        if det.track_id in switch_gt_ids:
            ax.scatter(*c, marker="o", s=200, facecolors="none", edgecolors="#d62728", linewidths=2, zorder=4)

    for det in frame_result.pred_detections:
        c = _center(det)
        if c is None:
            continue
        ax.scatter(*c, marker="^", s=80, color="#1f77b4", zorder=3, label="Pred" if det is frame_result.pred_detections[0] else "")
        ax.annotate(str(det.track_id), c, fontsize=6, color="#1f77b4", xytext=(3, -8), textcoords="offset points")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    if unique_h:
        ax.legend(unique_h, unique_l, loc="upper right", fontsize=8)

    ax.set_title(title or f"Frame {frame_result.frame_id}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    save_figure(fig, save_path)
    return fig
