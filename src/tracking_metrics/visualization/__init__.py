"""Optional visualization utilities for tracking-metrics.

Requires matplotlib (``pip install -e '.[viz]'``).
"""
from __future__ import annotations

from tracking_metrics.visualization.association import plot_association_heatmap
from tracking_metrics.visualization.curves import plot_hota_curves, plot_metric_curve
from tracking_metrics.visualization.frame_overlay import plot_frame_overlay
from tracking_metrics.visualization.id_switches import get_id_switch_table, plot_id_switch_timeline
from tracking_metrics.visualization.summary import plot_metric_summary
from tracking_metrics.visualization.timeline import (
    plot_frame_timeline,
    plot_track_coverage,
    plot_track_survival,
)

__all__ = [
    "plot_metric_summary",
    "plot_hota_curves",
    "plot_metric_curve",
    "plot_frame_timeline",
    "plot_track_coverage",
    "plot_track_survival",
    "plot_id_switch_timeline",
    "get_id_switch_table",
    "plot_association_heatmap",
    "plot_frame_overlay",
]
