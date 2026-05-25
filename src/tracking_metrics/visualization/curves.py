from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tracking_metrics.visualization.utils import require_matplotlib, save_figure

_DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist()

_CURVE_STYLES: dict[str, dict[str, Any]] = {
    "HOTA_curve":  {"label": "HOTA",  "color": "#1f77b4", "linewidth": 2.5},
    "DetA_curve":  {"label": "DetA",  "color": "#2ca02c", "linestyle": "--"},
    "AssA_curve":  {"label": "AssA",  "color": "#d62728", "linestyle": "--"},
    "LocA_curve":  {"label": "LocA",  "color": "#9467bd", "linestyle": ":"},
    "OWTA_curve":  {"label": "OWTA",  "color": "#8c564b", "linestyle": "-."},
    "DetRe_curve": {"label": "DetRe", "color": "#17becf", "linestyle": ":"},
    "DetPr_curve": {"label": "DetPr", "color": "#bcbd22", "linestyle": ":"},
    "AssRe_curve": {"label": "AssRe", "color": "#e377c2", "linestyle": ":"},
    "AssPr_curve": {"label": "AssPr", "color": "#7f7f7f", "linestyle": ":"},
}


def plot_hota_curves(
    results: dict[str, Any],
    save_path: str | Path | None = None,
    title: str = "HOTA Curves",
) -> Any:
    """Plot HOTA threshold curves.

    Parameters
    ----------
    results:
        Metrics dict containing curve lists (run HOTA with ``return_curves=True``).
    save_path:
        Optional path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If no curve data is present in *results*.
    """
    mpl, plt = require_matplotlib()

    thresholds: list[float] = results.get("thresholds", _DEFAULT_THRESHOLDS)

    curve_keys = [k for k in _CURVE_STYLES if k in results]
    if not curve_keys:
        raise ValueError(
            "No HOTA curve data found in results. "
            "Run HOTAMetric with return_curves=True:\n"
            "    HOTA(return_curves=True)"
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    for key in curve_keys:
        style = _CURVE_STYLES[key]
        ax.plot(thresholds, results[key], **style)

    ax.set_xlabel("Localization threshold (α)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save_figure(fig, save_path)
    return fig


def plot_metric_curve(
    thresholds: list[float],
    curves: dict[str, list[float]],
    save_path: str | Path | None = None,
    title: str = "Metric Curves",
) -> Any:
    """Generic multi-curve plot over a shared threshold axis.

    Parameters
    ----------
    thresholds:
        x-axis values.
    curves:
        Mapping of curve name → values list (same length as *thresholds*).
    save_path:
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mpl, plt = require_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, values in curves.items():
        ax.plot(thresholds, values, label=name, marker=".")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save_figure(fig, save_path)
    return fig
