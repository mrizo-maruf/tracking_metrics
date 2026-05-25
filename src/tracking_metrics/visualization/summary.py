from __future__ import annotations

from pathlib import Path
from typing import Any

from tracking_metrics.visualization.utils import require_matplotlib, save_figure

_COMMON_METRICS = [
    "MOTA", "MOTP", "IDF1", "HOTA", "DetA", "AssA", "LocA",
    "T-mIoU", "T-SR", "IDCons",
]


def plot_metric_summary(
    results: dict[str, Any],
    metrics: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str = "Tracking Metrics Summary",
) -> Any:
    """Bar chart of scalar metrics.

    Parameters
    ----------
    results:
        Flat metrics dict (output of ``TrackingEvaluator.evaluate``).
    metrics:
        Metric keys to include. Defaults to common scalar metrics present in *results*.
    save_path:
        If given, the figure is saved there (PNG/PDF/…).

    Returns
    -------
    matplotlib.figure.Figure
    """
    mpl, plt = require_matplotlib()

    if metrics is not None:
        keys = [k for k in metrics if k in results and isinstance(results[k], (int, float))]
    else:
        keys = [k for k in _COMMON_METRICS if k in results and isinstance(results[k], (int, float))]
        if not keys:
            keys = [k for k, v in results.items() if isinstance(v, (int, float))]

    values = [float(results[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.8 + 1), 4))

    colors = [
        "#2ca02c" if v >= 0.8 else "#ff7f0e" if v >= 0.5 else "#d62728"
        if 0 <= v <= 1 else "#1f77b4"
        for v in values
    ]

    bars = ax.bar(keys, values, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, max(max(values, default=1.0) * 1.15, 0.1))
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_figure(fig, save_path)
    return fig
