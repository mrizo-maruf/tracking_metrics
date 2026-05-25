"""Tests for visualization.curves."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from tracking_metrics.visualization.curves import plot_hota_curves, plot_metric_curve

_THRESHOLDS = list(np.round(np.arange(0.05, 1.0, 0.05), 2))
_N = len(_THRESHOLDS)


def _make_results(include_curves: bool = True) -> dict:
    results: dict = {
        "HOTA": 0.5,
        "DetA": 0.6,
        "AssA": 0.4,
    }
    if include_curves:
        results["thresholds"] = _THRESHOLDS
        results["HOTA_curve"] = [0.5] * _N
        results["DetA_curve"] = [0.6] * _N
        results["AssA_curve"] = [0.4] * _N
    return results


def test_plot_hota_curves_basic():
    fig = plot_hota_curves(_make_results())
    assert fig is not None


def test_plot_hota_curves_save(tmp_path):
    out = tmp_path / "curves.png"
    plot_hota_curves(_make_results(), save_path=out)
    assert out.exists()


def test_plot_hota_curves_raises_without_curve_data():
    with pytest.raises(ValueError, match="No HOTA curve data"):
        plot_hota_curves({"HOTA": 0.5})


def test_plot_hota_curves_uses_default_thresholds():
    results = {"HOTA_curve": [0.5] * _N}
    fig = plot_hota_curves(results)
    assert fig is not None


def test_plot_metric_curve_basic():
    thresholds = [0.1, 0.2, 0.3]
    curves = {"metric_a": [0.4, 0.5, 0.6], "metric_b": [0.3, 0.35, 0.4]}
    fig = plot_metric_curve(thresholds, curves)
    assert fig is not None


def test_plot_metric_curve_save(tmp_path):
    out = tmp_path / "curves.png"
    plot_metric_curve([0.1, 0.5, 0.9], {"a": [0.3, 0.5, 0.7]}, save_path=out)
    assert out.exists()
