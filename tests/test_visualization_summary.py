"""Tests for visualization.summary."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from tracking_metrics.visualization.summary import plot_metric_summary


def test_plot_metric_summary_basic():
    results = {"MOTA": 0.75, "MOTP": 0.62, "IDF1": 0.80}
    fig = plot_metric_summary(results)
    assert fig is not None


def test_plot_metric_summary_save(tmp_path):
    results = {"MOTA": 0.75, "MOTP": 0.62}
    out = tmp_path / "summary.png"
    plot_metric_summary(results, save_path=out)
    assert out.exists()


def test_plot_metric_summary_explicit_metrics():
    results = {"MOTA": 0.5, "MOTP": 0.3, "extra": 99}
    fig = plot_metric_summary(results, metrics=["MOTA", "MOTP"])
    assert fig is not None


def test_plot_metric_summary_fallback_to_all_scalars():
    results = {"foo": 0.1, "bar": 0.9, "label": "string_ignored"}
    fig = plot_metric_summary(results)
    assert fig is not None


def test_plot_metric_summary_unknown_metrics_ignored():
    results = {"MOTA": 0.6}
    fig = plot_metric_summary(results, metrics=["MOTA", "NONEXISTENT"])
    assert fig is not None
