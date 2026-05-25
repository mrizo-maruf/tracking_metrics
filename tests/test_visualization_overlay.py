"""Tests for visualization.frame_overlay."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from tests.conftest import make_box, make_det, make_sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.visualization.frame_overlay import plot_frame_overlay


def _make_frame_result():
    box_a = make_box(5, 5, 15, 15)
    box_b = make_box(50, 50, 60, 60)
    gt = make_sequence("s", [
        (0, [make_det(0, "1", box_a), make_det(0, "2", box_b)]),
    ])
    pred = make_sequence("s", [
        (0, [make_det(0, "x", box_a)]),
    ])
    evaluator = TrackingEvaluator(matcher=Box2DIoUMatcher(threshold=0.3), metrics=[])
    result = evaluator.evaluate_events(gt, pred)
    return result.frame_results[0]


def test_plot_frame_overlay_basic():
    fr = _make_frame_result()
    fig = plot_frame_overlay(fr)
    assert fig is not None


def test_plot_frame_overlay_save(tmp_path):
    fr = _make_frame_result()
    out = tmp_path / "overlay.png"
    plot_frame_overlay(fr, save_path=out)
    assert out.exists()


def test_plot_frame_overlay_with_image():
    import numpy as np
    fr = _make_frame_result()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    fig = plot_frame_overlay(fr, image=image)
    assert fig is not None


def test_plot_frame_overlay_custom_title():
    fr = _make_frame_result()
    fig = plot_frame_overlay(fr, title="Custom Title")
    assert fig is not None
