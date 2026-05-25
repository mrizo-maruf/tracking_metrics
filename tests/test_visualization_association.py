"""Tests for visualization.association and visualization.id_switches."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from tests.conftest import make_box, make_det, make_sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.visualization.association import plot_association_heatmap
from tracking_metrics.visualization.id_switches import get_id_switch_table, plot_id_switch_timeline


def _make_eval_result():
    box_a = make_box(0, 0, 10, 10)
    box_b = make_box(50, 50, 60, 60)
    gt = make_sequence("s", [
        (0, [make_det(0, "1", box_a)]),
        (1, [make_det(1, "1", box_a)]),
        (2, [make_det(2, "1", box_a)]),
    ])
    pred = make_sequence("s", [
        (0, [make_det(0, "x", box_a)]),
        (1, [make_det(1, "y", box_b)]),  # ID switch: new pred doesn't match
        (2, [make_det(2, "x", box_a)]),
    ])
    evaluator = TrackingEvaluator(matcher=Box2DIoUMatcher(threshold=0.3), metrics=[])
    return evaluator.evaluate_events(gt, pred)


def test_plot_association_heatmap():
    result = _make_eval_result()
    fig = plot_association_heatmap(result)
    assert fig is not None


def test_plot_association_heatmap_save(tmp_path):
    result = _make_eval_result()
    out = tmp_path / "assoc.png"
    plot_association_heatmap(result, save_path=out)
    assert out.exists()


def test_plot_association_heatmap_not_normalized():
    result = _make_eval_result()
    fig = plot_association_heatmap(result, normalize=False)
    assert fig is not None


def test_plot_id_switch_timeline():
    result = _make_eval_result()
    fig = plot_id_switch_timeline(result)
    assert fig is not None


def test_plot_id_switch_timeline_no_switches():
    box = make_box(0, 0, 10, 10)
    gt = make_sequence("s", [(0, [make_det(0, "1", box)])])
    pred = make_sequence("s", [(0, [make_det(0, "a", box)])])
    evaluator = TrackingEvaluator(matcher=Box2DIoUMatcher(threshold=0.3), metrics=[])
    result = evaluator.evaluate_events(gt, pred)
    fig = plot_id_switch_timeline(result)
    assert fig is not None


def test_get_id_switch_table():
    result = _make_eval_result()
    table = get_id_switch_table(result)
    assert isinstance(table, list)
    for row in table:
        assert "frame_id" in row
        assert "gt_track_id" in row
        assert "prev_pred_id" in row
        assert "new_pred_id" in row
