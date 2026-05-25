"""Tests for visualization.timeline."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from tests.conftest import make_box, make_det, make_sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.visualization.timeline import (
    plot_frame_timeline,
    plot_track_coverage,
    plot_track_survival,
)


def _make_eval_result():
    box = make_box(0, 0, 10, 10)
    gt = make_sequence("s", [
        (0, [make_det(0, "1", box)]),
        (1, [make_det(1, "1", box), make_det(1, "2", make_box(20, 20, 30, 30))]),
        (2, [make_det(2, "1", box)]),
    ])
    pred = make_sequence("s", [
        (0, [make_det(0, "a", box)]),
        (1, [make_det(1, "a", box)]),
        (2, []),
    ])
    evaluator = TrackingEvaluator(matcher=Box2DIoUMatcher(threshold=0.3), metrics=[])
    return evaluator.evaluate_events(gt, pred)


def test_plot_frame_timeline():
    result = _make_eval_result()
    fig = plot_frame_timeline(result)
    assert fig is not None


def test_plot_frame_timeline_save(tmp_path):
    result = _make_eval_result()
    out = tmp_path / "timeline.png"
    plot_frame_timeline(result, save_path=out)
    assert out.exists()


def test_plot_track_coverage():
    result = _make_eval_result()
    fig = plot_track_coverage(result)
    assert fig is not None


def test_plot_track_coverage_empty():
    from tracking_metrics.data.sequence import Sequence
    from tracking_metrics.data.frame import Frame
    evaluator = TrackingEvaluator(matcher=Box2DIoUMatcher(threshold=0.5), metrics=[])
    gt = Sequence(name="empty", frames=[Frame(frame_id=0, detections=[])])
    pred = Sequence(name="empty", frames=[Frame(frame_id=0, detections=[])])
    result = evaluator.evaluate_events(gt, pred)
    fig = plot_track_coverage(result)
    assert fig is not None


def test_plot_track_survival():
    result = _make_eval_result()
    fig = plot_track_survival(result)
    assert fig is not None
