from __future__ import annotations

import pytest

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.hota import HOTAMetric


def _box(x: float = 0.0) -> Box2D:
    return Box2D(x1=x, y1=0, x2=x + 2, y2=2)


def _det(track_id: str, frame_id: int, x: float = 0.0) -> Detection:
    return Detection(frame_id=frame_id, track_id=track_id, bbox2d=_box(x))


def _seq(*frames_data: tuple) -> Sequence:
    return Sequence(
        name="s",
        frames=[Frame(frame_id=fid, detections=dets) for fid, dets in frames_data],
    )


def _run_hota(gt: Sequence, pred: Sequence, **kwargs) -> dict:
    evaluator = TrackingEvaluator(
        matcher=Box2DIoUMatcher(threshold=0.5),
        metrics=[HOTAMetric(**kwargs)],
    )
    return evaluator.evaluate(gt, pred)


# --- Perfect tracking ---

def test_perfect_tracking_hota_is_one():
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]))
    pred = _seq((0, [_det("10", 0)]), (1, [_det("10", 1)]))
    scores = _run_hota(gt, pred)
    assert scores["HOTA"] == pytest.approx(1.0)
    assert scores["DetA"] == pytest.approx(1.0)
    assert scores["AssA"] == pytest.approx(1.0)
    assert scores["DetRe"] == pytest.approx(1.0, abs=1e-3)
    assert scores["DetPr"] == pytest.approx(1.0, abs=1e-3)


def test_perfect_tracking_loc_a_is_one():
    gt = _seq((0, [_det("1", 0)]))
    pred = _seq((0, [_det("10", 0)]))
    scores = _run_hota(gt, pred)
    assert scores["LocA"] == pytest.approx(1.0, abs=1e-3)


# --- No predictions ---

def test_no_pred_hota_zero():
    gt = _seq((0, [_det("1", 0)]))
    pred = _seq((0, []))
    scores = _run_hota(gt, pred)
    assert scores["HOTA"] == pytest.approx(0.0, abs=1e-6)
    assert scores["DetA"] == pytest.approx(0.0, abs=1e-6)


# --- False positives ---

def test_fp_reduces_deta():
    gt = _seq((0, [_det("1", 0)]))
    pred = _seq((0, [_det("10", 0), _det("20", 0, x=10.0)]))
    scores = _run_hota(gt, pred)
    # TP=1, FP=1, FN=0 → DetPr = 0.5, DetRe = 1.0 → DetA = sqrt(0.5)
    assert scores["DetPr"] < 1.0
    assert scores["HOTA"] < 1.0


# --- ID switch lowers AssA ---

def test_id_switch_lowers_assa():
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]))
    # pred uses two different IDs → ID switch
    pred_switch = _seq((0, [_det("10", 0)]), (1, [_det("20", 1)]))
    pred_stable = _seq((0, [_det("10", 0)]), (1, [_det("10", 1)]))

    scores_switch = _run_hota(gt, pred_switch)
    scores_stable = _run_hota(gt, pred_stable)

    assert scores_switch["AssA"] < scores_stable["AssA"]


# --- Multi-track ---

def test_two_perfect_tracks():
    gt = _seq(
        (0, [_det("1", 0, x=0.0), _det("2", 0, x=10.0)]),
        (1, [_det("1", 1, x=0.0), _det("2", 1, x=10.0)]),
    )
    pred = _seq(
        (0, [_det("A", 0, x=0.0), _det("B", 0, x=10.0)]),
        (1, [_det("A", 1, x=0.0), _det("B", 1, x=10.0)]),
    )
    scores = _run_hota(gt, pred)
    assert scores["HOTA"] == pytest.approx(1.0)


# --- HOTA curves ---

def test_hota_curves_returned_when_requested():
    gt = _seq((0, [_det("1", 0)]))
    pred = _seq((0, [_det("10", 0)]))
    scores = _run_hota(gt, pred, return_curves=True)
    assert "HOTA_curve" in scores
    assert len(scores["HOTA_curve"]) == 19  # 0.05..0.95 step 0.05


def test_hota_curves_not_returned_by_default():
    gt = _seq((0, [_det("1", 0)]))
    pred = _seq((0, [_det("10", 0)]))
    scores = _run_hota(gt, pred)
    assert "HOTA_curve" not in scores


# --- Empty sequence ---

def test_empty_sequence_hota_zero():
    gt = _seq((0, []))
    pred = _seq((0, []))
    scores = _run_hota(gt, pred)
    assert scores["HOTA"] == pytest.approx(0.0, abs=1e-6)


# --- OWTA ---

def test_owta_between_0_and_1():
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]))
    pred = _seq((0, [_det("10", 0)]), (1, [_det("10", 1)]))
    scores = _run_hota(gt, pred)
    assert 0.0 <= scores["OWTA"] <= 1.0
