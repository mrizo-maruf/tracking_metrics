import pytest

from tests.helpers import det, run_eval, seq
from tracking_metrics.data.boxes import Box2D


def test_perfect_match_motp():
    gt = seq("s", (0, [det("1", 0)]))
    pred = seq("s", (0, [det("10", 0)]))
    _, scores = run_eval(gt, pred)
    assert scores["MOTP"] == pytest.approx(1.0)


def test_no_matches_motp_zero():
    gt = seq("s", (0, []))
    pred = seq("s", (0, []))
    _, scores = run_eval(gt, pred)
    assert scores["MOTP"] == pytest.approx(0.0)


def test_partial_overlap_motp():
    a = Box2D(0, 0, 10, 10)
    b = Box2D(5, 5, 15, 15)
    gt = seq("s", (0, [det("1", 0, box=a)]))
    pred = seq("s", (0, [det("10", 0, box=b)]))
    _, scores = run_eval(gt, pred)
    # IoU is below 0.5 threshold, so no match -> MOTP=0.0
    assert scores["MOTP"] == pytest.approx(0.0)


def test_motp_is_average_over_matches():
    a = Box2D(0, 0, 100, 100)
    b = Box2D(0, 0, 100, 100)
    gt = seq("s", (0, [det("1", 0, box=a)]), (1, [det("1", 1, box=a)]))
    pred = seq("s", (0, [det("10", 0, box=b)]), (1, [det("10", 1, box=b)]))
    _, scores = run_eval(gt, pred)
    assert scores["MOTP"] == pytest.approx(1.0)
    assert scores["TP"] == 2
