import pytest

from tests.helpers import det, run_eval, seq


def test_perfect_tracking():
    gt = seq("s", (0, [det("1", 0)]), (1, [det("1", 1)]))
    pred = seq("s", (0, [det("10", 0)]), (1, [det("10", 1)]))
    _, scores = run_eval(gt, pred)
    assert scores["TP"] == 2
    assert scores["FP"] == 0
    assert scores["FN"] == 0
    assert scores["IDSW"] == 0
    assert scores["MOTA"] == pytest.approx(1.0)


def test_id_switch_mota():
    gt = seq("s", (0, [det("1", 0)]), (1, [det("1", 1)]))
    pred = seq("s", (0, [det("10", 0)]), (1, [det("20", 1)]))
    _, scores = run_eval(gt, pred)
    assert scores["IDSW"] == 1
    # GT=2, FN=0, FP=0, IDSW=1 -> MOTA = 1 - 1/2 = 0.5
    assert scores["MOTA"] == pytest.approx(0.5)


def test_false_positive():
    gt = seq("s", (0, []))
    pred = seq("s", (0, [det("10", 0)]))
    _, scores = run_eval(gt, pred)
    assert scores["FP"] == 1
    assert scores["FN"] == 0
    assert scores["TP"] == 0


def test_false_negative():
    gt = seq("s", (0, [det("1", 0)]))
    pred = seq("s", (0, []))
    _, scores = run_eval(gt, pred)
    assert scores["FN"] == 1
    assert scores["FP"] == 0
    assert scores["TP"] == 0


def test_iou_threshold_rejection():
    from tracking_metrics.data.boxes import Box2D
    gt = seq("s", (0, [det("1", 0, box=Box2D(0, 0, 100, 100))]))
    pred = seq("s", (0, [det("10", 0, box=Box2D(90, 90, 190, 190))]))
    _, scores = run_eval(gt, pred)
    assert scores["TP"] == 0
    assert scores["FP"] == 1
    assert scores["FN"] == 1


def test_no_gt_mota_is_zero():
    gt = seq("s", (0, []))
    pred = seq("s", (0, []))
    _, scores = run_eval(gt, pred)
    assert scores["MOTA"] == pytest.approx(0.0)
