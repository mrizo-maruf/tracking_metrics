import pytest

from tests.helpers import det, run_eval, seq


def test_perfect_tracking_idf1():
    gt = seq("s", (0, [det("1", 0)]), (1, [det("1", 1)]))
    pred = seq("s", (0, [det("10", 0)]), (1, [det("10", 1)]))
    _, scores = run_eval(gt, pred)
    assert scores["IDF1"] == pytest.approx(1.0)
    assert scores["IDP"] == pytest.approx(1.0)
    assert scores["IDR"] == pytest.approx(1.0)
    assert scores["IDTP"] == 2
    assert scores["IDFP"] == 0
    assert scores["IDFN"] == 0


def test_no_matches_idf1():
    gt = seq("s", (0, [det("1", 0)]))
    pred = seq("s", (0, []))
    _, scores = run_eval(gt, pred)
    assert scores["IDF1"] == pytest.approx(0.0)
    assert scores["IDFN"] == 1


def test_one_false_positive():
    gt = seq("s", (0, []))
    pred = seq("s", (0, [det("10", 0)]))
    _, scores = run_eval(gt, pred)
    assert scores["IDF1"] == pytest.approx(0.0)
    assert scores["IDFP"] == 1


def test_id_switch_reduces_idf1():
    # Two frames, same GT, but different pred track each frame
    gt = seq("s", (0, [det("1", 0)]), (1, [det("1", 1)]))
    pred = seq("s", (0, [det("10", 0)]), (1, [det("20", 1)]))
    _, scores = run_eval(gt, pred)
    # Hungarian picks the best single assignment (count=1 each)
    # IDTP=1, IDFP=1 (other pred track), IDFN=1 (other gt appearance)
    assert scores["IDTP"] == 1
    assert scores["IDF1"] == pytest.approx(2 * 1 / (2 * 1 + 1 + 1))
