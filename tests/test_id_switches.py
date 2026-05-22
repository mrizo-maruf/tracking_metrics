
from tests.helpers import det, run_eval, seq


def test_no_id_switch_perfect():
    gt = seq("s", (0, [det("1", 0)]), (1, [det("1", 1)]))
    pred = seq("s", (0, [det("10", 0)]), (1, [det("10", 1)]))
    _, scores = run_eval(gt, pred)
    assert scores["IDSW"] == 0


def test_one_id_switch():
    gt = seq("s", (0, [det("1", 0)]), (1, [det("1", 1)]))
    pred = seq("s", (0, [det("10", 0)]), (1, [det("20", 1)]))
    _, scores = run_eval(gt, pred)
    assert scores["IDSW"] == 1


def test_two_id_switches():
    gt = seq(
        "s",
        (0, [det("1", 0)]),
        (1, [det("1", 1)]),
        (2, [det("1", 2)]),
    )
    pred = seq(
        "s",
        (0, [det("10", 0)]),
        (1, [det("20", 1)]),
        (2, [det("30", 2)]),
    )
    _, scores = run_eval(gt, pred)
    assert scores["IDSW"] == 2


def test_first_appearance_not_a_switch():
    gt = seq("s", (0, [det("1", 0)]))
    pred = seq("s", (0, [det("10", 0)]))
    _, scores = run_eval(gt, pred)
    assert scores["IDSW"] == 0
