import numpy as np

from tracking_metrics.matching.hungarian import hungarian_match_from_similarity


def test_perfect_match():
    sim = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = hungarian_match_from_similarity(sim, threshold=0.5)
    assert sorted(result.matches) == [(0, 0), (1, 1)]
    assert result.unmatched_gt == []
    assert result.unmatched_pred == []


def test_below_threshold_rejected():
    sim = np.array([[0.3]])
    result = hungarian_match_from_similarity(sim, threshold=0.5)
    assert result.matches == []
    assert result.unmatched_gt == [0]
    assert result.unmatched_pred == [0]


def test_at_threshold_accepted():
    sim = np.array([[0.5]])
    result = hungarian_match_from_similarity(sim, threshold=0.5)
    assert result.matches == [(0, 0)]


def test_partial_match():
    sim = np.array([[0.8, 0.1], [0.1, 0.2]])
    result = hungarian_match_from_similarity(sim, threshold=0.5)
    assert (0, 0) in result.matches
    assert 1 in result.unmatched_gt


def test_empty_inputs():
    result = hungarian_match_from_similarity(np.zeros((0, 0)), threshold=0.5)
    assert result.matches == []
    assert result.unmatched_gt == []
    assert result.unmatched_pred == []


def test_more_gt_than_pred():
    sim = np.array([[0.9], [0.8]])
    result = hungarian_match_from_similarity(sim, threshold=0.5)
    assert len(result.matches) == 1
    assert len(result.unmatched_gt) == 1


def test_more_pred_than_gt():
    sim = np.array([[0.9, 0.8]])
    result = hungarian_match_from_similarity(sim, threshold=0.5)
    assert len(result.matches) == 1
    assert len(result.unmatched_pred) == 1
