from __future__ import annotations

import pytest

from tracking_metrics.data.boxes3d import Box3D
from tracking_metrics.data.detection import Detection
from tracking_metrics.matching.center_distance_matcher import CenterDistanceMatcher


def _det(track_id: str, center: tuple, class_id: str | None = None) -> Detection:
    return Detection(
        frame_id=0,
        track_id=track_id,
        bbox3d=Box3D(center=center, size=(1, 1, 1)),
        class_id=class_id,
    )


def test_within_max_distance_matches():
    gt = [_det("1", (0, 0, 0))]
    pred = [_det("10", (0.3, 0, 0))]
    result = CenterDistanceMatcher(max_distance=0.5).match(gt, pred)
    assert result.matches == [(0, 0)]


def test_exactly_at_max_distance_matches():
    gt = [_det("1", (0, 0, 0))]
    pred = [_det("10", (0.5, 0, 0))]
    result = CenterDistanceMatcher(max_distance=0.5).match(gt, pred)
    assert result.matches == [(0, 0)]


def test_above_max_distance_no_match():
    gt = [_det("1", (0, 0, 0))]
    pred = [_det("10", (1.0, 0, 0))]
    result = CenterDistanceMatcher(max_distance=0.5).match(gt, pred)
    assert result.matches == []
    assert 0 in result.unmatched_gt
    assert 0 in result.unmatched_pred


def test_no_bbox3d_no_match():
    gt = [Detection(frame_id=0, track_id="1")]
    pred = [Detection(frame_id=0, track_id="10")]
    result = CenterDistanceMatcher(max_distance=0.5).match(gt, pred)
    assert result.matches == []


def test_similarity_matrix_populated():
    gt = [_det("1", (0, 0, 0))]
    pred = [_det("10", (0.25, 0, 0))]
    result = CenterDistanceMatcher(max_distance=0.5).match(gt, pred)
    # distance=0.25, max=0.5 => similarity = 1 - 0.25/0.5 = 0.5
    assert result.similarity_matrix[0, 0] == pytest.approx(0.5)


def test_distance_matrix_populated():
    gt = [_det("1", (0, 0, 0))]
    pred = [_det("10", (3, 4, 0))]
    result = CenterDistanceMatcher(max_distance=10.0).match(gt, pred)
    assert result.distance_matrix is not None
    assert result.distance_matrix[0, 0] == pytest.approx(5.0)


def test_class_aware_different_class_no_match():
    gt = [_det("1", (0, 0, 0), class_id="car")]
    pred = [_det("10", (0, 0, 0), class_id="person")]
    result = CenterDistanceMatcher(max_distance=1.0, class_aware=True).match(gt, pred)
    assert result.matches == []


def test_class_aware_null_class_matches():
    gt = [_det("1", (0, 0, 0), class_id=None)]
    pred = [_det("10", (0, 0, 0), class_id="car")]
    result = CenterDistanceMatcher(max_distance=1.0, class_aware=True).match(gt, pred)
    assert result.matches == [(0, 0)]


def test_picks_closest_pred():
    gt = [_det("1", (0, 0, 0))]
    pred = [_det("10", (0.1, 0, 0)), _det("20", (0.4, 0, 0))]
    result = CenterDistanceMatcher(max_distance=1.0).match(gt, pred)
    assert len(result.matches) == 1
    assert result.matches[0] == (0, 0)  # pred index 0 is closer


def test_empty_inputs():
    result = CenterDistanceMatcher().match([], [])
    assert result.matches == []
    assert result.unmatched_gt == []
    assert result.unmatched_pred == []
