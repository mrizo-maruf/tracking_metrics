"""Verify the public top-level API is importable and works end-to-end."""

from __future__ import annotations

import pytest

import tracking_metrics as tm
from tracking_metrics import (
    HOTA,
    IDF1,
    MOTA,
    MOTP,
    Box2D,
    Box2DIoUMatcher,
    Box3D,
    Box3DIoUMatcher,
    CenterDistanceMatcher,
    Detection,
    DetectionCounts,
    Fragmentations,
    Frame,
    IDConsistency,
    IDSwitches,
    Mask2D,
    MaskIoUMatcher,
    MeanBox3DIoU,
    MeanCenterDistance3D,
    Sequence,
    TemporalIoU,
    TrackCoverage,
    TrackingEvaluator,
    TrackSurvivalRate,
)


def test_version_string():
    assert tm.__version__ == "0.5.0"


def test_data_types_importable():
    assert Box2D is not None
    assert Box3D is not None
    assert Detection is not None
    assert Frame is not None
    assert Mask2D is not None
    assert Sequence is not None


def test_matchers_importable():
    assert Box2DIoUMatcher is not None
    assert Box3DIoUMatcher is not None
    assert CenterDistanceMatcher is not None
    assert MaskIoUMatcher is not None


def test_metrics_importable():
    for cls in [
        DetectionCounts, Fragmentations, HOTA, IDConsistency,
        IDSwitches, IDF1, MeanBox3DIoU, MeanCenterDistance3D,
        MOTA, MOTP, TemporalIoU, TrackCoverage, TrackSurvivalRate,
    ]:
        assert cls is not None


def test_end_to_end_with_short_names():
    gt = Sequence(
        name="s",
        frames=[
            Frame(frame_id=0, detections=[
                Detection(frame_id=0, track_id="1", bbox2d=Box2D(0, 0, 2, 2)),
            ]),
        ],
    )
    pred = Sequence(
        name="s",
        frames=[
            Frame(frame_id=0, detections=[
                Detection(frame_id=0, track_id="A", bbox2d=Box2D(0, 0, 2, 2)),
            ]),
        ],
    )
    evaluator = TrackingEvaluator(
        matcher=Box2DIoUMatcher(threshold=0.5),
        metrics=[DetectionCounts(), MOTA(), IDSwitches()],
    )
    results = evaluator.evaluate(gt, pred)
    assert results["TP"] == 1
    assert results["MOTA"] == pytest.approx(1.0)
