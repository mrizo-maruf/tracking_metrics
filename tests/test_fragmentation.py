from __future__ import annotations

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.fragmentation import FragmentationsMetric


def _box() -> Box2D:
    return Box2D(x1=0, y1=0, x2=2, y2=2)


def _det(track_id: str, frame_id: int) -> Detection:
    return Detection(frame_id=frame_id, track_id=track_id, bbox2d=_box())


def _seq(*frames_data: tuple) -> Sequence:
    return Sequence(
        name="s",
        frames=[Frame(frame_id=fid, detections=dets) for fid, dets in frames_data],
    )


def _run(gt: Sequence, pred: Sequence) -> dict:
    evaluator = TrackingEvaluator(
        matcher=Box2DIoUMatcher(threshold=0.5),
        metrics=[FragmentationsMetric()],
    )
    return evaluator.evaluate(gt, pred)


def test_no_fragmentation_continuous_track():
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]), (2, [_det("1", 2)]))
    pred = _seq((0, [_det("A", 0)]), (1, [_det("A", 1)]), (2, [_det("A", 2)]))
    scores = _run(gt, pred)
    assert scores["Frag"] == 0


def test_one_fragmentation_gap_in_middle():
    # GT track appears in frames 0, 1, 2 but pred only covers 0 and 2 (gap at 1)
    gt = _seq(
        (0, [_det("1", 0)]),
        (1, [_det("1", 1)]),
        (2, [_det("1", 2)]),
    )
    pred = _seq(
        (0, [_det("A", 0)]),
        (1, []),          # no prediction → frame 1 becomes FN
        (2, [_det("A", 2)]),
    )
    scores = _run(gt, pred)
    assert scores["Frag"] == 1


def test_two_fragmentations():
    gt = _seq(
        (0, [_det("1", 0)]),
        (1, [_det("1", 1)]),
        (2, [_det("1", 2)]),
        (3, [_det("1", 3)]),
        (4, [_det("1", 4)]),
    )
    pred = _seq(
        (0, [_det("A", 0)]),
        (1, []),
        (2, [_det("A", 2)]),
        (3, []),
        (4, [_det("A", 4)]),
    )
    scores = _run(gt, pred)
    assert scores["Frag"] == 2


def test_fully_missed_track_zero_frags():
    gt = _seq((0, [_det("1", 0)]), (1, [_det("1", 1)]))
    pred = _seq((0, []), (1, []))
    scores = _run(gt, pred)
    # Never matched → 0 segments → Frag = 0
    assert scores["Frag"] == 0


def test_two_tracks_independent_frags():
    gt = _seq(
        (0, [_det("1", 0), _det("2", 0)]),
        (1, [_det("1", 1), _det("2", 1)]),
        (2, [_det("1", 2), _det("2", 2)]),
    )
    # Track 1 pred has gap at frame 1; track 2 pred is continuous
    pred = _seq(
        (0, [_det("A", 0), _det("B", 0)]),
        (1, [_det("B", 1)]),    # A missing → track 1 fragments
        (2, [_det("A", 2), _det("B", 2)]),
    )
    scores = _run(gt, pred)
    assert scores["Frag"] == 1
