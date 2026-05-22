
from tracking_metrics.data.detection import Detection
from tracking_metrics.evaluation.events import Match
from tracking_metrics.evaluation.identity_history import IdentityHistory


def _match(gt_id: str, pred_id: str, frame_id: int = 0) -> Match:
    gt = Detection(frame_id=frame_id, track_id=gt_id)
    pred = Detection(frame_id=frame_id, track_id=pred_id)
    return Match(frame_id=frame_id, gt=gt, pred=pred, similarity=1.0)


def test_no_switch_first_appearance():
    history = IdentityHistory()
    events = history.update(0, [_match("1", "10")])
    assert events == []


def test_no_switch_same_pred():
    history = IdentityHistory()
    history.update(0, [_match("1", "10")])
    events = history.update(1, [_match("1", "10")])
    assert events == []


def test_switch_detected():
    history = IdentityHistory()
    history.update(0, [_match("1", "10")])
    events = history.update(1, [_match("1", "20")])
    assert len(events) == 1
    e = events[0]
    assert e.gt_track_id == "1"
    assert e.previous_pred_track_id == "10"
    assert e.current_pred_track_id == "20"
    assert e.frame_id == 1


def test_multiple_switches():
    history = IdentityHistory()
    history.update(0, [_match("1", "10"), _match("2", "20")])
    events = history.update(1, [_match("1", "30"), _match("2", "40")])
    assert len(events) == 2


def test_no_switch_when_not_matched():
    history = IdentityHistory()
    history.update(0, [_match("1", "10")])
    events = history.update(1, [])
    assert events == []
    # After reappearing with a new pred, it should switch
    events2 = history.update(2, [_match("1", "20")])
    assert len(events2) == 1
