import json
import tempfile

import pytest

from tracking_metrics.data.sequence import Sequence

SAMPLE_JSON = {
    "sequence_name": "scene_001",
    "frames": [
        {
            "frame_id": 0,
            "detections": [
                {
                    "track_id": 1,
                    "class_id": "chair",
                    "score": 0.98,
                    "bbox2d": [100, 120, 220, 260],
                }
            ],
        },
        {
            "frame_id": 1,
            "detections": [
                {
                    "track_id": "2",
                    "bbox2d": [10, 20, 50, 60],
                }
            ],
        },
    ],
}


def test_from_json_parses_name():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(SAMPLE_JSON, f)
        path = f.name
    seq = Sequence.from_json(path)
    assert seq.name == "scene_001"


def test_from_json_track_id_is_str():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(SAMPLE_JSON, f)
        path = f.name
    seq = Sequence.from_json(path)
    assert seq.frames[0].detections[0].track_id == "1"
    assert seq.frames[1].detections[0].track_id == "2"


def test_from_json_bbox_parsed():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(SAMPLE_JSON, f)
        path = f.name
    seq = Sequence.from_json(path)
    b = seq.frames[0].detections[0].bbox2d
    assert b is not None
    assert b.x1 == 100.0
    assert b.y2 == 260.0


def test_from_json_score_and_class():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(SAMPLE_JSON, f)
        path = f.name
    seq = Sequence.from_json(path)
    det = seq.frames[0].detections[0]
    assert det.score == pytest.approx(0.98)
    assert det.class_id == "chair"


def test_round_trip():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(SAMPLE_JSON, f)
        in_path = f.name
    seq = Sequence.from_json(in_path)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f2:
        out_path = f2.name
    seq.to_json(out_path)
    seq2 = Sequence.from_json(out_path)
    assert seq2.name == seq.name
    assert len(seq2.frames) == len(seq.frames)
    b1 = seq.frames[0].detections[0].bbox2d
    b2 = seq2.frames[0].detections[0].bbox2d
    assert b1 == b2


def test_get_frame_missing_returns_empty():
    s = Sequence(name="x", frames=[])
    f = s.get_frame(99)
    assert f.frame_id == 99
    assert f.detections == []


def test_all_detections():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(SAMPLE_JSON, f)
        path = f.name
    seq = Sequence.from_json(path)
    all_dets = seq.all_detections()
    assert len(all_dets) == 2
