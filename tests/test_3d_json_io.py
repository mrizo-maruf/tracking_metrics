from __future__ import annotations

import json
import tempfile

import pytest

from tracking_metrics.data.sequence import Sequence

_3D_JSON = {
    "sequence_name": "3d_test",
    "frames": [
        {
            "frame_id": 0,
            "detections": [
                {
                    "track_id": 1,
                    "class_id": "chair",
                    "bbox3d": {
                        "center": [1.2, 0.4, 3.1],
                        "size": [0.7, 0.9, 0.8],
                        "yaw": 1.57,
                    },
                },
                {
                    "track_id": 2,
                    "bbox3d": {
                        "center": [5.0, 0.0, 1.0],
                        "size": [1.0, 1.0, 1.0],
                    },
                },
            ],
        }
    ],
}


def _write(data: dict) -> str:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(data, f)
        return f.name


def test_load_bbox3d_center():
    seq = Sequence.from_json(_write(_3D_JSON))
    det = seq.frames[0].detections[0]
    assert det.bbox3d is not None
    assert det.bbox3d.center == pytest.approx((1.2, 0.4, 3.1))


def test_load_bbox3d_size():
    seq = Sequence.from_json(_write(_3D_JSON))
    det = seq.frames[0].detections[0]
    assert det.bbox3d is not None
    assert det.bbox3d.size == pytest.approx((0.7, 0.9, 0.8))


def test_load_bbox3d_yaw():
    seq = Sequence.from_json(_write(_3D_JSON))
    det = seq.frames[0].detections[0]
    assert det.bbox3d is not None
    assert det.bbox3d.yaw == pytest.approx(1.57)


def test_load_bbox3d_no_yaw():
    seq = Sequence.from_json(_write(_3D_JSON))
    det = seq.frames[0].detections[1]
    assert det.bbox3d is not None
    assert det.bbox3d.yaw is None


def test_load_track_id_as_str():
    seq = Sequence.from_json(_write(_3D_JSON))
    assert seq.frames[0].detections[0].track_id == "1"


def test_round_trip_3d():
    seq = Sequence.from_json(_write(_3D_JSON))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
        out_name = out.name
    seq.to_json(out_name)
    seq2 = Sequence.from_json(out_name)
    b1 = seq.frames[0].detections[0].bbox3d
    b2 = seq2.frames[0].detections[0].bbox3d
    assert b1 is not None and b2 is not None
    assert b1.center == pytest.approx(b2.center)
    assert b1.size == pytest.approx(b2.size)
    assert b1.yaw == pytest.approx(b2.yaw)


def test_detection_without_bbox3d_still_loads():
    data = {
        "sequence_name": "x",
        "frames": [{"frame_id": 0, "detections": [{"track_id": 1, "bbox2d": [0, 0, 10, 10]}]}],
    }
    seq = Sequence.from_json(_write(data))
    assert seq.frames[0].detections[0].bbox3d is None


def test_detection_with_bbox2d_and_bbox3d():
    data = {
        "sequence_name": "x",
        "frames": [
            {
                "frame_id": 0,
                "detections": [
                    {
                        "track_id": 1,
                        "bbox2d": [0, 0, 10, 10],
                        "bbox3d": {"center": [0, 0, 0], "size": [1, 1, 1]},
                    }
                ],
            }
        ],
    }
    seq = Sequence.from_json(_write(data))
    det = seq.frames[0].detections[0]
    assert det.bbox2d is not None
    assert det.bbox3d is not None
