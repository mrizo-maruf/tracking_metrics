from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

from tracking_metrics.data.sequence import Sequence

_BINARY_JSON = {
    "sequence_name": "mask_test",
    "frames": [
        {
            "frame_id": 0,
            "detections": [
                {
                    "track_id": 1,
                    "mask": {
                        "type": "binary",
                        "size": [4, 4],
                        "data": [
                            [0, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 0],
                        ],
                    },
                }
            ],
        }
    ],
}


def _write(data: dict) -> str:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(data, f)
        return f.name


def test_load_binary_mask():
    path = _write(_BINARY_JSON)
    seq = Sequence.from_json(path)
    det = seq.frames[0].detections[0]
    assert det.mask is not None
    binary = det.mask.to_binary()
    assert binary.shape == (4, 4)
    assert binary[1, 1] is np.bool_(True)
    assert binary[0, 0] is np.bool_(False)


def test_binary_mask_area():
    path = _write(_BINARY_JSON)
    seq = Sequence.from_json(path)
    det = seq.frames[0].detections[0]
    assert det.mask is not None
    assert det.mask.area() == 4


def test_binary_mask_size_field():
    path = _write(_BINARY_JSON)
    seq = Sequence.from_json(path)
    det = seq.frames[0].detections[0]
    assert det.mask is not None
    assert det.mask.size == (4, 4)


def test_round_trip_binary_mask():
    path = _write(_BINARY_JSON)
    seq = Sequence.from_json(path)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
        out_name = out.name
    seq.to_json(out_name)

    seq2 = Sequence.from_json(out_name)
    m1 = seq.frames[0].detections[0].mask  # noqa: SIM115 (false positive on non-file variable)
    m2 = seq2.frames[0].detections[0].mask
    assert m1 is not None and m2 is not None
    np.testing.assert_array_equal(m1.to_binary(), m2.to_binary())


def test_unknown_mask_type_raises():
    bad = {
        "sequence_name": "x",
        "frames": [
            {
                "frame_id": 0,
                "detections": [
                    {"track_id": 1, "mask": {"type": "polygon", "data": []}}
                ],
            }
        ],
    }
    path = _write(bad)
    with pytest.raises(ValueError, match="Unknown mask type"):
        Sequence.from_json(path)


def test_detection_without_mask_still_loads():
    data = {
        "sequence_name": "x",
        "frames": [
            {
                "frame_id": 0,
                "detections": [{"track_id": 1, "bbox2d": [0, 0, 10, 10]}],
            }
        ],
    }
    path = _write(data)
    seq = Sequence.from_json(path)
    assert seq.frames[0].detections[0].mask is None


def test_detection_with_both_box_and_mask():
    data = {
        "sequence_name": "x",
        "frames": [
            {
                "frame_id": 0,
                "detections": [
                    {
                        "track_id": 1,
                        "bbox2d": [0, 0, 4, 4],
                        "mask": {
                            "type": "binary",
                            "size": [4, 4],
                            "data": [[1, 1, 0, 0]] * 4,
                        },
                    }
                ],
            }
        ],
    }
    path = _write(data)
    seq = Sequence.from_json(path)
    det = seq.frames[0].detections[0]
    assert det.bbox2d is not None
    assert det.mask is not None
