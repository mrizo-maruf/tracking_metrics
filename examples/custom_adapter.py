"""Example: implement a custom SequenceAdapter for a project-specific format.

This shows how to bridge a dataset-specific format to tracking-metrics
without adding dataset dependencies to the core library.
"""

from __future__ import annotations

from typing import Any

from tracking_metrics import (
    MOTA,
    Box2D,
    Box2DIoUMatcher,
    Detection,
    DetectionCounts,
    Frame,
    Sequence,
    TrackingEvaluator,
)
from tracking_metrics.report import print_results_table

# ---------------------------------------------------------------------------
# Simulated project-specific format
# ---------------------------------------------------------------------------

_EXAMPLE_ANNOTATIONS: dict[str, Any] = {
    "meta": {"scene": "hallway_01", "fps": 30},
    "objects": [
        {"id": 101, "frames": [
            {"t": 0, "x": 10, "y": 20, "w": 50, "h": 80},
            {"t": 1, "x": 12, "y": 21, "w": 50, "h": 80},
        ]},
        {"id": 102, "frames": [
            {"t": 0, "x": 200, "y": 100, "w": 40, "h": 60},
        ]},
    ],
}

_EXAMPLE_TRACKER_OUTPUT: dict[str, Any] = {
    "scene": "hallway_01",
    "tracks": [
        {"track_id": "T1", "detections": [
            {"frame": 0, "bbox": [10, 20, 60, 100]},
            {"frame": 1, "bbox": [13, 22, 63, 102]},
        ]},
        {"track_id": "T2", "detections": [
            {"frame": 0, "bbox": [201, 101, 241, 161]},
        ]},
    ],
}


# ---------------------------------------------------------------------------
# Custom adapter
# ---------------------------------------------------------------------------

class MyProjectAdapter:
    """Converts project-specific annotation/tracker formats to Sequence."""

    def load_ground_truth(self, data: dict[str, Any]) -> Sequence:
        frames_map: dict[int, list[Detection]] = {}
        for obj in data["objects"]:
            track_id = str(obj["id"])
            for fr in obj["frames"]:
                t = fr["t"]
                x, y, w, h = fr["x"], fr["y"], fr["w"], fr["h"]
                det = Detection(
                    frame_id=t,
                    track_id=track_id,
                    bbox2d=Box2D(x1=x, y1=y, x2=x + w, y2=y + h),
                )
                frames_map.setdefault(t, []).append(det)

        return Sequence(
            name=data["meta"]["scene"],
            frames=[Frame(frame_id=t, detections=dets) for t, dets in sorted(frames_map.items())],
        )

    def load_predictions(self, data: dict[str, Any]) -> Sequence:
        frames_map: dict[int, list[Detection]] = {}
        for track in data["tracks"]:
            track_id = track["track_id"]
            for det_data in track["detections"]:
                t = det_data["frame"]
                x1, y1, x2, y2 = det_data["bbox"]
                det = Detection(
                    frame_id=t,
                    track_id=track_id,
                    bbox2d=Box2D(x1=x1, y1=y1, x2=x2, y2=y2),
                )
                frames_map.setdefault(t, []).append(det)

        return Sequence(
            name=data["scene"],
            frames=[Frame(frame_id=t, detections=dets) for t, dets in sorted(frames_map.items())],
        )


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

adapter = MyProjectAdapter()
gt = adapter.load_ground_truth(_EXAMPLE_ANNOTATIONS)
pred = adapter.load_predictions(_EXAMPLE_TRACKER_OUTPUT)

evaluator = TrackingEvaluator(
    matcher=Box2DIoUMatcher(threshold=0.5),
    metrics=[DetectionCounts(), MOTA()],
)
results = evaluator.evaluate(gt, pred)
print_results_table(results)

# Optionally cache the converted sequence in the standard JSON format
# so future runs don't need the adapter:
# save_sequence_json(gt, "gt_hallway_01.json")
# save_sequence_json(pred, "pred_hallway_01.json")
