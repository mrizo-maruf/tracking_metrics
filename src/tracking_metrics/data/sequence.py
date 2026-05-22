from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame


@dataclass
class Sequence:
    name: str
    frames: list[Frame] = field(default_factory=list)

    def frame_ids(self) -> list[int]:
        return [f.frame_id for f in self.frames]

    def get_frame(self, frame_id: int) -> Frame:
        for f in self.frames:
            if f.frame_id == frame_id:
                return f
        return Frame(frame_id=frame_id, detections=[])

    def all_detections(self) -> list[Detection]:
        return [d for f in self.frames for d in f.detections]

    @classmethod
    def from_json(cls, path: str | Path) -> Sequence:
        with open(path) as fh:
            data = json.load(fh)
        return _sequence_from_dict(data)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            json.dump(_sequence_to_dict(self), fh, indent=2)


def _sequence_from_dict(data: dict) -> Sequence:  # type: ignore[type-arg]
    frames = []
    for fd in data.get("frames", []):
        detections = []
        for dd in fd.get("detections", []):
            bbox2d = None
            if "bbox2d" in dd and dd["bbox2d"] is not None:
                coords = dd["bbox2d"]
                bbox2d = Box2D(
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                )
            det = Detection(
                frame_id=int(fd["frame_id"]),
                track_id=str(dd["track_id"]),
                class_id=dd.get("class_id"),
                score=dd.get("score"),
                bbox2d=bbox2d,
                attributes={
                    k: v
                    for k, v in dd.items()
                    if k not in {"track_id", "class_id", "score", "bbox2d"}
                },
            )
            detections.append(det)
        frames.append(Frame(frame_id=int(fd["frame_id"]), detections=detections))
    return Sequence(name=str(data.get("sequence_name", "")), frames=frames)


def _sequence_to_dict(seq: Sequence) -> dict:  # type: ignore[type-arg]
    frames = []
    for frame in seq.frames:
        detections = []
        for det in frame.detections:
            dd: dict = {  # type: ignore[type-arg]
                "track_id": det.track_id,
            }
            if det.class_id is not None:
                dd["class_id"] = det.class_id
            if det.score is not None:
                dd["score"] = det.score
            if det.bbox2d is not None:
                b = det.bbox2d
                dd["bbox2d"] = [b.x1, b.y1, b.x2, b.y2]
            dd.update(det.attributes)
            detections.append(dd)
        frames.append({"frame_id": frame.frame_id, "detections": detections})
    return {"sequence_name": seq.name, "frames": frames}
