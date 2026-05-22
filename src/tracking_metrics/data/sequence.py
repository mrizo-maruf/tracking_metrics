from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.masks import Mask2D


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


def _parse_mask(mask_dict: dict[str, Any]) -> Mask2D:
    mask_type = mask_dict.get("type")
    size_raw = mask_dict.get("size")
    size: tuple[int, int] | None = tuple(size_raw) if size_raw is not None else None  # type: ignore[assignment]

    if mask_type == "binary":
        data = np.array(mask_dict["data"], dtype=bool)
        return Mask2D(data=data, size=size)

    if mask_type == "rle":
        rle = {"size": size_raw, "counts": mask_dict["counts"]}
        return Mask2D(rle=rle, size=size)

    raise ValueError(f"Unknown mask type: {mask_type!r}. Supported: 'binary', 'rle'.")


def _serialize_mask(mask: Mask2D) -> dict[str, Any]:
    if mask.data is not None:
        h, w = mask.data.shape
        return {
            "type": "binary",
            "size": [h, w],
            "data": mask.data.astype(int).tolist(),
        }
    if mask.rle is not None:
        rle = mask.rle
        return {
            "type": "rle",
            "size": rle.get("size"),
            "counts": rle.get("counts"),
        }
    raise ValueError("Mask2D has neither data nor rle set.")


_KNOWN_DET_KEYS = {"track_id", "class_id", "score", "bbox2d", "mask"}


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

            mask = None
            if "mask" in dd and dd["mask"] is not None:
                mask = _parse_mask(dd["mask"])

            det = Detection(
                frame_id=int(fd["frame_id"]),
                track_id=str(dd["track_id"]),
                class_id=dd.get("class_id"),
                score=dd.get("score"),
                bbox2d=bbox2d,
                mask=mask,
                attributes={k: v for k, v in dd.items() if k not in _KNOWN_DET_KEYS},
            )
            detections.append(det)
        frames.append(Frame(frame_id=int(fd["frame_id"]), detections=detections))
    return Sequence(name=str(data.get("sequence_name", "")), frames=frames)


def _sequence_to_dict(seq: Sequence) -> dict:  # type: ignore[type-arg]
    frames = []
    for frame in seq.frames:
        detections = []
        for det in frame.detections:
            dd: dict[str, Any] = {"track_id": det.track_id}
            if det.class_id is not None:
                dd["class_id"] = det.class_id
            if det.score is not None:
                dd["score"] = det.score
            if det.bbox2d is not None:
                b = det.bbox2d
                dd["bbox2d"] = [b.x1, b.y1, b.x2, b.y2]
            if det.mask is not None:
                dd["mask"] = _serialize_mask(det.mask)
            dd.update(det.attributes)
            detections.append(dd)
        frames.append({"frame_id": frame.frame_id, "detections": detections})
    return {"sequence_name": seq.name, "frames": frames}
