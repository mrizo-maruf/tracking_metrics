"""Shared fixtures for tracking_metrics tests."""

from __future__ import annotations

from tracking_metrics.data.boxes import Box2D
from tracking_metrics.data.detection import Detection
from tracking_metrics.data.frame import Frame
from tracking_metrics.data.sequence import Sequence


def make_box(x1: float, y1: float, x2: float, y2: float) -> Box2D:
    return Box2D(x1=x1, y1=y1, x2=x2, y2=y2)


def make_det(
    frame_id: int,
    track_id: str,
    box: Box2D | None = None,
    class_id: str | None = None,
) -> Detection:
    return Detection(frame_id=frame_id, track_id=track_id, bbox2d=box, class_id=class_id)


def make_sequence(name: str, frames_data: list[tuple[int, list[Detection]]]) -> Sequence:
    frames = [Frame(frame_id=fid, detections=dets) for fid, dets in frames_data]
    return Sequence(name=name, frames=frames)
