from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tracking_metrics.data.boxes import Box2D

if TYPE_CHECKING:
    from tracking_metrics.data.boxes3d import Box3D
    from tracking_metrics.data.masks import Mask2D


@dataclass
class Detection:
    frame_id: int
    track_id: str
    class_id: str | None = None
    score: float | None = None
    bbox2d: Box2D | None = None
    mask: Mask2D | None = None
    bbox3d: Box3D | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.track_id = str(self.track_id)
