from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tracking_metrics.data.boxes import Box2D


@dataclass
class Detection:
    frame_id: int
    track_id: str
    class_id: str | None = None
    score: float | None = None
    bbox2d: Box2D | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.track_id = str(self.track_id)
