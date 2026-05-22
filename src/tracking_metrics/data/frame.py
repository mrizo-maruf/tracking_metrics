from __future__ import annotations

from dataclasses import dataclass, field

from tracking_metrics.data.detection import Detection


@dataclass
class Frame:
    frame_id: int
    detections: list[Detection] = field(default_factory=list)
