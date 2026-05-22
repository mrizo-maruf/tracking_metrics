from __future__ import annotations

from dataclasses import dataclass, field

from tracking_metrics.data.detection import Detection


@dataclass
class Match:
    frame_id: int
    gt: Detection
    pred: Detection
    similarity: float


@dataclass
class IDSwitchEvent:
    frame_id: int
    gt_track_id: str
    previous_pred_track_id: str
    current_pred_track_id: str


@dataclass
class FrameResult:
    frame_id: int
    matches: list[Match] = field(default_factory=list)
    false_positives: list[Detection] = field(default_factory=list)
    false_negatives: list[Detection] = field(default_factory=list)
    id_switches: list[IDSwitchEvent] = field(default_factory=list)
