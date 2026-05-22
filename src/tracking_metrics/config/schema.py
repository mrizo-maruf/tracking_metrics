from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatcherConfig:
    type: str = "box2d-iou"
    threshold: float = 0.5
    max_distance: float = 0.5
    class_aware: bool = False


@dataclass
class OutputConfig:
    json: str | None = None
    csv: str | None = None
    latex: str | None = None


@dataclass
class EvalConfig:
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    metrics: list[str] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    hota_curves: bool = False
