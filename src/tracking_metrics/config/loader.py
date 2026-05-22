from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tracking_metrics.config.schema import EvalConfig, MatcherConfig, OutputConfig


def load_config(path: str | Path) -> EvalConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}

    matcher_raw = raw.get("matcher", {})
    matcher = MatcherConfig(
        type=matcher_raw.get("type", "box2d-iou"),
        threshold=float(matcher_raw.get("threshold", 0.5)),
        max_distance=float(matcher_raw.get("max_distance", 0.5)),
        class_aware=bool(matcher_raw.get("class_aware", False)),
    )

    output_raw = raw.get("output", {})
    output = OutputConfig(
        json=output_raw.get("json"),
        csv=output_raw.get("csv"),
        latex=output_raw.get("latex"),
    )

    return EvalConfig(
        matcher=matcher,
        metrics=raw.get("metrics", []),
        output=output,
        hota_curves=bool(raw.get("hota_curves", False)),
    )
