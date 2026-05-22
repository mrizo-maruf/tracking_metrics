from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Top-level configuration for evaluation runs."""

    sequence_name: str = ""
