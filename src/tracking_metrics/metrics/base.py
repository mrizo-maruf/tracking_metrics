from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from tracking_metrics.evaluation.evaluator import EvaluationResult


class Metric(Protocol):
    name: str

    def compute(self, result: EvaluationResult) -> dict[str, float | int]: ...
