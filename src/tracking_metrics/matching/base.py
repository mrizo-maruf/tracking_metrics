from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tracking_metrics.data.detection import Detection


@dataclass
class MatchResult:
    matches: list[tuple[int, int]]
    unmatched_gt: list[int]
    unmatched_pred: list[int]
    similarity_matrix: np.ndarray


class Matcher(Protocol):
    def match(self, gt: list[Detection], pred: list[Detection]) -> MatchResult: ...
