from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tracking_metrics.data.sequence import Sequence


class SequenceAdapter(Protocol):
    """Protocol for loading dataset-specific tracking data into the common Sequence format.

    Implement this in your project to convert a specific dataset's annotation
    and prediction format into ``Sequence`` objects that the evaluator understands.

    Example
    -------
    ::

        class MyDatasetAdapter:
            def load_ground_truth(self, path: str | Path) -> Sequence:
                ...  # parse dataset-specific annotations

            def load_predictions(self, path: str | Path) -> Sequence:
                ...  # parse tracker output

        adapter = MyDatasetAdapter()
        gt = adapter.load_ground_truth("annotations/scene_001")
        pred = adapter.load_predictions("tracker_output/scene_001")
    """

    def load_ground_truth(self, path: str | Path) -> Sequence:
        """Load ground-truth annotations from *path* and return a ``Sequence``."""
        ...

    def load_predictions(self, path: str | Path) -> Sequence:
        """Load tracker predictions from *path* and return a ``Sequence``."""
        ...


def save_sequence_json(sequence: Sequence, path: str | Path) -> None:
    """Serialize *sequence* to the tracking-metrics JSON format.

    Useful in adapter implementations to cache converted sequences.
    """
    sequence.to_json(path)
