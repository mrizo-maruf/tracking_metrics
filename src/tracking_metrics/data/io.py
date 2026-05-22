from pathlib import Path

from tracking_metrics.data.sequence import Sequence


def load_sequence(path: str | Path) -> Sequence:
    return Sequence.from_json(path)


def save_sequence(seq: Sequence, path: str | Path) -> None:
    seq.to_json(path)
