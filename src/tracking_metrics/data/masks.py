from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Mask2D:
    data: np.ndarray | None = None
    rle: dict[str, Any] | None = None
    size: tuple[int, int] | None = None

    def to_binary(self) -> np.ndarray:
        if self.data is not None:
            arr = np.asarray(self.data, dtype=bool)
            return arr

        if self.rle is not None:
            try:
                from pycocotools import mask as coco_mask
            except ImportError as exc:
                raise ImportError(
                    "RLE mask decoding requires pycocotools. "
                    "Install it with: pip install tracking-metrics[masks]"
                ) from exc
            decoded = coco_mask.decode(self.rle)
            return decoded.astype(bool)

        raise ValueError("Mask2D has neither data nor rle set.")

    def area(self) -> int:
        return int(self.to_binary().sum())
