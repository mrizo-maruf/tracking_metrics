# Adapters

`tracking-metrics` is dataset-independent. It does not include adapters for specific datasets —
those belong in your project.

## The adapter protocol

```python
from tracking_metrics.adapters.base import SequenceAdapter

class SequenceAdapter(Protocol):
    def load_ground_truth(self, path: str | Path) -> Sequence: ...
    def load_predictions(self, path: str | Path) -> Sequence: ...
```

Implement this in your project to convert dataset annotations into `Sequence` objects.

## Example

```python
from pathlib import Path
from tracking_metrics import Sequence, Box2D, Detection, Frame
from tracking_metrics.adapters.base import SequenceAdapter, save_sequence_json

class KITTIAdapter:
    def load_ground_truth(self, path: str | Path) -> Sequence:
        # parse KITTI label files → Detection objects
        frames = []
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            fid, track_id = int(parts[0]), parts[1]
            x1, y1, x2, y2 = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
            # ... group by frame_id, build Frame/Detection objects
        return Sequence(name=str(path), frames=frames)

    def load_predictions(self, path: str | Path) -> Sequence:
        ...
```

## Suggested adapters (not included)

These would live in your project or a separate package:

| Adapter | Dataset |
|---------|---------|
| `IsaacSimAdapter` | NVIDIA Isaac Sim synthetic data |
| `ScanNetPPAdapter` | ScanNet++ scene annotations |
| `CODaAdapter` | CODa driving dataset |
| `ThreeRScanAdapter` | 3RScan indoor scene re-scans |
| `MOT17Adapter` | MOT17 benchmark |
| `KITTIAdapter` | KITTI tracking benchmark |

## Caching converted sequences

Once converted, cache sequences in the standard JSON format to avoid re-parsing:

```python
from tracking_metrics.adapters.base import save_sequence_json

adapter = MyAdapter()
gt = adapter.load_ground_truth("raw_annotations/")
save_sequence_json(gt, "converted/gt_scene_001.json")
```
