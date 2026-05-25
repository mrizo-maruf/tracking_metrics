# tracking-metrics

A clean, installable Python library for evaluating multi-object tracking — independent of any
dataset or annotation format.

## Why tracking-metrics?

Most MOT evaluation code is coupled to a single dataset (MOT17, KITTI, nuScenes…). This library
defines a minimal data model and lets you evaluate any tracker on any dataset with one consistent
API. Bring your own annotation converter; the evaluation logic is reusable.

---

## Installation

This package is not on PyPI. Install directly from GitHub or from a local clone.

> **Note:** the install name and import name differ.
> Install source: `tracking-metrics` · Python import: `tracking_metrics`

### Option 1 — Editable development install (recommended)

```bash
git clone git@github.com:<ORG_OR_USER>/tracking-metrics.git
cd tracking-metrics
pip install -e .
```

### Option 2 — Install directly from GitHub

```bash
pip install git+https://github.com/<ORG_OR_USER>/tracking-metrics.git
```

Install a specific tag:

```bash
pip install git+https://github.com/<ORG_OR_USER>/tracking-metrics.git@v0.5.0
```

### Option 3 — Add to `requirements.txt`

```
git+https://github.com/<ORG_OR_USER>/tracking-metrics.git
```

### With COCO RLE mask support

```bash
pip install -e ".[masks]"
# or
pip install "git+https://github.com/<ORG_OR_USER>/tracking-metrics.git#egg=tracking-metrics[masks]"
```

### Development dependencies

```bash
pip install -e ".[dev]"
```

### Using from another project (e.g. TAGY)

```
~/work/
├── tagy/
└── tracking-metrics/
```

```bash
conda activate tagy
cd ~/work/tracking-metrics
pip install -e .
```

Then inside your project:

```python
from tracking_metrics import TrackingEvaluator
from tracking_metrics.matching import MaskIoUMatcher
from tracking_metrics.metrics import MOTA, MOTP, IDF1, IDSwitches, HOTA
```

---

## Quick Python example

```python
from tracking_metrics import (
    Sequence, Box2DIoUMatcher, TrackingEvaluator,
    DetectionCounts, MOTA, MOTP, IDF1, HOTA,
)
from tracking_metrics.report import print_results_table

gt   = Sequence.from_json("gt.json")
pred = Sequence.from_json("pred.json")

evaluator = TrackingEvaluator(
    matcher=Box2DIoUMatcher(threshold=0.5),
    metrics=[DetectionCounts(), MOTA(), MOTP(), IDF1(), HOTA()],
)
results = evaluator.evaluate(gt, pred)
print_results_table(results)
```

---

## Quick CLI example

```bash
track-metrics evaluate \
  --gt gt.json --pred pred.json \
  --matcher box2d-iou --threshold 0.5 \
  --metrics counts --metrics mota --metrics idf1 --metrics hota \
  --output results.json
```

---

## Supported modalities

### 2D boxes

```python
from tracking_metrics import Box2DIoUMatcher, Box2D
matcher = Box2DIoUMatcher(threshold=0.5)
```

### 2D masks

```python
from tracking_metrics import MaskIoUMatcher
matcher = MaskIoUMatcher(threshold=0.5)
```

Requires `mask` field in detections (binary array or COCO RLE). See [docs/masks.md](docs/masks.md).

### 3D boxes

```python
from tracking_metrics import Box3DIoUMatcher, CenterDistanceMatcher
matcher = Box3DIoUMatcher(threshold=0.25)
# or
matcher = CenterDistanceMatcher(max_distance=0.5)
```

See [docs/3d_tracking.md](docs/3d_tracking.md).

---

## Supported matchers

| Matcher | Class | Threshold |
|---------|-------|-----------|
| 2D box IoU | `Box2DIoUMatcher` | `threshold` (IoU) |
| 2D mask IoU | `MaskIoUMatcher` | `threshold` (IoU) |
| 3D box IoU | `Box3DIoUMatcher` | `threshold` (IoU) |
| Center distance | `CenterDistanceMatcher` | `max_distance` (world units) |

All matchers accept `class_aware=True` to restrict matching to same-class detections.

---

## Supported metrics

| Key | Class | Description |
|-----|-------|-------------|
| `TP/FP/FN/GT/Pred` | `DetectionCounts` | Detection count statistics |
| `IDSW` | `IDSwitches` | Identity switches |
| `MOTA` | `MOTA` | Multi-Object Tracking Accuracy |
| `MOTP` | `MOTP` | Multi-Object Tracking Precision |
| `IDF1/IDP/IDR` | `IDF1` | Identity F1 score |
| `HOTA/DetA/AssA/LocA` | `HOTA` | Higher Order Tracking Accuracy |
| `Frag` | `Fragmentations` | GT track fragmentation count |
| `MT/PT/ML` | `TrackCoverage` | Mostly/Partially/Mostly-Lost track counts |
| `T-SR` | `TrackSurvivalRate` | Fraction of GT tracks with ≥1 match |
| `IDCons` | `IDConsistency` | Mean majority-ID consistency per GT track |
| `T-mIoU/T-Dice` | `TemporalIoU` | Temporal mask metrics (mask tracking) |
| `MeanBox3DIoU` | `MeanBox3DIoU` | Mean 3D IoU of matched pairs |
| `MeanCenterDist3D` | `MeanCenterDistance3D` | Mean center distance of matched pairs |

---

## Data format

```json
{
  "sequence_name": "scene_001",
  "frames": [
    {
      "frame_id": 0,
      "detections": [
        {"track_id": "1", "bbox2d": [x1, y1, x2, y2]},
        {"track_id": "2", "bbox2d": [x1, y1, x2, y2], "class_id": "car", "score": 0.95}
      ]
    }
  ]
}
```

`track_id` may be integer or string. `class_id` and `score` are optional.
A detection may include `bbox2d`, `mask`, and `bbox3d` simultaneously.

See [docs/data_format.md](docs/data_format.md) for full format documentation.

---

## Batch evaluation

Evaluate multiple sequences and get per-sequence, averaged, and global aggregated metrics:

```python
from tracking_metrics.evaluation.batch import BatchEvaluator, SequencePair

pairs = [
    SequencePair("scene_001", gt1, pred1),
    SequencePair("scene_002", gt2, pred2),
]

batch = BatchEvaluator(evaluator)
results = batch.evaluate(pairs)

results["sequences"]["scene_001"]  # per-sequence scores
results["average"]                 # arithmetic mean of scalar metrics
results["global"]                  # metrics on merged frame pool
```

CLI:

```bash
track-metrics evaluate-batch \
  --gt-dir gt/ --pred-dir pred/ \
  --metrics mota --metrics idf1 --metrics hota \
  --output batch_results.json
```

Files are matched by stem: `gt/scene_001.json` ↔ `pred/scene_001.json`.

---

## Config files

```yaml
# eval_config.yaml
matcher:
  type: box2d-iou
  threshold: 0.5

metrics:
  - counts
  - mota
  - idf1
  - hota

output:
  json: results.json
  csv: results.csv
  latex: results.tex
```

```bash
track-metrics evaluate --gt gt.json --pred pred.json --config eval_config.yaml
```

---

## Adapter concept

`tracking-metrics` is dataset-independent. Implement the `SequenceAdapter` protocol in your
project to convert any dataset's format into `Sequence` objects:

```python
from tracking_metrics.adapters.base import SequenceAdapter, save_sequence_json

class MyDatasetAdapter:
    def load_ground_truth(self, path) -> Sequence: ...
    def load_predictions(self, path) -> Sequence: ...

adapter = MyDatasetAdapter()
gt = adapter.load_ground_truth("annotations/")
```

See [docs/adapters.md](docs/adapters.md) and `examples/custom_adapter.py`.

---

## Visualization

An optional matplotlib-based visualization layer is included.

Install with:

```bash
pip install -e '.[viz]'
```

```python
from tracking_metrics.visualization import (
    plot_metric_summary,
    plot_hota_curves,
    plot_frame_timeline,
    plot_track_coverage,
    plot_association_heatmap,
)

eval_result = evaluator.evaluate_events(gt_seq, pred_seq)
metric_results = evaluator.compute_metrics(eval_result)

plot_metric_summary(metric_results, save_path="summary.png")
plot_hota_curves(metric_results, save_path="hota_curves.png")
plot_frame_timeline(eval_result, save_path="timeline.png")
plot_track_coverage(eval_result, save_path="coverage.png")
plot_association_heatmap(eval_result, save_path="association.png")
```

**CLI visualization commands:**

```bash
# Generate summary + HOTA curves from a results JSON
track-metrics visualize --results results.json --output-dir plots/

# Run evaluation and generate all event-level plots
track-metrics visualize-events --gt gt.json --pred pred.json --output-dir plots/
```

See [docs/visualization.md](docs/visualization.md) and
[examples/visualize_results.py](examples/visualize_results.py).

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

See [RELEASE.md](RELEASE.md) for the release checklist and [CHANGELOG.md](CHANGELOG.md) for
version history.

---

## Limitations

- Oriented 3D IoU (using yaw) is not yet implemented.
- HOTA is computed globally per sequence; multi-sequence HOTA is averaged over sequences.
- Visualization requires the optional `[viz]` extra (`pip install -e '.[viz]'`).
- No dataset-specific adapters in the core library. Bring your own converter.
