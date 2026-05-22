# tracking-metrics

A clean, installable Python library for evaluating multi-object tracking from common data structures, independent of any dataset.

## Installation

```bash
pip install -e .
```

With RLE mask support (requires `pycocotools`):

```bash
pip install -e ".[masks]"
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

## JSON Input Format

Both ground-truth and prediction files use the same format:

```json
{
  "sequence_name": "scene_001",
  "frames": [
    {
      "frame_id": 0,
      "detections": [
        {
          "track_id": 1,
          "class_id": "person",
          "score": 0.98,
          "bbox2d": [x1, y1, x2, y2]
        }
      ]
    }
  ]
}
```

- `track_id` may be an integer or string; always stored as `str` internally.
- `class_id` and `score` are optional.
- `bbox2d` is `[x1, y1, x2, y2]` in pixel coordinates.
- A detection can contain both `bbox2d` and `mask`, or neither.

### Mask extension

Dense binary mask:

```json
{
  "track_id": 1,
  "mask": {
    "type": "binary",
    "size": [H, W],
    "data": [[0, 1, 1, 0], ...]
  }
}
```

RLE mask (requires `pycocotools`):

```json
{
  "track_id": 1,
  "mask": {
    "type": "rle",
    "size": [480, 640],
    "counts": "encoded_counts_here"
  }
}
```

## Python API

### 2D Box Tracking

```python
from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric
from tracking_metrics.metrics.id_switches import IDSwitchesMetric
from tracking_metrics.metrics.idf1 import IDF1Metric
from tracking_metrics.metrics.mota import MOTAMetric
from tracking_metrics.metrics.motp import MOTPMetric
from tracking_metrics.report.terminal import print_results_table

gt = Sequence.from_json("gt.json")
pred = Sequence.from_json("pred.json")

matcher = Box2DIoUMatcher(threshold=0.5)
metrics = [DetectionCountsMetric(), IDSwitchesMetric(), MOTAMetric(), MOTPMetric(), IDF1Metric()]

evaluator = TrackingEvaluator(matcher=matcher, metrics=metrics)
results = evaluator.evaluate(gt, pred)

print_results_table(results)
```

### 2D Mask Tracking

```python
from tracking_metrics.matching.mask_iou_matcher import MaskIoUMatcher
from tracking_metrics.metrics.temporal_iou import TemporalIoUMetric

gt = Sequence.from_json("gt_masks.json")
pred = Sequence.from_json("pred_masks.json")

matcher = MaskIoUMatcher(threshold=0.5)
metrics = [
    DetectionCountsMetric(), IDSwitchesMetric(),
    MOTAMetric(), MOTPMetric(), IDF1Metric(), TemporalIoUMetric(),
]

evaluator = TrackingEvaluator(matcher=matcher, metrics=metrics)
results = evaluator.evaluate(gt, pred)

print_results_table(results)
```

Access frame-level events directly:

```python
evaluation_result = evaluator.evaluate_events(gt, pred)

for fr in evaluation_result.frame_results:
    print(f"Frame {fr.frame_id}: {len(fr.matches)} matches, "
          f"{len(fr.false_positives)} FP, {len(fr.false_negatives)} FN, "
          f"{len(fr.id_switches)} ID switches")
```

## CLI

### Box tracking

```bash
track-metrics evaluate \
  --gt examples/gt.json \
  --pred examples/pred.json \
  --matcher box2d-iou \
  --threshold 0.5 \
  --metrics counts --metrics mota --metrics motp --metrics idsw --metrics idf1 \
  --output results.json
```

### Mask tracking

```bash
track-metrics evaluate \
  --gt examples/gt_masks.json \
  --pred examples/pred_masks.json \
  --matcher mask-iou \
  --threshold 0.5 \
  --metrics counts --metrics mota --metrics motp --metrics idsw --metrics idf1 --metrics t-miou \
  --output results_mask.json
```

Pass `--metrics` once per metric name. Omitting `--metrics` entirely runs all metrics.

Supported matchers: `box2d-iou`, `mask-iou`.

## Supported Metrics (v0.2)

| Metric | Description |
|--------|-------------|
| `TP` | True positives (matched detections) |
| `FP` | False positives (unmatched predictions) |
| `FN` | False negatives (unmatched ground-truth) |
| `GT` | Total ground-truth detections |
| `Pred` | Total predicted detections |
| `IDSW` | Number of identity switches |
| `MOTA` | Multi-Object Tracking Accuracy: `1 - (FN + FP + IDSW) / GT` |
| `MOTP` | Average similarity of matched pairs (IoU for box matcher; mask IoU for mask matcher) |
| `IDF1` | ID F1 score |
| `IDP` | ID Precision |
| `IDR` | ID Recall |
| `IDTP` | ID True Positives |
| `IDFP` | ID False Positives |
| `IDFN` | ID False Negatives |
| `T-mIoU` | Temporal mean mask IoU over all matched pairs with masks |
| `T-Dice` | Temporal mean Dice score over all matched pairs with masks |

`MOTP` always reflects the matcher's similarity measure. With `box2d-iou` it is average box IoU; with `mask-iou` it is average mask IoU. `T-mIoU` specifically measures mask IoU regardless of which matcher was used.

## Current Limitations

- 3D boxes are not supported.
- HOTA is not implemented.
- Visualization tools are not included.
- No dataset-specific adapters. Bring your own converter to the JSON format above.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```
