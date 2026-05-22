# tracking-metrics

A clean, installable Python library for evaluating multi-object tracking from common data structures, independent of any dataset.

## Installation

```bash
pip install -e .
```

Or with development dependencies:

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

- `track_id` may be an integer or string; it is always stored as `str` internally.
- `class_id` and `score` are optional.
- `bbox2d` is `[x1, y1, x2, y2]` in pixel coordinates.

## Python API

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

Access frame-level events directly:

```python
evaluation_result = evaluator.evaluate_events(gt, pred)

for fr in evaluation_result.frame_results:
    print(f"Frame {fr.frame_id}: {len(fr.matches)} matches, "
          f"{len(fr.false_positives)} FP, {len(fr.false_negatives)} FN, "
          f"{len(fr.id_switches)} ID switches")
```

## CLI

```bash
track-metrics evaluate \
  --gt examples/gt.json \
  --pred examples/pred.json \
  --matcher box2d-iou \
  --threshold 0.5 \
  --metrics counts --metrics mota --metrics motp --metrics idsw --metrics idf1 \
  --output results.json
```

If `--output` is omitted, results are printed to the terminal only.

Pass `--metrics` once per metric name. Omitting `--metrics` entirely runs all metrics.

Supported `--metrics` values: `counts`, `mota`, `motp`, `idsw`, `idf1`.

## Supported Metrics (v0.1)

| Metric | Description |
|--------|-------------|
| `TP` | True positives (matched detections) |
| `FP` | False positives (unmatched predictions) |
| `FN` | False negatives (unmatched ground-truth) |
| `GT` | Total ground-truth detections |
| `Pred` | Total predicted detections |
| `IDSW` | Number of identity switches |
| `MOTA` | Multi-Object Tracking Accuracy: `1 - (FN + FP + IDSW) / GT` |
| `MOTP` | Multi-Object Tracking Precision: average IoU of matched pairs |
| `IDF1` | ID F1 score |
| `IDP` | ID Precision |
| `IDR` | ID Recall |
| `IDTP` | ID True Positives |
| `IDFP` | ID False Positives |
| `IDFN` | ID False Negatives |

## Current Limitations

- Only 2D bounding box matching is supported (IoU-based). Masks and 3D boxes will be added in a later version.
- Only one matcher is available: `box2d-iou`. Additional matchers (e.g., center-distance, 3D IoU) are planned.
- HOTA is not implemented in v0.1.
- Visualization tools are not included.
- No dataset-specific adapters. Bring your own converter to the JSON format above.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```
