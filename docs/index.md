# tracking-metrics

A clean, installable Python library for evaluating multi-object tracking.

## Why tracking-metrics?

Most MOT evaluation code is tightly coupled to a specific dataset (MOT17, KITTI, nuScenes…).
`tracking-metrics` defines a minimal common data model and lets you evaluate any tracker, on any
dataset, with a consistent API.

## Features

- **2D box, 2D mask, 3D box** tracking evaluation out of the box
- **IoU and distance-based matchers** (Hungarian assignment)
- **All standard metrics**: TP/FP/FN, IDSW, MOTA, MOTP, IDF1, HOTA, Frag, MT/PT/ML, T-SR, IDCons
- **Batch evaluation** across multiple sequences with averaged and global aggregation
- **YAML config files** for reproducible evaluations
- **JSON, CSV, LaTeX** report output
- **CLI** (`track-metrics evaluate`, `track-metrics evaluate-batch`)
- **Adapter pattern** for integrating custom dataset formats
- No dataset-specific dependencies

## Quick start

```bash
git clone git@github.com:<ORG_OR_USER>/tracking-metrics.git
cd tracking-metrics
pip install -e .
```

```python
from tracking_metrics import (
    Sequence, Box2DIoUMatcher, TrackingEvaluator,
    DetectionCounts, MOTA, MOTP, IDF1, HOTA,
)

gt   = Sequence.from_json("gt.json")
pred = Sequence.from_json("pred.json")

evaluator = TrackingEvaluator(
    matcher=Box2DIoUMatcher(threshold=0.5),
    metrics=[DetectionCounts(), MOTA(), MOTP(), IDF1(), HOTA()],
)
results = evaluator.evaluate(gt, pred)
print(results)
```

## Pages

- [Installation](installation.md)
- [Data format](data_format.md)
- [Matching](matching.md)
- [Metrics](metrics.md)
- [Advanced metrics](advanced_metrics.md)
- [CLI reference](cli.md)
- [Python API](python_api.md)
- [Adapters](adapters.md)
- [2D mask tracking](masks.md)
- [3D tracking](3d_tracking.md)
- [Examples](examples.md)
