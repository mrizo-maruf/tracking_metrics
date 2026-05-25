# Visualization

`tracking-metrics` ships an optional visualization layer built on
[matplotlib](https://matplotlib.org/).  All functions are lazy-imported so the
core library stays lightweight.

## Installation

```bash
pip install -e '.[viz]'
# or
pip install matplotlib
```

Matplotlib is listed under the `viz` optional dependency group in
`pyproject.toml`.

---

## Metric summary bar chart

```python
from tracking_metrics.visualization import plot_metric_summary

results = evaluator.evaluate(gt_seq, pred_seq)
fig = plot_metric_summary(results, save_path="summary.png")
```

Bars are colour-coded: green ≥ 0.8, orange ≥ 0.5, red < 0.5 (for metrics in
the 0–1 range).  Pass `metrics=[...]` to select a specific subset.

---

## HOTA threshold curves

Requires running `HOTAMetric(return_curves=True)`.

```python
from tracking_metrics.metrics.hota import HOTAMetric
from tracking_metrics.visualization import plot_hota_curves

results = evaluator.evaluate(gt_seq, pred_seq)  # evaluator has HOTAMetric(return_curves=True)
fig = plot_hota_curves(results, save_path="hota_curves.png")
```

Use `plot_metric_curve(thresholds, curves)` for a generic multi-curve plot
over any shared threshold axis.

---

## Per-frame event timeline

```python
from tracking_metrics.visualization import plot_frame_timeline

eval_result = evaluator.evaluate_events(gt_seq, pred_seq)
fig = plot_frame_timeline(eval_result, save_path="timeline.png")
```

Shows stacked bars of TP / FP / FN / ID-switch counts per frame.

---

## Track coverage and survival

```python
from tracking_metrics.visualization import plot_track_coverage, plot_track_survival

plot_track_coverage(eval_result, save_path="coverage.png")
plot_track_survival(eval_result, save_path="survival.png")
```

- **Coverage**: shows which frames each GT track is active in.
- **Survival**: fraction of GT tracks still alive at each frame offset from
  their start.

---

## ID switch timeline

```python
from tracking_metrics.visualization import plot_id_switch_timeline, get_id_switch_table

fig = plot_id_switch_timeline(eval_result, save_path="id_switches.png")

table = get_id_switch_table(eval_result)
# [{"frame_id": 3, "gt_track_id": "1", "prev_pred_id": "a", "new_pred_id": "c"}, ...]
```

---

## GT–Pred association heatmap

```python
from tracking_metrics.visualization import plot_association_heatmap

fig = plot_association_heatmap(eval_result, save_path="association.png")
# normalize=False for raw frame counts instead of fractions
```

---

## Per-frame 2-D overlay

```python
from tracking_metrics.visualization import plot_frame_overlay

frame_result = eval_result.frame_results[3]
fig = plot_frame_overlay(frame_result, save_path="frame_3.png")

# Optional background image
import numpy as np
img = np.zeros((480, 640, 3), dtype=np.uint8)
fig = plot_frame_overlay(frame_result, image=img, save_path="frame_3_bg.png")
```

GT detections are plotted as green squares, predictions as blue triangles.
ID switch events are highlighted with a red ring around the GT position.

---

## CLI commands

### `visualize`

Reads a JSON results file and generates `summary.png` (and `hota_curves.png`
if curve data is present).

```bash
track-metrics evaluate --gt gt.json --pred pred.json \
    --metrics mota motp hota --hota-curves \
    --output results.json

track-metrics visualize --results results.json --output-dir plots/
```

### `visualize-events`

Runs event evaluation on a GT/pred pair and generates all event-level plots.

```bash
track-metrics visualize-events \
    --gt gt.json --pred pred.json \
    --output-dir plots/ \
    --title "My Sequence"
```

Output files: `frame_timeline.png`, `track_coverage.png`,
`track_survival.png`, `id_switches.png`, `association.png`.

---

## Full example

See [examples/visualize_results.py](../examples/visualize_results.py) for a
complete self-contained script that builds a synthetic sequence, runs
evaluation, and saves all eight plot types.
