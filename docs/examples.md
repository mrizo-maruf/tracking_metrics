# Examples

All examples are in `examples/` and are runnable with `python examples/<name>.py`.

## `toy_2d_boxes.py`

Basic 2D box tracking evaluation with a short synthetic sequence.
Demonstrates: `Box2DIoUMatcher`, `MOTA`, `IDF1`, `HOTA`, `TrackCoverage`.

## `toy_masks.py`

2D mask tracking with binary masks, `MaskIoUMatcher`, and `TemporalIoU`.

## `toy_3d_boxes.py`

3D box tracking with `Box3DIoUMatcher` and `CenterDistanceMatcher`.
Demonstrates `MeanBox3DIoU` and `MeanCenterDistance3D`.

## `batch_eval.py`

Evaluates two sequences from `examples/data/` and reports per-sequence,
average, and global metrics.

```bash
python examples/batch_eval.py
```

## `custom_adapter.py`

Shows how to implement a `SequenceAdapter` for a project-specific annotation format
and evaluate without touching the core library.

```bash
python examples/custom_adapter.py
```

## `evaluate_example.py`

Original quick-start example using the `examples/gt.json` and `examples/pred.json` files.

## Config-driven evaluation

```bash
track-metrics evaluate \
  --gt examples/gt_masks.json \
  --pred examples/pred_masks.json \
  --config examples/configs/mask_eval.yaml
```
