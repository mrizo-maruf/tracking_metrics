# CLI Reference

## Single-sequence evaluation

```bash
track-metrics evaluate \
  --gt gt.json \
  --pred pred.json \
  --matcher box2d-iou \
  --threshold 0.5 \
  --metrics counts --metrics mota --metrics idf1 --metrics hota \
  --output results.json
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gt` | required | Ground-truth JSON file |
| `--pred` | required | Predictions JSON file |
| `--matcher` | `box2d-iou` | `box2d-iou`, `mask-iou`, `box3d-iou`, `center-distance` |
| `--threshold` | `0.5` | IoU threshold (for IoU matchers) |
| `--max-distance` | `0.5` | Max center distance (for `center-distance`) |
| `--metrics` | all | Metric name(s); pass once per metric |
| `--output` | — | Save report (`.json` / `.csv` / `.tex`) |
| `--hota-curves` | off | Include per-threshold HOTA curves in JSON output |
| `--config` | — | YAML config file (see below) |

### Supported metric names

`counts`, `mota`, `motp`, `idsw`, `idf1`, `t-miou`, `t-dice`,
`mean-box3d-iou`, `mean-center-dist-3d`,
`hota`, `frag`, `track-coverage`, `t-sr`, `id-cons`

## Batch evaluation

```bash
track-metrics evaluate-batch \
  --gt-dir path/to/gt/ \
  --pred-dir path/to/pred/ \
  --matcher box2d-iou \
  --threshold 0.5 \
  --metrics mota --metrics idf1 --metrics hota \
  --output batch_results.json
```

Files are paired by stem: `gt/scene_001.json` ↔ `pred/scene_001.json`.
Missing pairs are reported as warnings and skipped.

Output JSON contains three sections: `sequences`, `average`, `global`.

## Config-driven evaluation

```bash
track-metrics evaluate \
  --gt gt.json \
  --pred pred.json \
  --config eval_config.yaml
```

Example config (`examples/configs/mask_eval.yaml`):

```yaml
matcher:
  type: mask-iou
  threshold: 0.5

metrics:
  - counts
  - mota
  - idf1
  - hota

output:
  json: results.json
  csv: results.csv
```

**Precedence**: explicit CLI flags override config values for `matcher` and `threshold`
(when they differ from their defaults). Config `output.*` paths are used unless `--output` is given.

## Output formats

The `--output` extension controls the format:

| Extension | Format |
|-----------|--------|
| `.json` | JSON report |
| `.csv` | CSV (metric, value) |
| `.tex` / `.latex` | LaTeX `tabular` |
