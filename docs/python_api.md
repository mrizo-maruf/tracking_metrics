# Python API

## Top-level imports

All commonly used classes are available directly from `tracking_metrics`:

```python
from tracking_metrics import (
    # Data
    Sequence, Frame, Detection, Box2D, Box3D, Mask2D,
    # Evaluator
    TrackingEvaluator, EvaluationResult, FrameResult,
    # Matchers
    Box2DIoUMatcher, MaskIoUMatcher, Box3DIoUMatcher, CenterDistanceMatcher,
    # Metrics
    DetectionCounts, IDSwitches, MOTA, MOTP, IDF1, HOTA,
    Fragmentations, TrackCoverage, TrackSurvivalRate, IDConsistency,
    TemporalIoU, MeanBox3DIoU, MeanCenterDistance3D,
)
```

## Single-sequence evaluation

```python
from tracking_metrics import TrackingEvaluator, Box2DIoUMatcher, Sequence, MOTA, IDF1

gt   = Sequence.from_json("gt.json")
pred = Sequence.from_json("pred.json")

evaluator = TrackingEvaluator(
    matcher=Box2DIoUMatcher(threshold=0.5),
    metrics=[MOTA(), IDF1()],
)

# Returns a flat dict of metric name → value
results = evaluator.evaluate(gt, pred)

# Returns EvaluationResult with per-frame detail
ev_result = evaluator.evaluate_events(gt, pred)
for fr in ev_result.frame_results:
    print(fr.frame_id, len(fr.matches), len(fr.false_positives))
```

## Batch evaluation

```python
from tracking_metrics.evaluation.batch import BatchEvaluator, SequencePair

pairs = [
    SequencePair("s1", gt1, pred1),
    SequencePair("s2", gt2, pred2),
]

batch = BatchEvaluator(evaluator)
results = batch.evaluate(pairs)

results["sequences"]["s1"]   # per-sequence scores
results["average"]           # arithmetic mean of scalar metrics
results["global"]            # metrics on merged frame results
```

## Reports

```python
from tracking_metrics.report import (
    print_results_table,
    save_json_report,
    save_csv_report,
    save_latex_report,
    format_summary,
)

print_results_table(results)
save_json_report(results, "results.json")
save_csv_report(results, "results.csv")
save_latex_report(results, "results.tex", caption="Results")
print(format_summary(results))
```

## Config files

```python
from tracking_metrics.config import load_config

cfg = load_config("eval_config.yaml")
# cfg.matcher.type, cfg.metrics, cfg.output.json, ...
```

## HOTA with curves

```python
from tracking_metrics import HOTA

metric = HOTA(return_curves=True)
results = evaluator.evaluate(gt, pred)
# results["HOTA"], results["HOTA_curve"], results["DetA_curve"], ...
```
