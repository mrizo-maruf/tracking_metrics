"""Batch evaluation example: evaluate multiple sequences and aggregate results."""

from pathlib import Path

from tracking_metrics import (
    IDF1,
    MOTA,
    MOTP,
    Box2DIoUMatcher,
    DetectionCounts,
    Fragmentations,
    IDSwitches,
    Sequence,
    TrackCoverage,
    TrackingEvaluator,
)
from tracking_metrics.evaluation.batch import BatchEvaluator, SequencePair
from tracking_metrics.report import print_results_table, save_json_report

here = Path(__file__).parent

# Pair GT and prediction files by sequence name
pairs = [
    SequencePair(
        name="scene_001",
        gt=Sequence.from_json(here / "data" / "gt_scene_001.json"),
        pred=Sequence.from_json(here / "data" / "pred_scene_001.json"),
    ),
    SequencePair(
        name="scene_002",
        gt=Sequence.from_json(here / "data" / "gt_scene_002.json"),
        pred=Sequence.from_json(here / "data" / "pred_scene_002.json"),
    ),
]

evaluator = TrackingEvaluator(
    matcher=Box2DIoUMatcher(threshold=0.5),
    metrics=[
        DetectionCounts(),
        IDSwitches(),
        MOTA(),
        MOTP(),
        IDF1(),
        Fragmentations(),
        TrackCoverage(),
    ],
)

batch = BatchEvaluator(evaluator)
results = batch.evaluate(pairs)

print("\n=== Per-sequence results ===")
for name, scores in results["sequences"].items():
    print(f"\n  {name}:")
    for k, v in scores.items():
        if isinstance(v, (int, float)):
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

print("\n=== Average across sequences ===")
print_results_table(results["average"])

print("\n=== Global (all frames merged) ===")
print_results_table(results["global"])

save_json_report(results, here / "batch_results.json")
print("\nBatch results saved to examples/batch_results.json")
