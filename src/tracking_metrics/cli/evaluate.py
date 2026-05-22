from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric
from tracking_metrics.metrics.id_switches import IDSwitchesMetric
from tracking_metrics.metrics.idf1 import IDF1Metric
from tracking_metrics.metrics.mota import MOTAMetric
from tracking_metrics.metrics.motp import MOTPMetric
from tracking_metrics.report.json_report import save_json_report
from tracking_metrics.report.terminal import print_results_table

app = typer.Typer(help="Evaluate multi-object tracking results.")


@app.callback()
def _root() -> None:
    """tracking-metrics: dataset-independent MOT evaluation library."""

_METRIC_MAP = {
    "counts": DetectionCountsMetric,
    "mota": MOTAMetric,
    "motp": MOTPMetric,
    "idsw": IDSwitchesMetric,
    "idf1": IDF1Metric,
}


@app.command()
def evaluate(
    gt: Annotated[Path, typer.Option("--gt", help="Path to ground-truth JSON file.")],
    pred: Annotated[Path, typer.Option("--pred", help="Path to predictions JSON file.")],
    matcher: Annotated[str, typer.Option("--matcher", help="Matcher type.")] = "box2d-iou",
    threshold: Annotated[float, typer.Option("--threshold", help="IoU threshold.")] = 0.5,
    metrics: Annotated[
        list[str] | None, typer.Option("--metrics", help="Metrics to compute.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Path to save JSON report.")
    ] = None,
) -> None:
    if matcher != "box2d-iou":
        typer.echo(f"Unknown matcher: {matcher}. Only 'box2d-iou' is supported.", err=True)
        raise typer.Exit(code=1)

    selected_metrics = metrics or list(_METRIC_MAP.keys())
    unknown = [m for m in selected_metrics if m not in _METRIC_MAP]
    if unknown:
        typer.echo(f"Unknown metrics: {unknown}. Supported: {list(_METRIC_MAP.keys())}", err=True)
        raise typer.Exit(code=1)

    metric_instances = [_METRIC_MAP[m]() for m in selected_metrics]

    gt_seq = Sequence.from_json(gt)
    pred_seq = Sequence.from_json(pred)

    iou_matcher = Box2DIoUMatcher(threshold=threshold)
    evaluator = TrackingEvaluator(matcher=iou_matcher, metrics=metric_instances)
    results = evaluator.evaluate(gt_seq, pred_seq)

    print_results_table(results)

    if output is not None:
        save_json_report(results, output)
        typer.echo(f"Results saved to {output}")
