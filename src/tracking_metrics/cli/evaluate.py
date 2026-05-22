from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.evaluator import TrackingEvaluator
from tracking_metrics.matching.box2d_iou_matcher import Box2DIoUMatcher
from tracking_metrics.matching.box3d_iou_matcher import Box3DIoUMatcher
from tracking_metrics.matching.center_distance_matcher import CenterDistanceMatcher
from tracking_metrics.matching.mask_iou_matcher import MaskIoUMatcher
from tracking_metrics.metrics.detection_counts import DetectionCountsMetric
from tracking_metrics.metrics.fragmentation import FragmentationsMetric
from tracking_metrics.metrics.hota import HOTAMetric
from tracking_metrics.metrics.id_consistency import IDConsistencyMetric
from tracking_metrics.metrics.id_switches import IDSwitchesMetric
from tracking_metrics.metrics.idf1 import IDF1Metric
from tracking_metrics.metrics.localization3d import MeanBox3DIoU, MeanCenterDistance3D
from tracking_metrics.metrics.mota import MOTAMetric
from tracking_metrics.metrics.motp import MOTPMetric
from tracking_metrics.metrics.temporal_iou import TemporalDiceMetric, TemporalIoUMetric
from tracking_metrics.metrics.track_coverage import TrackCoverageMetric
from tracking_metrics.metrics.track_survival import TrackSurvivalRateMetric
from tracking_metrics.report.csv_report import save_csv_report
from tracking_metrics.report.json_report import save_json_report
from tracking_metrics.report.latex_table import save_latex_report
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
    "t-miou": TemporalIoUMetric,
    "t-dice": TemporalDiceMetric,
    "mean-center-dist-3d": MeanCenterDistance3D,
    "mean-box3d-iou": MeanBox3DIoU,
    "hota": HOTAMetric,
    "frag": FragmentationsMetric,
    "track-coverage": TrackCoverageMetric,
    "t-sr": TrackSurvivalRateMetric,
    "id-cons": IDConsistencyMetric,
}

_MATCHERS = ("box2d-iou", "mask-iou", "box3d-iou", "center-distance")


@app.command()
def evaluate(
    gt: Annotated[Path, typer.Option("--gt", help="Path to ground-truth JSON file.")],
    pred: Annotated[Path, typer.Option("--pred", help="Path to predictions JSON file.")],
    matcher: Annotated[
        str, typer.Option("--matcher", help=f"Matcher type. One of: {_MATCHERS}.")
    ] = "box2d-iou",
    threshold: Annotated[
        float, typer.Option("--threshold", help="IoU threshold (box2d-iou, mask-iou, box3d-iou).")
    ] = 0.5,
    max_distance: Annotated[
        float, typer.Option("--max-distance", help="Max center distance for center-distance matcher.")
    ] = 0.5,
    metrics: Annotated[
        list[str] | None, typer.Option("--metrics", help="Metrics to compute.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Path to save report (.json/.csv/.tex).")
    ] = None,
    hota_curves: Annotated[
        bool, typer.Option("--hota-curves/--no-hota-curves", help="Include per-threshold HOTA curves in output.")
    ] = False,
) -> None:
    if matcher not in _MATCHERS:
        typer.echo(f"Unknown matcher: {matcher!r}. Supported: {_MATCHERS}", err=True)
        raise typer.Exit(code=1)

    selected_metrics = metrics or list(_METRIC_MAP.keys())
    unknown = [m for m in selected_metrics if m not in _METRIC_MAP]
    if unknown:
        typer.echo(f"Unknown metrics: {unknown}. Supported: {list(_METRIC_MAP.keys())}", err=True)
        raise typer.Exit(code=1)

    metric_instances = []
    for m in selected_metrics:
        cls = _METRIC_MAP[m]
        if m == "hota":
            metric_instances.append(cls(return_curves=hota_curves))
        else:
            metric_instances.append(cls())

    gt_seq = Sequence.from_json(gt)
    pred_seq = Sequence.from_json(pred)

    if matcher == "box2d-iou":
        active_matcher = Box2DIoUMatcher(threshold=threshold)
    elif matcher == "mask-iou":
        active_matcher = MaskIoUMatcher(threshold=threshold)
    elif matcher == "box3d-iou":
        active_matcher = Box3DIoUMatcher(threshold=threshold)
    else:
        active_matcher = CenterDistanceMatcher(max_distance=max_distance)

    evaluator = TrackingEvaluator(matcher=active_matcher, metrics=metric_instances)
    results = evaluator.evaluate(gt_seq, pred_seq)

    print_results_table(results)

    if output is not None:
        suffix = Path(output).suffix.lower()
        if suffix == ".csv":
            save_csv_report(results, output)
        elif suffix in (".tex", ".latex"):
            save_latex_report(results, output)
        else:
            save_json_report(results, output)
        typer.echo(f"Results saved to {output}")
