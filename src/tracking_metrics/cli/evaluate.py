from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from tracking_metrics.data.sequence import Sequence
from tracking_metrics.evaluation.batch import BatchEvaluator, SequencePair
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


def _build_matcher(matcher: str, threshold: float, max_distance: float):
    if matcher == "box2d-iou":
        return Box2DIoUMatcher(threshold=threshold)
    if matcher == "mask-iou":
        return MaskIoUMatcher(threshold=threshold)
    if matcher == "box3d-iou":
        return Box3DIoUMatcher(threshold=threshold)
    return CenterDistanceMatcher(max_distance=max_distance)


def _build_metrics(selected: list[str], hota_curves: bool) -> list:
    instances = []
    for m in selected:
        cls = _METRIC_MAP[m]
        instances.append(cls(return_curves=hota_curves) if m == "hota" else cls())
    return instances


def _save_output(results: dict[str, Any], output: Path) -> None:
    suffix = output.suffix.lower()
    if suffix == ".csv":
        save_csv_report(results, output)
    elif suffix in (".tex", ".latex"):
        save_latex_report(results, output)
    else:
        save_json_report(results, output)
    typer.echo(f"Results saved to {output}")


def _validate_metrics(selected: list[str]) -> None:
    unknown = [m for m in selected if m not in _METRIC_MAP]
    if unknown:
        typer.echo(f"Unknown metrics: {unknown}. Supported: {list(_METRIC_MAP.keys())}", err=True)
        raise typer.Exit(code=1)


def _validate_matcher(matcher: str) -> None:
    if matcher not in _MATCHERS:
        typer.echo(f"Unknown matcher: {matcher!r}. Supported: {_MATCHERS}", err=True)
        raise typer.Exit(code=1)


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
        bool,
        typer.Option(
            "--hota-curves/--no-hota-curves",
            help="Include per-threshold HOTA curves in output.",
        ),
    ] = False,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML evaluation config file.")
    ] = None,
) -> None:
    # Apply config first; explicit CLI args override config values when non-default.
    cfg_output_paths: list[tuple[str | None, Any]] = []
    if config is not None:
        from tracking_metrics.config.loader import load_config

        cfg = load_config(config)
        if matcher == "box2d-iou":
            matcher = cfg.matcher.type
        if threshold == 0.5:
            threshold = cfg.matcher.threshold
        if max_distance == 0.5:
            max_distance = cfg.matcher.max_distance
        if metrics is None:
            metrics = cfg.metrics or None
        if not hota_curves:
            hota_curves = cfg.hota_curves
        cfg_output_paths = [
            (cfg.output.json, save_json_report),
            (cfg.output.csv, save_csv_report),
            (cfg.output.latex, save_latex_report),
        ]

    _validate_matcher(matcher)
    selected_metrics = metrics or list(_METRIC_MAP.keys())
    _validate_metrics(selected_metrics)

    metric_instances = _build_metrics(selected_metrics, hota_curves)
    active_matcher = _build_matcher(matcher, threshold, max_distance)

    gt_seq = Sequence.from_json(gt)
    pred_seq = Sequence.from_json(pred)

    evaluator = TrackingEvaluator(matcher=active_matcher, metrics=metric_instances)
    results = evaluator.evaluate(gt_seq, pred_seq)

    print_results_table(results)

    if output is not None:
        _save_output(results, output)

    for path_str, saver in cfg_output_paths:
        if path_str and output is None:
            saver(results, Path(path_str))
            typer.echo(f"Results saved to {path_str}")


@app.command(name="evaluate-batch")
def evaluate_batch(
    gt_dir: Annotated[
        Path, typer.Option("--gt-dir", help="Directory containing ground-truth JSON files.")
    ],
    pred_dir: Annotated[
        Path, typer.Option("--pred-dir", help="Directory containing prediction JSON files.")
    ],
    matcher: Annotated[
        str, typer.Option("--matcher", help=f"Matcher type. One of: {_MATCHERS}.")
    ] = "box2d-iou",
    threshold: Annotated[
        float, typer.Option("--threshold", help="IoU threshold.")
    ] = 0.5,
    max_distance: Annotated[
        float, typer.Option("--max-distance", help="Max center distance.")
    ] = 0.5,
    metrics: Annotated[
        list[str] | None, typer.Option("--metrics", help="Metrics to compute.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Path to save batch report (.json/.csv).")
    ] = None,
    hota_curves: Annotated[
        bool,
        typer.Option("--hota-curves/--no-hota-curves"),
    ] = False,
) -> None:
    _validate_matcher(matcher)
    selected_metrics = metrics or list(_METRIC_MAP.keys())
    _validate_metrics(selected_metrics)

    gt_files = {p.stem: p for p in sorted(gt_dir.glob("*.json"))}
    pred_files = {p.stem: p for p in sorted(pred_dir.glob("*.json"))}

    common = sorted(set(gt_files) & set(pred_files))
    gt_only = sorted(set(gt_files) - set(pred_files))
    pred_only = sorted(set(pred_files) - set(gt_files))

    if gt_only:
        typer.echo(f"Warning: GT files without matching pred: {gt_only}", err=True)
    if pred_only:
        typer.echo(f"Warning: pred files without matching GT: {pred_only}", err=True)

    if not common:
        typer.echo("No matching sequence pairs found.", err=True)
        raise typer.Exit(code=1)

    sequence_pairs = [
        SequencePair(
            name=stem,
            gt=Sequence.from_json(gt_files[stem]),
            pred=Sequence.from_json(pred_files[stem]),
        )
        for stem in common
    ]

    metric_instances = _build_metrics(selected_metrics, hota_curves)
    active_matcher = _build_matcher(matcher, threshold, max_distance)

    evaluator = TrackingEvaluator(matcher=active_matcher, metrics=metric_instances)
    batch_evaluator = BatchEvaluator(evaluator)
    results = batch_evaluator.evaluate(sequence_pairs)

    typer.echo(f"\nEvaluated {len(common)} sequences: {common}")

    typer.echo("\n=== Per-sequence results ===")
    for name, scores in results["sequences"].items():
        typer.echo(f"\n  {name}:")
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                typer.echo(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    typer.echo("\n=== Average ===")
    print_results_table(results["average"])

    typer.echo("\n=== Global ===")
    print_results_table(results["global"])

    if output is not None:
        _save_output(results, output)
