# Changelog

## v0.5.0

- **Public API**: all major classes importable from `tracking_metrics` with short metric aliases (`MOTA`, `IDF1`, `HOTA`, etc.)
- **Batch evaluation**: `BatchEvaluator` + `SequencePair` for multi-sequence evaluation with `sequences`, `average`, and `global` aggregation
- **Config files**: YAML-driven evaluation via `--config`, parsed by `tracking_metrics.config`
- **Adapter interface**: `SequenceAdapter` protocol + `save_sequence_json` helper in `tracking_metrics.adapters`
- **CLI**: new `evaluate-batch` command; `evaluate` command supports `--config` and saves to `.csv`/`.tex` from config `output.*` keys
- **Packaging**: added `pyyaml` dependency, `docs` optional group, project classifiers, authors field
- **CI**: GitHub Actions workflow (lint + typecheck + tests on Python 3.10/3.11/3.12)
- **Docs**: full docs site structure (`docs/`)
- **Examples**: `toy_2d_boxes.py`, `batch_eval.py`, `custom_adapter.py`, `examples/configs/`, `examples/data/`
- **Version**: bumped to 0.5.0

## v0.4.0

- **HOTA**: threshold-sweep implementation (DetA, AssA, LocA, OWTA, DetRe/Pr, AssRe/Pr, optional per-threshold curves)
- **Fragmentations**: count of GT track breaks
- **TrackCoverage**: MT/PT/ML classification with configurable thresholds
- **TrackSurvivalRate**: fraction of GT tracks with at least one match
- **IDConsistency**: average majority-ID coverage per GT track
- **FrameResult**: extended with `gt_detections`, `pred_detections`, `similarity_matrix`, `distance_matrix`
- **HOTAData**: internal adapter for HOTA computation from `EvaluationResult`
- **Reports**: CSV (`save_csv_report`), LaTeX (`save_latex_report`), summary (`format_summary`)
- **CLI**: added `hota`, `frag`, `track-coverage`, `t-sr`, `id-cons` metric names; `--hota-curves` flag; `.csv`/`.tex` output routing
- **Version**: bumped to 0.4.0

## v0.3.0

- **Box3D**: 3D bounding-box data model with `center`, `size`, `yaw`
- **Axis-aligned 3D IoU**: `box3d_iou_axis_aligned`, `box3d_iou_matrix_axis_aligned`
- **Box3DIoUMatcher**: Hungarian matching on 3D IoU
- **CenterDistanceMatcher**: Hungarian matching on Euclidean center distance, `--max-distance` option
- **MeanBox3DIoU**: average axis-aligned 3D IoU over matched pairs
- **MeanCenterDistance3D**: average Euclidean center distance in world units
- **MatchResult**: extended with `distance_matrix`
- **JSON format**: `bbox3d` field support
- **CLI**: `box3d-iou` and `center-distance` matchers; `mean-box3d-iou`, `mean-center-dist-3d` metrics
- **Version**: bumped to 0.3.0

## v0.2.0

- **Mask2D**: binary array and COCO RLE mask support (optional `pycocotools` dependency)
- **Mask geometry**: `mask_iou`, `mask_dice`, `mask_iou_matrix`
- **MaskIoUMatcher**: Hungarian matching on mask IoU
- **TemporalIoUMetric**: T-mIoU (mean mask IoU over matched pairs)
- **TemporalDiceMetric**: T-Dice score
- **JSON format**: `mask` field support (binary and RLE)
- **CLI**: `mask-iou` matcher; `t-miou`, `t-dice` metrics
- **Version**: bumped to 0.2.0

## v0.1.0

- Initial release
- **Data model**: `Box2D`, `Detection`, `Frame`, `Sequence` with JSON I/O
- **Box2D IoU matching**: `Box2DIoUMatcher` with Hungarian assignment
- **Metrics**: TP/FP/FN/GT/Pred, IDSW, MOTA, MOTP, IDF1/IDP/IDR
- **Identity tracking**: `IdentityHistory` for ID switch detection
- **CLI**: `track-metrics evaluate` command
- **Reports**: terminal table (Rich), JSON output
