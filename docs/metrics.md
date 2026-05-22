# Metrics

## Detection counts

| Key | Description |
|-----|-------------|
| `TP` | True positives — matched GT–pred pairs per frame |
| `FP` | False positives — unmatched predictions |
| `FN` | False negatives — unmatched GT detections |
| `GT` | Total GT detections |
| `Pred` | Total predicted detections |

```python
from tracking_metrics import DetectionCounts
```

## IDSW — ID Switches

Number of times a GT track's matched predicted ID changes.

```python
from tracking_metrics import IDSwitches
```

## MOTA — Multi-Object Tracking Accuracy

`MOTA = 1 − (FN + FP + IDSW) / GT`

Range: (−∞, 1]. Value of 1 means perfect tracking. Can be negative.

```python
from tracking_metrics import MOTA
```

## MOTP — Multi-Object Tracking Precision

Mean similarity of matched pairs. Range 0–1 (matcher-dependent scale).

```python
from tracking_metrics import MOTP
```

## IDF1 / IDP / IDR

Identity-aware F1/Precision/Recall via global identity alignment.

| Key | Description |
|-----|-------------|
| `IDF1` | Harmonic mean of IDP and IDR |
| `IDP` | ID Precision |
| `IDR` | ID Recall |
| `IDTP` | ID True Positives |
| `IDFP` | ID False Positives |
| `IDFN` | ID False Negatives |

```python
from tracking_metrics import IDF1
```

## T-mIoU / T-Dice — Temporal mask metrics

Mean mask IoU / Dice score over matched pairs (mask tracking only).

```python
from tracking_metrics import TemporalIoU
from tracking_metrics.metrics.temporal_iou import TemporalDiceMetric
```

## MeanBox3DIoU / MeanCenterDist3D — 3D localization metrics

Axis-aligned 3D IoU and Euclidean center distance over matched pairs.

```python
from tracking_metrics import MeanBox3DIoU, MeanCenterDistance3D
```

## Advanced metrics

See [advanced_metrics.md](advanced_metrics.md) for HOTA, Fragmentations, TrackCoverage,
TrackSurvivalRate, and IDConsistency.
