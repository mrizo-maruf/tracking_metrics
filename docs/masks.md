# Mask Tracking Support

## Binary Masks

A binary mask is a 2D `numpy` array of shape `H × W` with dtype `bool` or `uint8`.

```python
import numpy as np
from tracking_metrics.data.masks import Mask2D

mask = Mask2D(data=np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool))
print(mask.area())      # 4
print(mask.to_binary()) # HxW bool array
```

## RLE Masks

Run-length encoded masks follow the COCO RLE format and require `pycocotools`:

```bash
pip install "tracking-metrics[masks]"
```

```python
from tracking_metrics.data.masks import Mask2D

rle = {"size": [480, 640], "counts": "<encoded>"}
mask = Mask2D(rle=rle, size=(480, 640))
binary = mask.to_binary()  # requires pycocotools
```

Attempting RLE decoding without `pycocotools` raises `ImportError` with an install hint.

## JSON Format

### Dense binary mask

```json
{
  "track_id": 1,
  "mask": {
    "type": "binary",
    "size": [4, 4],
    "data": [
      [0, 0, 0, 0],
      [0, 1, 1, 0],
      [0, 1, 1, 0],
      [0, 0, 0, 0]
    ]
  }
}
```

### RLE mask

```json
{
  "track_id": 1,
  "mask": {
    "type": "rle",
    "size": [480, 640],
    "counts": "encoded_counts_here"
  }
}
```

A detection may contain both `bbox2d` and `mask`. Neither is required.

## Mask IoU

Mask IoU is computed as:

```
IoU = |A ∩ B| / |A ∪ B|
```

Masks must have identical shape. Mismatched shapes raise `ValueError`.

```python
from tracking_metrics.geometry.mask_ops import mask_iou

iou = mask_iou(mask_a, mask_b)
```

## Mask Matcher

```python
from tracking_metrics.matching.mask_iou_matcher import MaskIoUMatcher

matcher = MaskIoUMatcher(threshold=0.5)
result = matcher.match(gt_detections, pred_detections)
```

- Detections without a mask have similarity `0.0` against all others.
- `class_aware=True` prevents matching detections with different non-null class IDs.

## T-mIoU (Temporal mean IoU)

T-mIoU is the average mask IoU over all matched GT–prediction pairs that have masks:

```
T-mIoU = (1 / |matches with masks|) × Σ mask_IoU(gt_mask, pred_mask)
```

If no matched pair has masks, T-mIoU is `0.0`.

T-mIoU is matcher-agnostic: it recomputes mask IoU directly from the detections, so it can be used even when the Box2D matcher was used (the result will be `0.0` if no masks are present).

```python
from tracking_metrics.metrics.temporal_iou import TemporalIoUMetric

metric = TemporalIoUMetric()
scores = metric.compute(evaluation_result)
# {"T-mIoU": 0.94}
```

## MOTP with Mask Matching

When using `MaskIoUMatcher`, `MOTP` is the average matched mask IoU (same as `T-mIoU` in this case). With `Box2DIoUMatcher`, `MOTP` is the average matched box IoU. `MOTP` always measures the average similarity of matched pairs, regardless of the matcher.
