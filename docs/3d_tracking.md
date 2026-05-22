# 3D Tracking Support

## 3D Box Data Model

```python
from tracking_metrics.data.boxes3d import Box3D

box = Box3D(
    center=(x, y, z),
    size=(dx, dy, dz),   # width, height, length
    yaw=1.57,            # optional — stored but not used by axis-aligned IoU
)

box.volume()       # dx * dy * dz
box.min_corner()   # center - size/2  (numpy array)
box.max_corner()   # center + size/2  (numpy array)
box.is_valid()     # True if all size components >= 0
```

**Note:** `yaw` is stored for future oriented IoU support but is **not used** in v0.3 axis-aligned IoU or center-distance computations.

## 3D Input JSON Schema

```json
{
  "track_id": 1,
  "class_id": "car",
  "bbox3d": {
    "center": [x, y, z],
    "size": [dx, dy, dz],
    "yaw": 1.57
  }
}
```

`yaw` is optional. A detection can contain `bbox2d`, `mask`, and `bbox3d` simultaneously.

## Axis-Aligned 3D IoU

Computed as:

```
IoU = intersection_volume / union_volume
```

Corners are derived from `center ± size/2`. Yaw is ignored. Mismatched or invalid boxes return `0.0`.

```python
from tracking_metrics.geometry.box3d_ops import box3d_iou_axis_aligned

iou = box3d_iou_axis_aligned(box_a, box_b)
```

**Recommended threshold:** `0.25` (common in 3D object detection benchmarks).

## Center Distance Matching

Similarity is derived from Euclidean center distance:

```
similarity = max(0, 1 - distance / max_distance)
```

A match is accepted when `distance <= max_distance`.

```python
from tracking_metrics.matching.center_distance_matcher import CenterDistanceMatcher

matcher = CenterDistanceMatcher(max_distance=0.5)
```

**Recommended max_distance:** `0.5 m` for pedestrians; `1.0–2.0 m` for vehicles.

`MOTP` with this matcher reflects the normalized similarity (0–1), not raw distance. Use `MeanCenterDist3D` for the raw average distance in meters.

## 3D IoU Matcher

```python
from tracking_metrics.matching.box3d_iou_matcher import Box3DIoUMatcher

matcher = Box3DIoUMatcher(threshold=0.25)
```

## Class-Aware Mode

Both matchers support `class_aware=True`. When enabled, detections with different non-null `class_id` values cannot be matched.

## Metrics

| Metric | Description |
|--------|-------------|
| `MOTP` | Average similarity of matched pairs (normalized; depends on matcher) |
| `MeanBox3DIoU` | Average axis-aligned 3D IoU over matched pairs with 3D boxes |
| `MeanCenterDist3D` | Average Euclidean center distance (meters) over matched pairs with 3D boxes |

`MeanBox3DIoU` and `MeanCenterDist3D` re-compute from the detections directly, so they can be used with any matcher.

## CLI

### Box3D IoU

```bash
track-metrics evaluate \
  --gt gt_3d.json \
  --pred pred_3d.json \
  --matcher box3d-iou \
  --threshold 0.25 \
  --metrics counts --metrics mota --metrics motp --metrics idsw --metrics idf1 \
  --metrics mean-box3d-iou --metrics mean-center-dist-3d
```

### Center Distance

```bash
track-metrics evaluate \
  --gt gt_3d.json \
  --pred pred_3d.json \
  --matcher center-distance \
  --max-distance 0.5 \
  --metrics counts --metrics mota --metrics motp --metrics idsw --metrics idf1 \
  --metrics mean-center-dist-3d
```
