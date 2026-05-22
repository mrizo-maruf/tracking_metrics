# Matching

Each frame's GT and predicted detections are matched using the Hungarian algorithm.
A `Matcher` computes a similarity matrix and returns matched pairs, unmatched GT, and
unmatched predictions.

## Available matchers

### `Box2DIoUMatcher`

Matches by 2D bounding-box IoU. `threshold` is the minimum IoU to accept a match.

```python
from tracking_metrics import Box2DIoUMatcher
matcher = Box2DIoUMatcher(threshold=0.5)
```

### `MaskIoUMatcher`

Matches by binary-mask IoU. Requires detections with a `mask` field.

```python
from tracking_metrics import MaskIoUMatcher
matcher = MaskIoUMatcher(threshold=0.5)
```

### `Box3DIoUMatcher`

Matches by axis-aligned 3D bounding-box IoU. Requires detections with a `bbox3d` field.

```python
from tracking_metrics import Box3DIoUMatcher
matcher = Box3DIoUMatcher(threshold=0.25)
```

Recommended threshold: 0.25 for outdoor, 0.5 for indoor scenes. See [3d_tracking.md](3d_tracking.md).

### `CenterDistanceMatcher`

Matches by Euclidean center distance. Accepts a pair if `distance ≤ max_distance`.

```python
from tracking_metrics import CenterDistanceMatcher
matcher = CenterDistanceMatcher(max_distance=0.5)
```

The internal similarity is `max(0, 1 − dist / max_distance)`, used for MOTP and HOTA.

## Class-aware matching

All matchers accept `class_aware=True` to restrict matching to same-class detections.

```python
matcher = Box2DIoUMatcher(threshold=0.5, class_aware=True)
```

## MOTP interpretation

MOTP is the mean similarity of matched pairs. Its range and interpretation depends on the matcher:

| Matcher | MOTP range | Meaning |
|---------|-----------|---------|
| `box2d-iou` | 0–1 | Average IoU of matched boxes |
| `mask-iou` | 0–1 | Average mask IoU |
| `box3d-iou` | 0–1 | Average 3D IoU |
| `center-distance` | 0–1 | Average normalized proximity |
