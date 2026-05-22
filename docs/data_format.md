# Data Format

All input and output uses a simple JSON format.

## Sequence JSON

```json
{
  "sequence_name": "scene_001",
  "frames": [
    {
      "frame_id": 0,
      "detections": [
        {
          "track_id": "1",
          "class_id": "person",
          "score": 0.98,
          "bbox2d": [x1, y1, x2, y2]
        }
      ]
    }
  ]
}
```

- `track_id` — integer or string; stored as `str` internally.
- `class_id`, `score` — optional.
- `bbox2d` — `[x1, y1, x2, y2]` pixel coordinates.
- A detection may contain `bbox2d`, `mask`, and `bbox3d` simultaneously.

## 2D Mask

```json
{
  "track_id": "1",
  "mask": {
    "type": "binary",
    "size": [H, W],
    "data": [[0, 1, 1, 0], ...]
  }
}
```

Or COCO RLE (requires `pycocotools`):

```json
{
  "track_id": "1",
  "mask": {
    "type": "rle",
    "size": [480, 640],
    "counts": "encoded_counts_here"
  }
}
```

## 3D Box

```json
{
  "track_id": "1",
  "bbox3d": {
    "center": [x, y, z],
    "size": [dx, dy, dz],
    "yaw": 1.57
  }
}
```

`yaw` is optional. Axis-aligned IoU ignores it.

## Python

```python
from tracking_metrics import Sequence

seq = Sequence.from_json("gt.json")
seq.to_json("output.json")
```

## Extra fields

Unknown detection fields are preserved in `detection.attributes`:

```json
{"track_id": "1", "bbox2d": [0, 0, 10, 10], "my_field": 42}
```

```python
det.attributes["my_field"]  # 42
```
