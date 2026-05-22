"""Minimal example: evaluate 2D box tracking with the Python API."""

from tracking_metrics import (
    IDF1,
    MOTA,
    MOTP,
    Box2D,
    Box2DIoUMatcher,
    Detection,
    DetectionCounts,
    Fragmentations,
    Frame,
    IDSwitches,
    Sequence,
    TrackCoverage,
    TrackingEvaluator,
    TrackSurvivalRate,
)
from tracking_metrics.report import print_results_table

# Two frames, two tracks crossing paths
gt = Sequence(
    name="scene_001",
    frames=[
        Frame(frame_id=0, detections=[
            Detection(frame_id=0, track_id="1", bbox2d=Box2D(0, 0, 50, 50)),
            Detection(frame_id=0, track_id="2", bbox2d=Box2D(100, 100, 150, 150)),
        ]),
        Frame(frame_id=1, detections=[
            Detection(frame_id=1, track_id="1", bbox2d=Box2D(5, 5, 55, 55)),
            Detection(frame_id=1, track_id="2", bbox2d=Box2D(105, 105, 155, 155)),
        ]),
        Frame(frame_id=2, detections=[
            Detection(frame_id=2, track_id="1", bbox2d=Box2D(10, 10, 60, 60)),
        ]),
    ],
)

pred = Sequence(
    name="scene_001",
    frames=[
        Frame(frame_id=0, detections=[
            Detection(frame_id=0, track_id="A", bbox2d=Box2D(1, 1, 51, 51)),
            Detection(frame_id=0, track_id="B", bbox2d=Box2D(101, 101, 151, 151)),
        ]),
        Frame(frame_id=1, detections=[
            Detection(frame_id=1, track_id="A", bbox2d=Box2D(6, 6, 56, 56)),
            Detection(frame_id=1, track_id="C", bbox2d=Box2D(106, 106, 156, 156)),  # ID switch
        ]),
        Frame(frame_id=2, detections=[
            Detection(frame_id=2, track_id="A", bbox2d=Box2D(11, 11, 61, 61)),
        ]),
    ],
)

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
        TrackSurvivalRate(),
    ],
)

results = evaluator.evaluate(gt, pred)
print_results_table(results)
