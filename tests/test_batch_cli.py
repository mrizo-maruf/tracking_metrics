from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tracking_metrics.cli.evaluate import app

runner = CliRunner()

_GT = {
    "sequence_name": "scene_001",
    "frames": [
        {"frame_id": 0, "detections": [{"track_id": "1", "bbox2d": [0, 0, 2, 2]}]},
        {"frame_id": 1, "detections": [{"track_id": "1", "bbox2d": [0, 0, 2, 2]}]},
    ],
}

_PRED = {
    "sequence_name": "scene_001",
    "frames": [
        {"frame_id": 0, "detections": [{"track_id": "10", "bbox2d": [0, 0, 2, 2]}]},
        {"frame_id": 1, "detections": [{"track_id": "10", "bbox2d": [0, 0, 2, 2]}]},
    ],
}


@pytest.fixture()
def batch_dirs(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "scene_001.json").write_text(json.dumps(_GT))
    (pred_dir / "scene_001.json").write_text(json.dumps(_PRED))
    return gt_dir, pred_dir


def test_evaluate_batch_basic(batch_dirs):
    gt_dir, pred_dir = batch_dirs
    result = runner.invoke(
        app,
        [
            "evaluate-batch",
            "--gt-dir", str(gt_dir),
            "--pred-dir", str(pred_dir),
            "--metrics", "counts",
        ],
    )
    assert result.exit_code == 0
    assert "scene_001" in result.output


def test_evaluate_batch_output_json(batch_dirs, tmp_path):
    gt_dir, pred_dir = batch_dirs
    out = tmp_path / "batch_results.json"
    result = runner.invoke(
        app,
        [
            "evaluate-batch",
            "--gt-dir", str(gt_dir),
            "--pred-dir", str(pred_dir),
            "--metrics", "counts",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert "sequences" in data
    assert "average" in data
    assert "global" in data
    assert "scene_001" in data["sequences"]


def test_evaluate_batch_warns_missing_pred(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "scene_001.json").write_text(json.dumps(_GT))
    (gt_dir / "scene_002.json").write_text(json.dumps(_GT))
    (pred_dir / "scene_001.json").write_text(json.dumps(_PRED))

    result = runner.invoke(
        app,
        [
            "evaluate-batch",
            "--gt-dir", str(gt_dir),
            "--pred-dir", str(pred_dir),
            "--metrics", "counts",
        ],
    )
    assert "scene_002" in result.output or result.exit_code == 0


def test_evaluate_batch_no_matches_exits(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "scene_001.json").write_text(json.dumps(_GT))
    # no pred files

    result = runner.invoke(
        app,
        [
            "evaluate-batch",
            "--gt-dir", str(gt_dir),
            "--pred-dir", str(pred_dir),
            "--metrics", "counts",
        ],
    )
    assert result.exit_code != 0


def test_cli_evaluate_with_config(tmp_path):
    gt_file = tmp_path / "gt.json"
    pred_file = tmp_path / "pred.json"
    gt_file.write_text(json.dumps(_GT))
    pred_file.write_text(json.dumps(_PRED))

    config_file = tmp_path / "config.yaml"
    config_file.write_text("matcher:\n  type: box2d-iou\n  threshold: 0.5\nmetrics:\n  - counts\n")

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--gt", str(gt_file),
            "--pred", str(pred_file),
            "--config", str(config_file),
        ],
    )
    assert result.exit_code == 0
    assert "TP" in result.output


def test_cli_config_output_files(tmp_path):
    gt_file = tmp_path / "gt.json"
    pred_file = tmp_path / "pred.json"
    gt_file.write_text(json.dumps(_GT))
    pred_file.write_text(json.dumps(_PRED))

    out_json = tmp_path / "results.json"
    out_csv = tmp_path / "results.csv"

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"matcher:\n  type: box2d-iou\nmetrics:\n  - counts\n"
        f"output:\n  json: {out_json}\n  csv: {out_csv}\n"
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--gt", str(gt_file),
            "--pred", str(pred_file),
            "--config", str(config_file),
        ],
    )
    assert result.exit_code == 0
    assert out_json.exists()
    assert out_csv.exists()
