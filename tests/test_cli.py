import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tracking_metrics.cli.evaluate import app

runner = CliRunner()


GT_DATA = {
    "sequence_name": "test",
    "frames": [
        {"frame_id": 0, "detections": [{"track_id": "1", "bbox2d": [0, 0, 100, 100]}]},
        {"frame_id": 1, "detections": [{"track_id": "1", "bbox2d": [0, 0, 100, 100]}]},
    ],
}

PRED_DATA = {
    "sequence_name": "test",
    "frames": [
        {"frame_id": 0, "detections": [{"track_id": "10", "bbox2d": [0, 0, 100, 100]}]},
        {"frame_id": 1, "detections": [{"track_id": "10", "bbox2d": [0, 0, 100, 100]}]},
    ],
}


def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data))


def test_cli_basic_run(tmp_path: Path):
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    _write_json(GT_DATA, gt_path)
    _write_json(PRED_DATA, pred_path)

    result = runner.invoke(app, ["evaluate", "--gt", str(gt_path), "--pred", str(pred_path)])
    assert result.exit_code == 0


def test_cli_output_file(tmp_path: Path):
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    out_path = tmp_path / "results.json"
    _write_json(GT_DATA, gt_path)
    _write_json(PRED_DATA, pred_path)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--gt", str(gt_path),
            "--pred", str(pred_path),
            "--output", str(out_path),
        ],
    )
    assert result.exit_code == 0
    assert out_path.exists()
    scores = json.loads(out_path.read_text())
    assert scores["TP"] == 2
    assert scores["MOTA"] == pytest.approx(1.0)
    assert scores["IDF1"] == pytest.approx(1.0)


def test_cli_metric_selection(tmp_path: Path):
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    out_path = tmp_path / "results.json"
    _write_json(GT_DATA, gt_path)
    _write_json(PRED_DATA, pred_path)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--gt", str(gt_path),
            "--pred", str(pred_path),
            "--metrics", "mota",
            "--metrics", "idsw",
            "--output", str(out_path),
        ],
    )
    assert result.exit_code == 0
    scores = json.loads(out_path.read_text())
    assert "MOTA" in scores
    assert "IDSW" in scores
    assert "TP" not in scores


def test_cli_unknown_matcher(tmp_path: Path):
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    _write_json(GT_DATA, gt_path)
    _write_json(PRED_DATA, pred_path)

    result = runner.invoke(
        app,
        ["evaluate", "--gt", str(gt_path), "--pred", str(pred_path), "--matcher", "bad"],
    )
    assert result.exit_code != 0


def test_cli_unknown_metric(tmp_path: Path):
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    _write_json(GT_DATA, gt_path)
    _write_json(PRED_DATA, pred_path)

    result = runner.invoke(
        app,
        ["evaluate", "--gt", str(gt_path), "--pred", str(pred_path), "--metrics", "nosuchmetric"],
    )
    assert result.exit_code != 0
