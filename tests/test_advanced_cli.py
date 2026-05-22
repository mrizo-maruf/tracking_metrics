from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tracking_metrics.cli.evaluate import app

runner = CliRunner()

_GT = {
    "sequence_name": "test",
    "frames": [
        {"frame_id": 0, "detections": [{"track_id": "1", "bbox2d": [0, 0, 2, 2]}]},
        {"frame_id": 1, "detections": [{"track_id": "1", "bbox2d": [0, 0, 2, 2]}]},
    ],
}

_PRED = {
    "sequence_name": "test",
    "frames": [
        {"frame_id": 0, "detections": [{"track_id": "10", "bbox2d": [0, 0, 2, 2]}]},
        {"frame_id": 1, "detections": [{"track_id": "10", "bbox2d": [0, 0, 2, 2]}]},
    ],
}


@pytest.fixture()
def gt_pred_files(tmp_path):
    gt_file = tmp_path / "gt.json"
    pred_file = tmp_path / "pred.json"
    gt_file.write_text(json.dumps(_GT))
    pred_file.write_text(json.dumps(_PRED))
    return gt_file, pred_file


def test_cli_hota_metric(gt_pred_files):
    gt, pred = gt_pred_files
    result = runner.invoke(app, ["evaluate", "--gt", str(gt), "--pred", str(pred), "--metrics", "hota"])
    assert result.exit_code == 0
    assert "HOTA" in result.output


def test_cli_frag_metric(gt_pred_files):
    gt, pred = gt_pred_files
    result = runner.invoke(app, ["evaluate", "--gt", str(gt), "--pred", str(pred), "--metrics", "frag"])
    assert result.exit_code == 0
    assert "Frag" in result.output


def test_cli_track_coverage_metric(gt_pred_files):
    gt, pred = gt_pred_files
    result = runner.invoke(
        app,
        ["evaluate", "--gt", str(gt), "--pred", str(pred), "--metrics", "track-coverage"],
    )
    assert result.exit_code == 0
    assert "MT" in result.output


def test_cli_t_sr_metric(gt_pred_files):
    gt, pred = gt_pred_files
    result = runner.invoke(app, ["evaluate", "--gt", str(gt), "--pred", str(pred), "--metrics", "t-sr"])
    assert result.exit_code == 0
    assert "T-SR" in result.output


def test_cli_id_cons_metric(gt_pred_files):
    gt, pred = gt_pred_files
    result = runner.invoke(
        app,
        ["evaluate", "--gt", str(gt), "--pred", str(pred), "--metrics", "id-cons"],
    )
    assert result.exit_code == 0
    assert "IDCons" in result.output


def test_cli_output_csv(gt_pred_files, tmp_path):
    gt, pred = gt_pred_files
    out = tmp_path / "results.csv"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--gt", str(gt),
            "--pred", str(pred),
            "--metrics", "counts",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.exists()
    content = out.read_text()
    assert "metric" in content
    assert "TP" in content


def test_cli_output_latex(gt_pred_files, tmp_path):
    gt, pred = gt_pred_files
    out = tmp_path / "results.tex"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--gt", str(gt),
            "--pred", str(pred),
            "--metrics", "counts",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.exists()
    content = out.read_text()
    assert "\\begin{tabular}" in content


def test_cli_hota_curves_flag(gt_pred_files, tmp_path):
    gt, pred = gt_pred_files
    out = tmp_path / "results.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--gt", str(gt),
            "--pred", str(pred),
            "--metrics", "hota",
            "--hota-curves",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert "HOTA_curve" in data
    assert len(data["HOTA_curve"]) == 19


def test_cli_all_v4_metrics_together(gt_pred_files):
    gt, pred = gt_pred_files
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--gt", str(gt),
            "--pred", str(pred),
            "--metrics", "hota",
            "--metrics", "frag",
            "--metrics", "track-coverage",
            "--metrics", "t-sr",
            "--metrics", "id-cons",
        ],
    )
    assert result.exit_code == 0
