from __future__ import annotations

import textwrap

import pytest

from tracking_metrics.config.loader import load_config
from tracking_metrics.config.schema import EvalConfig


def _write_yaml(tmp_path, content: str):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_load_minimal_config(tmp_path):
    p = _write_yaml(tmp_path, """
        matcher:
          type: box2d-iou
          threshold: 0.5
    """)
    cfg = load_config(p)
    assert isinstance(cfg, EvalConfig)
    assert cfg.matcher.type == "box2d-iou"
    assert cfg.matcher.threshold == pytest.approx(0.5)


def test_load_full_config(tmp_path):
    p = _write_yaml(tmp_path, """
        matcher:
          type: mask-iou
          threshold: 0.4
          class_aware: true

        metrics:
          - counts
          - mota
          - hota

        hota_curves: true

        output:
          json: results.json
          csv: results.csv
          latex: results.tex
    """)
    cfg = load_config(p)
    assert cfg.matcher.type == "mask-iou"
    assert cfg.matcher.threshold == pytest.approx(0.4)
    assert cfg.matcher.class_aware is True
    assert cfg.metrics == ["counts", "mota", "hota"]
    assert cfg.hota_curves is True
    assert cfg.output.json == "results.json"
    assert cfg.output.csv == "results.csv"
    assert cfg.output.latex == "results.tex"


def test_load_empty_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("")
    cfg = load_config(p)
    assert cfg.matcher.type == "box2d-iou"
    assert cfg.metrics == []


def test_load_center_distance_matcher(tmp_path):
    p = _write_yaml(tmp_path, """
        matcher:
          type: center-distance
          max_distance: 2.0
    """)
    cfg = load_config(p)
    assert cfg.matcher.type == "center-distance"
    assert cfg.matcher.max_distance == pytest.approx(2.0)


def test_config_defaults():
    cfg = EvalConfig()
    assert cfg.matcher.type == "box2d-iou"
    assert cfg.metrics == []
    assert cfg.output.json is None
    assert cfg.hota_curves is False
