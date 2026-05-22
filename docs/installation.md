# Installation

This package is not on PyPI. Install it from GitHub or from a local clone.

> **Note:** install name and import name differ.
> Install source: `tracking-metrics` · Python import: `tracking_metrics`

## Requirements

- Python ≥ 3.10
- numpy, scipy, pydantic, typer, rich, pyyaml (installed automatically)

## Option 1 — Editable development install

```bash
git clone git@github.com:<ORG_OR_USER>/tracking-metrics.git
cd tracking-metrics
pip install -e .
```

## Option 2 — Install directly from GitHub

```bash
pip install git+https://github.com/<ORG_OR_USER>/tracking-metrics.git
```

Install a specific tag:

```bash
pip install git+https://github.com/<ORG_OR_USER>/tracking-metrics.git@v0.5.0
```

## Option 3 — Add to `requirements.txt`

```
git+https://github.com/<ORG_OR_USER>/tracking-metrics.git
```

## With COCO RLE mask support

```bash
pip install -e ".[masks]"
```

Requires `pycocotools` (Linux/macOS).

## Development dependencies

```bash
pip install -e ".[dev]"
```

## With documentation tools

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Using from another project

```
~/work/
├── your-project/
└── tracking-metrics/
```

```bash
conda activate your-project-env
cd ~/work/tracking-metrics
pip install -e .
```
