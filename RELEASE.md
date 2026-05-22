# Release / Versioning

This repository is versioned through Git tags. It is not published to PyPI.

## Tagging a release

```bash
# Update version in two places first:
#   src/tracking_metrics/__init__.py  →  __version__ = "X.Y.Z"
#   pyproject.toml                    →  version = "X.Y.Z"

# Update CHANGELOG.md, commit, then tag:
git add src/tracking_metrics/__init__.py pyproject.toml CHANGELOG.md
git commit -m "chore: release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

## Installing a specific tag

```bash
pip install git+https://github.com/<ORG_OR_USER>/tracking-metrics.git@vX.Y.Z
```

## Checklist before tagging

1. `pytest` — all tests pass
2. `ruff check .` — no lint errors
3. `mypy src` — no type errors
4. Version string updated in `__init__.py` and `pyproject.toml`
5. `CHANGELOG.md` updated with a new section
6. CI is green on the commit being tagged

## Testing the install

```bash
python -m venv /tmp/test_env
source /tmp/test_env/bin/activate
pip install git+https://github.com/<ORG_OR_USER>/tracking-metrics.git@vX.Y.Z
python -c "import tracking_metrics; print(tracking_metrics.__version__)"
track-metrics --help
deactivate
```
