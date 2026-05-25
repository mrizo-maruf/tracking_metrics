from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json_report(results: dict[str, Any], path: str | Path) -> None:
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)


def load_json_report(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)
