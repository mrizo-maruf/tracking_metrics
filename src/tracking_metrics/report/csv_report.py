from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def save_csv_report(results: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            if isinstance(v, (int, float)):
                writer.writerow([k, v])
