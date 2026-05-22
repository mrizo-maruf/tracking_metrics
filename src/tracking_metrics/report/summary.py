from __future__ import annotations

from typing import Any


def format_summary(results: dict[str, Any], precision: int = 4) -> str:
    lines = []
    for k, v in results.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.{precision}f}")
        elif isinstance(v, int):
            lines.append(f"{k}: {v}")
    return "\n".join(lines)
