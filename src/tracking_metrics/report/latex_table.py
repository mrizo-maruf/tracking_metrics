from __future__ import annotations

from pathlib import Path
from typing import Any


def results_to_latex(results: dict[str, Any], caption: str = "", label: str = "") -> str:
    rows = [(k, v) for k, v in results.items() if isinstance(v, (int, float))]

    lines = ["\\begin{table}[h]", "\\centering", "\\begin{tabular}{lr}", "\\hline"]
    lines.append("Metric & Value \\\\")
    lines.append("\\hline")

    for k, v in rows:
        if isinstance(v, float):
            lines.append(f"{k} & {v:.4f} \\\\")
        else:
            lines.append(f"{k} & {v} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")

    lines.append("\\end{table}")
    return "\n".join(lines)


def save_latex_report(
    results: dict[str, Any],
    path: str | Path,
    caption: str = "",
    label: str = "",
) -> None:
    Path(path).write_text(results_to_latex(results, caption=caption, label=label))
