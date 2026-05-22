from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

_console = Console()


def print_results_table(results: dict[str, Any]) -> None:
    table = Table(title="Tracking Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    for key, value in results.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    _console.print(table)
