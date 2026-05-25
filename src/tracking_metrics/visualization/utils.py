from __future__ import annotations

from pathlib import Path
from typing import Any

_INSTALL_MSG = (
    "Visualization requires matplotlib.\n"
    "Install with:\n"
    "    pip install -e '.[viz]'\n"
    "or:\n"
    "    pip install matplotlib"
)


def require_matplotlib() -> tuple[Any, Any]:
    """Return (matplotlib, pyplot). Raise ImportError with install hint if absent."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        return matplotlib, plt
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def save_figure(fig: Any, path: str | Path | None) -> None:
    """Save *fig* to *path* if provided."""
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
