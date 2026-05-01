from __future__ import annotations

from pathlib import Path


def _resolve_path(*args: str, validate: bool = True) -> Path:
    from pathlib import Path

    full_path = Path(*args).resolve()

    if validate and not full_path.exists():
        raise FileNotFoundError(f"Path does not exist: {full_path}")

    return full_path
