"""Project-root resolution without machine-specific absolute paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT_MARKERS = ("pyproject.toml", "README.md")


class ProjectRootNotFoundError(RuntimeError):
    """Raised when the repository root cannot be located from ``start``."""


def _is_project_root(path: Path) -> bool:
    return all((path / marker).is_file() for marker in _PROJECT_ROOT_MARKERS)


@lru_cache(maxsize=8)
def get_project_root(start: str | Path | None = None) -> Path:
    """Return the LeakageLens repository root.

    Walks parents from ``start`` (or this file) until ``pyproject.toml`` and
    ``README.md`` are found. Callers must not pass or store
    user-specific absolute paths in committed configs.
    """

    current = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if _is_project_root(candidate):
            return candidate

    raise ProjectRootNotFoundError(
        "Could not locate the LeakageLens project root. "
        "Expected to find pyproject.toml and README.md."
    )


def get_configs_dir(start: str | Path | None = None) -> Path:
    return get_project_root(start) / "configs"


def get_data_dir(start: str | Path | None = None) -> Path:
    return get_project_root(start) / "data"


def get_results_dir(start: str | Path | None = None) -> Path:
    return get_project_root(start) / "results"


def resolve_under_root(path: str | Path, *, start: str | Path | None = None) -> Path:
    """Resolve a path against the project root unless it is already absolute.

    Absolute paths are allowed at runtime (for the cluster scratch prefixes) but must
    not be committed in configuration files.
    """

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (get_project_root(start) / candidate).resolve()
