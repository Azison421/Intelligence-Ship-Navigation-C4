"""Local artifact storage policy for NavAIg.

The navigation process may run on a ROS computer, but large source packages,
sidecars, grids, bags and checkpoints belong to the NavAIg workspace.  This
module gives producer tools a single safe default and rejects paths outside
the project root unless the caller explicitly copies files out of band.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


class StoragePolicyError(ValueError):
    """Raised when an artifact destination is not local to NavAIg."""


def resolve_project_storage(path: str | Path | None = None, *, category: str = "misc") -> Path:
    """Resolve an artifact destination under the NavAIg project root.

    Relative paths are interpreted below ``artifacts/<category>``.  Absolute
    paths are accepted only when their resolved location is still inside the
    project root.  UNC paths, ROS-host paths and other drives are therefore
    rejected before any directory or file is created.
    """

    if not category or Path(category).name != category or category in {".", ".."}:
        raise StoragePolicyError("storage category must be one path component")
    root = PROJECT_ROOT.resolve()
    if path is None:
        candidate = DEFAULT_ARTIFACT_ROOT / category
    else:
        raw = Path(path)
        if raw.is_absolute():
            candidate = raw
        else:
            candidate = DEFAULT_ARTIFACT_ROOT / category / raw
    if str(candidate).startswith("\\\\"):
        raise StoragePolicyError("UNC/remote storage paths are forbidden")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StoragePolicyError(
            f"artifact path must remain under NavAIg project root: {resolved}"
        ) from exc
    return resolved


def ensure_project_storage(path: str | Path | None = None, *, category: str = "misc") -> Path:
    """Create and return a project-local artifact directory."""

    directory = resolve_project_storage(path, category=category)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "PROJECT_ROOT",
    "StoragePolicyError",
    "ensure_project_storage",
    "resolve_project_storage",
]
