"""Validated filesystem protocol shared by the demo components."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from agentic_manipulation.errors import ConfigurationError


_REPLACE_RETRY_DELAYS_S = (0.01, 0.02, 0.04, 0.08, 0.10, 0.10, 0.10)


def _replace_with_permission_retry(temporary: Path, target: Path) -> None:
    for attempt in range(len(_REPLACE_RETRY_DELAYS_S) + 1):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == len(_REPLACE_RETRY_DELAYS_S):
                raise
            time.sleep(_REPLACE_RETRY_DELAYS_S[attempt])


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a user-provided path and require it to stay in the project."""

    root = Path(project_root).resolve()
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigurationError(
            f"path is outside project root {root}: {value}"
        ) from exc
    return resolved


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Serialize an object fully before atomically replacing its JSON file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        _replace_with_permission_retry(temporary, target)
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"failed to write JSON {target}: {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def read_json(path: Path) -> dict[str, object]:
    """Read a UTF-8 JSON object with a component-oriented error message."""

    source = Path(path)
    try:
        value: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"failed to read JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{source} must contain a JSON object")
    return value


def homogeneous_matrix(value: object, field: str) -> np.ndarray:
    """Return a validated float64 homogeneous transform."""

    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be numeric: {exc}") from exc
    if matrix.shape != (4, 4):
        raise ConfigurationError(f"{field} shape must be (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ConfigurationError(f"{field} must contain finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ConfigurationError(
            f"{field} last row must be [0, 0, 0, 1]"
        )
    return matrix


def point_cloud(value: object, field: str) -> np.ndarray:
    """Return a validated nonempty float32 ``(N, 3)`` point cloud."""

    try:
        points = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be numeric: {exc}") from exc
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ConfigurationError(f"{field} shape must be (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ConfigurationError(f"{field} must be nonempty")
    if not np.isfinite(points).all():
        raise ConfigurationError(f"{field} must contain finite values")
    return points
