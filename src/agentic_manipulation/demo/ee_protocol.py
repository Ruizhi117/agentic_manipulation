"""Validated EE control commands and calibrated camera responses."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from agentic_manipulation.demo.protocol import homogeneous_matrix
from agentic_manipulation.errors import ConfigurationError


@dataclass(frozen=True)
class EECommand:
    command_id: int
    target_pose: np.ndarray
    gripper: float
    steps: int


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{field} must be a positive integer")
    return value


def parse_ee_command(payload: Mapping[str, object]) -> EECommand:
    """Parse a control payload without coercing unsafe values."""

    command_id = _positive_integer(payload.get("command_id"), "command_id")
    steps = _positive_integer(payload.get("steps", 50), "steps")
    target_pose = homogeneous_matrix(payload.get("target_pose"), "target_pose")
    gripper_value = payload.get("gripper", 0.0)
    if isinstance(gripper_value, bool):
        raise ConfigurationError("gripper must be finite")
    try:
        gripper = float(gripper_value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("gripper must be finite") from exc
    if not math.isfinite(gripper):
        raise ConfigurationError("gripper must be finite")
    return EECommand(command_id, target_pose, gripper, steps)


def _intrinsic_matrix(value: object) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"intrinsic must be numeric: {exc}") from exc
    if matrix.shape != (3, 3):
        raise ConfigurationError(
            f"intrinsic shape must be (3, 3), got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ConfigurationError("intrinsic must contain finite values")
    return matrix


def camera_response(
    command_id: int,
    ee_pose: object,
    rgb_path: str,
    depth_path: str,
    intrinsic: object,
    world_from_camera: object,
    *,
    is_mock: bool,
    status: str = "ok",
    message: str = "",
) -> dict[str, object]:
    """Build one protocol-compatible post-command vision response."""

    command = _positive_integer(command_id, "command_id")
    if status not in ("ok", "error"):
        raise ConfigurationError("status must be 'ok' or 'error'")
    if not isinstance(is_mock, bool):
        raise ConfigurationError("is_mock must be boolean")
    if not isinstance(rgb_path, str) or not isinstance(depth_path, str):
        raise ConfigurationError("rgb_path and depth_path must be strings")
    ee = homogeneous_matrix(ee_pose, "ee_pose")
    camera_intrinsic = _intrinsic_matrix(intrinsic)
    world_camera = homogeneous_matrix(world_from_camera, "world_from_camera")
    result: dict[str, object] = {
        "command_id": command,
        "status": status,
        "ee_pose": ee.tolist(),
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "intrinsic": camera_intrinsic.tolist(),
        "world_from_camera": world_camera.tolist(),
        "is_mock": is_mock,
    }
    if message:
        result["message"] = str(message)
    return result
