"""One-shot GraspNet component with explicit camera-to-EE calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from agentic_manipulation.demo.protocol import (
    homogeneous_matrix,
    point_cloud,
    resolve_project_path,
)
from agentic_manipulation.errors import (
    ConfigurationError,
    GraspNetUnavailableError,
)
from agentic_manipulation.models.graspnet import (
    DeterministicTopDownGraspProvider,
    GraspNetProvider,
    GraspProvider,
)


def compose_world_ee(
    world_from_camera: object,
    camera_from_grasp: object,
    grasp_from_ee: object,
) -> np.ndarray:
    """Compose calibrated transforms into a world-frame EE/TCP target."""

    world_camera = homogeneous_matrix(world_from_camera, "world_from_camera")
    camera_grasp = homogeneous_matrix(camera_from_grasp, "camera_from_grasp")
    grasp_ee = homogeneous_matrix(grasp_from_ee, "grasp_from_ee")
    return world_camera @ camera_grasp @ grasp_ee


def _positive_request_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError("request_id must be a positive integer")
    return value


def _cloud_from_request(
    request: Mapping[str, object], project_root: Path, field: str
) -> tuple[np.ndarray, str]:
    value = request.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a nonempty string")
    path = resolve_project_path(project_root, value)
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"failed to load point cloud {path}: {exc}") from exc
    return point_cloud(loaded, field.removesuffix("_path")), value


def _max_width(value: object) -> float:
    if isinstance(value, bool):
        raise ConfigurationError("max_width_m must be finite and positive")
    try:
        width = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("max_width_m must be finite and positive") from exc
    if not math.isfinite(width) or width <= 0:
        raise ConfigurationError("max_width_m must be finite and positive")
    return width


def run_grasp_request(
    request: Mapping[str, object],
    project_root: Path,
    mode: str,
    provider: GraspProvider | None = None,
) -> dict[str, object]:
    """Predict one grasp in camera coordinates and return calibrated world poses."""

    request_id = _positive_request_id(request.get("request_id"))
    if mode not in ("real", "mock"):
        raise ConfigurationError("mode must be 'real' or 'mock'")
    target_points, target_points_path = _cloud_from_request(
        request, project_root, "target_points_path"
    )
    workspace_points, workspace_points_path = _cloud_from_request(
        request, project_root, "workspace_points_path"
    )
    world_from_camera = homogeneous_matrix(
        request.get("world_from_camera"), "world_from_camera"
    )
    grasp_from_ee = homogeneous_matrix(
        request.get("grasp_from_ee"), "grasp_from_ee"
    )

    active_provider: GraspProvider
    if provider is not None:
        active_provider = provider
    elif mode == "mock":
        active_provider = DeterministicTopDownGraspProvider()
    else:
        checkpoint_value = request.get("checkpoint_path")
        if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
            raise ConfigurationError(
                "real grasp request requires checkpoint_path"
            )
        checkpoint = resolve_project_path(project_root, checkpoint_value)
        device = request.get("device", "cuda")
        if not isinstance(device, str) or not device.strip():
            raise ConfigurationError("device must be a nonempty string")
        active_provider = GraspNetProvider(checkpoint, device=device)

    max_width = _max_width(request.get("max_width_m", 0.081))
    candidates = active_provider.predict(target_points, workspace_points)
    valid = tuple(
        candidate
        for candidate in candidates
        if candidate.collision_free and candidate.width_m <= max_width
    )
    if not valid:
        raise GraspNetUnavailableError(
            "no collision-free grasp candidate satisfies max_width_m"
        )
    selected = max(valid, key=lambda candidate: candidate.score)
    camera_from_grasp = homogeneous_matrix(
        selected.world_from_gripper, "camera_from_grasp"
    )
    world_from_grasp = world_from_camera @ camera_from_grasp
    world_from_ee = compose_world_ee(
        world_from_camera, camera_from_grasp, grasp_from_ee
    )
    return {
        "request_id": request_id,
        "status": "ok",
        "provider": selected.provider_name,
        "is_mock": mode == "mock",
        "score": selected.score,
        "width_m": selected.width_m,
        "target_points_path": target_points_path,
        "workspace_points_path": workspace_points_path,
        "target_point_count": int(len(target_points)),
        "workspace_point_count": int(len(workspace_points)),
        "camera_from_grasp": camera_from_grasp.tolist(),
        "world_from_grasp": world_from_grasp.tolist(),
        "world_from_ee": world_from_ee.tolist(),
        "camera_grasp_position": camera_from_grasp[:3, 3].tolist(),
        "world_grasp_position": world_from_grasp[:3, 3].tolist(),
        "world_ee_position": world_from_ee[:3, 3].tolist(),
    }
