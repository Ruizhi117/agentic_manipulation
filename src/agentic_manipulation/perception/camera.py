"""Normalize ManiSkill camera observations into calibrated RGB-D frames."""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

import numpy as np

from agentic_manipulation.errors import PerceptionError
from agentic_manipulation.types import CameraFrame


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _matrix_for_env(value: Any, env_index: int, shape: tuple[int, int]) -> np.ndarray:
    matrix = _to_numpy(value)
    if matrix.ndim == 3:
        matrix = matrix[env_index]
    if matrix.shape != shape:
        raise PerceptionError(f"camera matrix must have shape {shape}, got {matrix.shape}")
    return matrix.astype(np.float32, copy=False)


def _extrinsic_for_env(value: Any, env_index: int) -> np.ndarray:
    matrix = _to_numpy(value)
    if matrix.ndim == 3:
        matrix = matrix[env_index]
    if matrix.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float32)
        homogeneous[:3] = matrix
        return homogeneous
    if matrix.shape != (4, 4):
        raise PerceptionError(
            "camera extrinsic must have shape (3, 4) or (4, 4), "
            f"got {matrix.shape}"
        )
    return matrix.astype(np.float32, copy=False)


class CameraAdapter:
    """Extract one environment's RGB-D observation and CV calibration."""

    def capture(
        self,
        observation: Mapping[str, Any],
        camera_params: Mapping[str, Any],
        camera_uid: str,
        *,
        env_index: int = 0,
        timestamp: float | None = None,
    ) -> CameraFrame:
        sensors = observation.get("sensor_data", observation)
        if camera_uid not in sensors:
            raise PerceptionError(f"camera '{camera_uid}' missing from sensor_data")
        sensor = sensors[camera_uid]
        if not isinstance(sensor, Mapping):
            raise PerceptionError(f"camera '{camera_uid}' sensor data must be a mapping")
        for field in ("rgb", "depth"):
            if field not in sensor:
                raise PerceptionError(f"camera '{camera_uid}' missing {field}")

        rgb = _to_numpy(sensor["rgb"])
        if rgb.ndim == 4:
            rgb = rgb[env_index]
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise PerceptionError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating) and np.isfinite(rgb).all():
                scale = 255.0 if float(np.max(rgb, initial=0.0)) <= 1.0 else 1.0
                rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
            else:
                raise PerceptionError(f"rgb dtype must be uint8 or finite float, got {rgb.dtype}")

        depth = _to_numpy(sensor["depth"])
        if depth.ndim == 4:
            depth = depth[env_index]
        elif depth.ndim == 3 and depth.shape[1:] == rgb.shape[:2]:
            depth = depth[env_index]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != rgb.shape[:2]:
            raise PerceptionError(
                f"depth must match rgb height/width {rgb.shape[:2]}, got {depth.shape}"
            )
        if depth.dtype == np.int16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32, copy=False)

        segmentation = None
        if "segmentation" in sensor:
            segmentation = _to_numpy(sensor["segmentation"])
            if segmentation.ndim == 4:
                segmentation = segmentation[env_index]
            elif segmentation.ndim == 3 and segmentation.shape[1:] == rgb.shape[:2]:
                segmentation = segmentation[env_index]
            if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
                segmentation = segmentation[..., 0]
            if segmentation.shape != rgb.shape[:2]:
                raise PerceptionError(
                    "segmentation must match rgb height/width "
                    f"{rgb.shape[:2]}, got {segmentation.shape}"
                )
            segmentation = segmentation.astype(np.int32, copy=False)

        params = camera_params.get(camera_uid)
        if not isinstance(params, Mapping):
            raise PerceptionError(f"camera '{camera_uid}' parameters are missing")
        if "intrinsic_cv" not in params:
            raise PerceptionError(f"camera '{camera_uid}' missing intrinsic_cv")
        if "extrinsic_cv" not in params:
            raise PerceptionError(f"camera '{camera_uid}' missing extrinsic_cv")
        intrinsic = _matrix_for_env(params["intrinsic_cv"], env_index, (3, 3))
        camera_from_world = _extrinsic_for_env(params["extrinsic_cv"], env_index)
        try:
            world_from_camera = np.linalg.inv(camera_from_world).astype(np.float32)
        except np.linalg.LinAlgError as exc:
            raise PerceptionError("extrinsic_cv must be invertible") from exc

        return CameraFrame(
            rgb=rgb,
            depth_m=depth,
            intrinsic=intrinsic,
            world_from_camera=world_from_camera,
            segmentation=segmentation,
            timestamp=time.monotonic() if timestamp is None else timestamp,
        )
