"""Offline stage video and TCP/gripper trace recorder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from agentic_manipulation.errors import ExecutionError


def _numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _unbatch(value: object, shape: tuple[int, ...], field: str) -> np.ndarray:
    array = _numpy(value)
    if array.shape == (1, *shape):
        array = array[0]
    if array.shape != shape or not np.isfinite(array).all():
        raise ExecutionError(f"{field} must be a finite {shape} array")
    return array.astype(np.float64, copy=False)


def _matrix44(value: object, field: str) -> np.ndarray:
    array = _numpy(value)
    if array.shape == (1, 3, 4):
        array = array[0]
    if array.shape == (3, 4):
        result = np.eye(4, dtype=np.float64)
        result[:3] = array
        array = result
    elif array.shape == (1, 4, 4):
        array = array[0]
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ExecutionError(f"{field} must be a finite 3x4 or 4x4 matrix")
    return array.astype(np.float64, copy=False)


def _rgb(value: object) -> np.ndarray:
    array = _numpy(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ExecutionError("scene_camera RGB must have shape (H, W, 3|4)")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ExecutionError("scene_camera RGB must contain finite values")
        array = np.clip(array * (255.0 if array.max(initial=0.0) <= 1.0 else 1.0), 0, 255)
    return array.astype(np.uint8, copy=True)


def _project(point_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[int, int] | None:
    camera = extrinsic @ point_world
    if camera[2] <= 1e-6:
        return None
    pixel = intrinsic @ camera[:3]
    return int(round(pixel[0] / pixel[2])), int(round(pixel[1] / pixel[2]))


class StageRecorder:
    """Collect overview frames, wrist-axis overlays, and executed TCP poses."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.motion: list[dict[str, object]] = []

    def capture(self, stage: str, observation: dict[str, object] | None = None) -> None:
        if observation is None:
            raise ExecutionError("capture requires a ManiSkill observation")
        try:
            sensors = observation["sensor_data"]
            params = observation["sensor_param"]
            frame = _rgb(sensors["scene_camera"]["rgb"])
            scene_param = params["scene_camera"]
            hand_param = params["hand_camera"]
            intrinsic = _unbatch(scene_param["intrinsic_cv"], (3, 3), "scene intrinsic")
            scene_extrinsic = _matrix44(scene_param["extrinsic_cv"], "scene extrinsic")
            hand_extrinsic = _matrix44(hand_param["extrinsic_cv"], "hand extrinsic")
        except (KeyError, TypeError) as exc:
            raise ExecutionError(f"observation lacks recorder camera data: {exc}") from exc

        world_from_hand = np.linalg.inv(hand_extrinsic)
        origin = world_from_hand @ np.array([0.0, 0.0, 0.0, 1.0])
        endpoint = world_from_hand @ np.array([0.0, 0.0, 0.20, 1.0])
        start_pixel = _project(origin, intrinsic, scene_extrinsic)
        end_pixel = _project(endpoint, intrinsic, scene_extrinsic)

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        if start_pixel is not None and end_pixel is not None:
            draw.line((start_pixel, end_pixel), fill=(0, 255, 0), width=2)
            radius = 2
            draw.ellipse(
                (
                    start_pixel[0] - radius,
                    start_pixel[1] - radius,
                    start_pixel[0] + radius,
                    start_pixel[1] + radius,
                ),
                fill=(0, 255, 0),
            )
        draw.rectangle((0, 0, max(1, 7 * len(stage) + 4), 14), fill=(0, 0, 0))
        draw.text((2, 1), stage, fill=(255, 255, 255))
        self.frames.append(np.asarray(image, dtype=np.uint8))

    def record_motion(self, stage: str, world_from_ee: object, gripper: float) -> None:
        pose = _matrix44(world_from_ee, "world_from_ee")
        if not np.isfinite(gripper):
            raise ExecutionError("gripper must be finite")
        self.motion.append(
            {
                "index": len(self.motion),
                "stage": str(stage),
                "world_from_ee": pose.tolist(),
                "gripper": float(gripper),
            }
        )

    def write_motion_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(self.motion, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(target)
        except (OSError, TypeError, ValueError) as exc:
            raise ExecutionError(f"failed to write motion JSON {target}: {exc}") from exc

    def write_mp4(self, path: str | Path, fps: int = 20) -> None:
        if not self.frames:
            raise ExecutionError("cannot write MP4 without captured frames")
        if isinstance(fps, bool) or int(fps) != fps or fps <= 0:
            raise ExecutionError("fps must be a positive integer")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio

            imageio.mimsave(
                target,
                self.frames,
                fps=int(fps),
                codec="libx264",
                macro_block_size=1,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ExecutionError(f"failed to write MP4 {target}: {exc}") from exc
