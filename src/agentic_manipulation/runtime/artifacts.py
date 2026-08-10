"""Opt-in run artifacts; disabled by default."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
from PIL import Image

from agentic_manipulation.types import CameraFrame, GraspCandidate


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class ArtifactWriter:
    def __init__(self, root: Path | None, *, run_id: str | None = None) -> None:
        self._root = Path(root).resolve() if root is not None else None
        if run_id is not None and not _SAFE_NAME.fullmatch(run_id):
            raise ValueError("run_id may contain only letters, digits, '_' and '-'")
        self._run_id = run_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )

    @property
    def run_dir(self) -> Path | None:
        return None if self._root is None else self._root / self._run_id

    def _ensure_dir(self) -> Path | None:
        directory = self.run_dir
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_frame(self, frame: CameraFrame, *, name: str | None = None) -> None:
        if name is not None and not _SAFE_NAME.fullmatch(name):
            raise ValueError("artifact name may contain only letters, digits, '_' and '-'")
        directory = self._ensure_dir()
        if directory is None:
            return
        prefix = "" if name is None else f"{name}_"
        Image.fromarray(frame.rgb, mode="RGB").save(directory / f"{prefix}rgb.png")
        np.save(directory / f"{prefix}depth.npy", frame.depth_m)
        if name is not None and frame.segmentation is not None:
            np.save(directory / f"{prefix}segmentation.npy", frame.segmentation)

    def write_rgb(self, name: str, rgb: np.ndarray) -> Path | None:
        """Save an RGB debug overlay and return the concrete artifact path."""

        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("artifact name may contain only letters, digits, '_' and '-'")
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("rgb overlay must be a uint8 (H, W, 3) array")
        directory = self._ensure_dir()
        if directory is None:
            return None
        path = directory / f"{name}_rgb.png"
        Image.fromarray(image, mode="RGB").save(path)
        return path

    def write_array(self, name: str, value: np.ndarray) -> None:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("artifact name may contain only letters, digits, '_' and '-'")
        directory = self._ensure_dir()
        if directory is None:
            return
        np.save(directory / f"{name}.npy", np.asarray(value))

    def write_json(self, name: str, payload: Any) -> None:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("artifact name may contain only letters, digits, '_' and '-'")
        directory = self._ensure_dir()
        if directory is None:
            return
        if is_dataclass(payload):
            payload = asdict(payload)
        (directory / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def write_grasps(
        self,
        candidates: Sequence[GraspCandidate],
        *,
        name: str = "grasps",
    ) -> None:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("artifact name may contain only letters, digits, '_' and '-'")
        directory = self._ensure_dir()
        if directory is None:
            return
        poses = np.stack(
            [candidate.world_from_gripper for candidate in candidates], axis=0
        ) if candidates else np.empty((0, 4, 4), dtype=np.float32)
        np.savez_compressed(
            directory / f"{name}.npz",
            poses=poses,
            widths=np.asarray([candidate.width_m for candidate in candidates]),
            scores=np.asarray([candidate.score for candidate in candidates]),
            collision_free=np.asarray(
                [candidate.collision_free for candidate in candidates], dtype=bool
            ),
        )
