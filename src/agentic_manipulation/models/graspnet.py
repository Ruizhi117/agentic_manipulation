"""GraspNet boundary: point clouds in, grasp poses out."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from agentic_manipulation.errors import GraspNetUnavailableError
from agentic_manipulation.types import GraspCandidate


class GraspProvider(Protocol):
    provider_name: str

    def predict(
        self, target_points: np.ndarray, workspace_points: np.ndarray
    ) -> tuple[GraspCandidate, ...]: ...


InferenceFn = Callable[[np.ndarray, np.ndarray], Sequence[Mapping[str, Any]]]


def _point_cloud(value: np.ndarray, name: str, *, allow_empty: bool) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise GraspNetUnavailableError(f"{name} must have shape (N, 3)")
    if not allow_empty and len(points) == 0:
        raise GraspNetUnavailableError(f"{name} is empty")
    if not np.isfinite(points).all():
        raise GraspNetUnavailableError(f"{name} must contain finite points")
    return points


class DeterministicTopDownGraspProvider:
    """Geometric fixture for tests; never presented as real GraspNet."""

    provider_name = "deterministic"

    def predict(
        self, target_points: np.ndarray, workspace_points: np.ndarray
    ) -> tuple[GraspCandidate, ...]:
        target = _point_cloud(target_points, "target point cloud", allow_empty=False)
        _point_cloud(workspace_points, "workspace point cloud", allow_empty=False)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        pose[:3, 3] = np.mean(target, axis=0)
        span = np.ptp(target[:, :2], axis=0)
        width = max(0.01, min(0.08, float(np.min(span)) + 0.01))
        return (
            GraspCandidate(
                world_from_gripper=pose,
                width_m=width,
                score=1.0,
                collision_free=True,
                provider_name=self.provider_name,
                metadata={"mode": "mock", "strategy": "top_down_centroid"},
            ),
        )


class _OfficialGraspNetInference:
    """Lazy adapter following the official graspnet-baseline demo API."""

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        *,
        num_points: int = 20_000,
        collision_threshold: float = 0.01,
        voxel_size: float = 0.01,
    ) -> None:
        try:
            torch = importlib.import_module("torch")
            graspnet_module = importlib.import_module("graspnet")
            graspnet_api = importlib.import_module("graspnetAPI")
        except ImportError as exc:
            raise GraspNetUnavailableError(
                "official GraspNet modules are missing; install graspnet-baseline "
                "pointnet2/knn operators and graspnetAPI, then expose its models "
                "and utils directories on PYTHONPATH"
            ) from exc
        self._torch = torch
        self._pred_decode = graspnet_module.pred_decode
        self._grasp_group = graspnet_api.GraspGroup
        self._device = torch.device(device)
        self._num_points = num_points
        self._collision_threshold = collision_threshold
        self._voxel_size = voxel_size
        self._net = graspnet_module.GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        ).to(self._device)
        try:
            state = torch.load(
                checkpoint, map_location=self._device, weights_only=False
            )
            self._net.load_state_dict(state["model_state_dict"])
        except (OSError, KeyError, RuntimeError) as exc:
            raise GraspNetUnavailableError(
                f"failed to load GraspNet checkpoint {checkpoint}: {exc}"
            ) from exc
        self._net.eval()

    def __call__(
        self, target_points: np.ndarray, workspace_points: np.ndarray
    ) -> list[dict[str, Any]]:
        torch = self._torch
        rng = np.random.default_rng(0)
        replace = len(workspace_points) < self._num_points
        indices = rng.choice(len(workspace_points), self._num_points, replace=replace)
        sampled = workspace_points[indices].astype(np.float32, copy=False)
        end_points = {
            "point_clouds": torch.from_numpy(sampled[None]).to(self._device),
            "cloud_colors": np.zeros_like(sampled, dtype=np.float32),
        }
        with torch.no_grad():
            decoded = self._pred_decode(self._net(end_points))[0]
        group = self._grasp_group(decoded.detach().cpu().numpy())

        collision_free = np.ones(len(group), dtype=bool)
        if self._collision_threshold > 0 and len(group):
            try:
                detector_module = importlib.import_module("collision_detector")
                detector = detector_module.ModelFreeCollisionDetector(
                    workspace_points, voxel_size=self._voxel_size
                )
                collision_free = ~detector.detect(
                    group,
                    approach_dist=0.05,
                    collision_thresh=self._collision_threshold,
                )
            except ImportError as exc:
                raise GraspNetUnavailableError(
                    "collision_detector from graspnet-baseline utils is missing"
                ) from exc

        padding = 0.04
        target_low = np.min(target_points, axis=0) - padding
        target_high = np.max(target_points, axis=0) + padding
        rows: list[dict[str, Any]] = []
        for index in range(len(group)):
            translation = group.translations[index]
            if not np.all((translation >= target_low) & (translation <= target_high)):
                continue
            rows.append(
                {
                    "translation": translation,
                    "rotation_matrix": group.rotation_matrices[index],
                    "width": float(group.widths[index]),
                    "score": float(group.scores[index]),
                    "collision_free": bool(collision_free[index]),
                }
            )
        return rows


class GraspNetProvider:
    provider_name = "graspnet"

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        *,
        inference_fn: InferenceFn | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise GraspNetUnavailableError(
                f"GraspNet checkpoint does not exist: {self.checkpoint}"
            )
        self.device = device
        self._inference_fn = inference_fn

    def _inference(self) -> InferenceFn:
        if self._inference_fn is None:
            self._inference_fn = _OfficialGraspNetInference(
                self.checkpoint, self.device
            )
        return self._inference_fn

    def predict(
        self, target_points: np.ndarray, workspace_points: np.ndarray
    ) -> tuple[GraspCandidate, ...]:
        target = _point_cloud(target_points, "target point cloud", allow_empty=False)
        workspace = _point_cloud(
            workspace_points, "workspace point cloud", allow_empty=False
        )
        try:
            rows = self._inference()(target, workspace)
            candidates = []
            for row in rows:
                rotation = np.asarray(row["rotation_matrix"], dtype=np.float32)
                translation = np.asarray(row["translation"], dtype=np.float32)
                if rotation.shape != (3, 3) or translation.shape != (3,):
                    raise ValueError("invalid rotation or translation shape")
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = rotation
                pose[:3, 3] = translation
                candidates.append(
                    GraspCandidate(
                        world_from_gripper=pose,
                        width_m=float(row["width"]),
                        score=float(row["score"]),
                        collision_free=bool(row.get("collision_free", True)),
                        provider_name=self.provider_name,
                        metadata={"checkpoint": str(self.checkpoint)},
                    )
                )
            return tuple(candidates)
        except GraspNetUnavailableError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise GraspNetUnavailableError(
                f"GraspNet inference returned invalid candidates: {exc}"
            ) from exc


def select_grasp(
    candidates: Sequence[GraspCandidate],
    workspace_bounds: tuple[np.ndarray, np.ndarray],
    max_width_m: float,
    reachable: Callable[[np.ndarray], bool],
) -> GraspCandidate:
    low = np.asarray(workspace_bounds[0], dtype=np.float64)
    high = np.asarray(workspace_bounds[1], dtype=np.float64)
    if low.shape != (3,) or high.shape != (3,) or np.any(low >= high):
        raise ValueError("workspace_bounds must contain ordered 3-vectors")
    valid = []
    for candidate in candidates:
        position = candidate.world_from_gripper[:3, 3]
        if not candidate.collision_free:
            continue
        if candidate.width_m > max_width_m:
            continue
        if not np.all((position >= low) & (position <= high)):
            continue
        if not reachable(candidate.world_from_gripper):
            continue
        valid.append(candidate)
    if not valid:
        raise GraspNetUnavailableError("no valid grasp candidate remains after filtering")
    return max(valid, key=lambda candidate: candidate.score)
