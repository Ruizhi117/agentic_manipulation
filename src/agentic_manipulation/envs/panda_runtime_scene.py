"""Runtime state/camera adapter for the registered Panda sorting scene."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from agentic_manipulation.envs.ee_camera_scene import (
    DESTINATION_INSTANCE_IDS,
    GRASPABLE_INSTANCE_IDS,
)
from agentic_manipulation.errors import ConfigurationError
from agentic_manipulation.perception.camera import CameraAdapter
from agentic_manipulation.types import CameraFrame


def _bool(value: object) -> bool:
    if hasattr(value, "detach"):
        return bool(value.reshape(-1)[0].detach().cpu().item())
    return bool(np.asarray(value).reshape(-1)[0])


class PandaSortingScene:
    """Expose wrist RGBD and authoritative simulator truth to the Panda agent."""

    def __init__(self, env: Any, camera_uid: str = "hand_camera") -> None:
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if self.base_env.robot_uids != "panda_wristcam":
            raise ConfigurationError("PandaSortingScene requires panda_wristcam")
        if camera_uid not in getattr(self.base_env, "_sensors", {}):
            raise ConfigurationError(f"Panda camera is unavailable: {camera_uid}")
        self.camera_uid = camera_uid
        self._camera = CameraAdapter()

    def observation(self) -> Mapping[str, Any]:
        return self.base_env.get_obs()

    def capture(self) -> CameraFrame:
        observation = self.observation()
        return self._camera.capture(
            observation,
            observation["sensor_param"],
            self.camera_uid,
        )

    @property
    def home_pose(self) -> np.ndarray:
        return np.asarray(self.base_env.observation_home_pose, dtype=np.float64).copy()

    def visible_instances(self) -> Mapping[str, str]:
        return {
            instance_id: instance_id
            for instance_id in GRASPABLE_INSTANCE_IDS + DESTINATION_INSTANCE_IDS
        }

    def segmentation_ids(self) -> Mapping[int, str]:
        return dict(self.base_env.instance_segmentation_ids)

    def centers(self) -> Mapping[str, np.ndarray]:
        return self.base_env.object_centers()

    def bin_inner_aabb(self, bin_id: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            low, high = self.base_env.bin_inner_aabbs[bin_id]
        except KeyError as exc:
            raise ConfigurationError(f"unknown destination bin: {bin_id}") from exc
        return np.asarray(low).copy(), np.asarray(high).copy()

    def object_half_height(self, instance_id: str) -> float:
        try:
            return float(self.base_env.object_half_heights[instance_id])
        except KeyError as exc:
            raise ConfigurationError(f"unknown graspable object: {instance_id}") from exc

    def is_grasping(self, instance_id: str) -> bool:
        try:
            actor = self.base_env.semantic_actors[instance_id]
        except KeyError as exc:
            raise ConfigurationError(f"unknown semantic actor: {instance_id}") from exc
        return _bool(self.base_env.agent.is_grasping(actor))

    def is_in_bin(self, instance_id: str, bin_id: str) -> bool:
        try:
            center = self.base_env.semantic_actors[instance_id].pose.p[0]
        except KeyError as exc:
            raise ConfigurationError(f"unknown semantic actor: {instance_id}") from exc
        if hasattr(center, "detach"):
            center = center.detach().cpu().numpy()
        center = np.asarray(center, dtype=np.float64)
        low, high = self.bin_inner_aabb(bin_id)
        return bool(
            np.all(center[:2] >= low[:2])
            and np.all(center[:2] <= high[:2])
            and center[2] >= low[2]
        )

    def is_released(self, instance_id: str) -> bool:
        return not self.is_grasping(instance_id)

    def is_stable(self, instance_id: str) -> bool:
        try:
            actor = self.base_env.semantic_actors[instance_id]
        except KeyError as exc:
            raise ConfigurationError(f"unknown semantic actor: {instance_id}") from exc
        return _bool(actor.is_static())

    def close(self) -> None:
        self.env.close()
