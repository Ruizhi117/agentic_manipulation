"""Adapter from a live ManiSkill environment to the runtime scene Protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from agentic_manipulation.errors import ConfigurationError
from agentic_manipulation.perception.camera import CameraAdapter
from agentic_manipulation.types import CameraFrame


class ManiSkillRuntimeScene:
    def __init__(self, env: Any, camera_uid: str = "scene_camera") -> None:
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.camera_uid = camera_uid
        sensors = getattr(self.base_env, "_sensors", {})
        if camera_uid not in sensors:
            raise ConfigurationError(
                f"ManiSkill camera {camera_uid!r} is unavailable; "
                f"available cameras: {sorted(sensors)}"
            )
        self._camera = CameraAdapter()
        self._specs = {
            spec.instance_id: spec for spec in self.base_env._layout_specs
        }

    def capture(self) -> CameraFrame:
        observation = self.base_env.get_obs(unflattened=True)
        return self._camera.capture(
            observation,
            observation["sensor_param"],
            self.camera_uid,
        )

    def visible_instances(self) -> Mapping[str, str]:
        return {
            instance_id: spec.semantic_label
            for instance_id, spec in self._specs.items()
        }

    def segmentation_ids(self) -> Mapping[int, str]:
        return dict(self.base_env.instance_segmentation_ids)

    def centers(self) -> Mapping[str, np.ndarray]:
        return self.base_env.object_centers()

    def workspace_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([-0.70, -0.42, 0.0], dtype=np.float32),
            np.array([-0.30, 0.18, 0.45], dtype=np.float32),
        )

    def bin_inner_aabb(self, bin_id: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            return self.base_env.bin_inner_aabbs[bin_id]
        except KeyError as exc:
            raise ConfigurationError(f"unknown destination bin: {bin_id}") from exc

    def object_half_height(self, instance_id: str) -> float:
        try:
            return float(self._specs[instance_id].half_size_xyz[2])
        except KeyError as exc:
            raise ConfigurationError(f"unknown scene object: {instance_id}") from exc

    def is_in_bin(self, instance_id: str, bin_id: str) -> bool:
        value = self.base_env.is_actor_in_bin(instance_id, bin_id)
        return bool(value[0].item())

    def is_released(self, instance_id: str) -> bool:
        actor = self.base_env.semantic_actors[instance_id]
        held = self.base_env.agent.is_grasping(actor, arm_id=1)
        return not bool(held[0].item())

    def is_stable(self, instance_id: str) -> bool:
        actor = self.base_env.semantic_actors[instance_id]
        return bool(actor.is_static()[0].item())

    def close(self) -> None:
        self.env.close()
