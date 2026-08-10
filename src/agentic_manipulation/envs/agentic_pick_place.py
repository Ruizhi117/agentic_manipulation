"""ManiSkill semantic pick-and-place scene for the compound command demo."""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from .layout import SceneObjectSpec, bin_wall_components, sample_layout


TABLE_HEIGHT = 0.9196429
BIN_WALL_THICKNESS = 0.008


@register_env("AgenticPickPlace-v1", max_episode_steps=1000)
class AgenticPickPlaceEnv(BaseEnv):
    """A two-category, two-bin tabletop scene using the mobile dual-arm xlerobot."""

    SUPPORTED_ROBOTS = ["xlerobot"]

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "xlerobot",
        robot_init_qpos_noise: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.semantic_actors: dict[str, Any] = {}
        self.bin_inner_aabbs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.instance_segmentation_ids: dict[int, str] = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        pose = sapien_utils.look_at(
            eye=[0.10, -0.10, 0.72], target=[-0.50, -0.10, 0.02]
        )
        return [CameraConfig("scene_camera", pose, 512, 512, 1.0, 0.01, 10.0)]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(eye=[1.0, 0.0, 1.25], target=[0.0, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 640, 480, 1.0, 0.01, 100)

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(
            options,
            sapien.Pose(p=[-0.72, 0.0, -TABLE_HEIGHT]),
        )

    def _build_bin(self, spec: SceneObjectSpec):
        builder = self.scene.create_actor_builder()
        material = sapien.render.RenderMaterial(base_color=spec.rgba)
        components = bin_wall_components(
            spec.half_size_xyz, wall_thickness=BIN_WALL_THICKNESS
        )
        for center, half_size in components:
            pose = sapien.Pose(p=center)
            builder.add_box_collision(pose=pose, half_size=half_size)
            builder.add_box_visual(pose=pose, half_size=half_size, material=material)
        builder.initial_pose = sapien.Pose(p=spec.center_xyz)
        return builder.build_kinematic(name=spec.instance_id)

    def _build_actor(self, spec: SceneObjectSpec):
        pose = sapien.Pose(p=spec.center_xyz)
        if spec.shape == "cylinder":
            return actors.build_cylinder(
                self.scene,
                radius=float(spec.half_size_xyz[0]),
                half_length=float(spec.half_size_xyz[2]),
                color=spec.rgba,
                name=spec.instance_id,
                initial_pose=pose,
            )
        if spec.shape == "banana":
            builder = self.scene.create_actor_builder()
            yellow = sapien.render.RenderMaterial(base_color=spec.rgba)
            tip = sapien.render.RenderMaterial(base_color=(0.32, 0.16, 0.03, 1.0))
            segment_specs = (
                ([-0.032, 0.006, 0.0], 0.32),
                ([0.0, -0.006, 0.0], 0.0),
                ([0.032, 0.006, 0.0], -0.32),
            )
            for center, angle in segment_specs:
                local_pose = sapien.Pose(
                    p=center,
                    q=[np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)],
                )
                builder.add_capsule_collision(
                    pose=local_pose,
                    radius=0.014,
                    half_length=0.018,
                    density=350,
                )
                builder.add_capsule_visual(
                    pose=local_pose,
                    radius=0.014,
                    half_length=0.018,
                    material=yellow,
                )
            builder.add_sphere_visual(
                pose=sapien.Pose(p=[-0.053, 0.013, 0.0]),
                radius=0.007,
                material=tip,
            )
            builder.add_sphere_visual(
                pose=sapien.Pose(p=[0.053, 0.013, 0.0]),
                radius=0.007,
                material=tip,
            )
            builder.initial_pose = pose
            return builder.build(name=spec.instance_id)
        if spec.shape == "box":
            return actors.build_box(
                self.scene,
                half_sizes=spec.half_size_xyz,
                color=spec.rgba,
                name=spec.instance_id,
                initial_pose=pose,
            )
        if spec.shape == "bin":
            return self._build_bin(spec)
        raise ValueError(f"unsupported scene shape: {spec.shape}")

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._layout_specs = sample_layout(seed=0)
        self.semantic_actors = {
            spec.instance_id: self._build_actor(spec) for spec in self._layout_specs
        }
        for spec in self._layout_specs:
            if spec.shape == "bin":
                inset = np.array(
                    [BIN_WALL_THICKNESS, BIN_WALL_THICKNESS, 0.0], dtype=np.float64
                )
                low = spec.center_xyz - spec.half_size_xyz + inset
                high = spec.center_xyz + spec.half_size_xyz - inset
                low[2] = spec.center_xyz[2] + BIN_WALL_THICKNESS
                self.bin_inner_aabbs[spec.instance_id] = (low, high)
        self.instance_segmentation_ids = {
            int(actor.per_scene_id[0].item()): instance_id
            for instance_id, actor in self.semantic_actors.items()
        }

    def _initialize_episode(
        self, env_idx: torch.Tensor, options: dict[str, Any]
    ) -> None:
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self.agent.reset(self.agent.keyframes["rest"].qpos)
            self.agent.robot.set_pose(
                sapien.Pose(p=[-0.72, 0.0, -TABLE_HEIGHT])
            )
            seed = int(self._episode_seed[int(env_idx[0].item())])
            layout = sample_layout(seed)
            self._layout_specs = layout
            count = len(env_idx)
            for spec in layout:
                xyz = torch.as_tensor(
                    np.repeat(spec.center_xyz[None, :], count, axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.semantic_actors[spec.instance_id].set_pose(
                    Pose.create_from_pq(xyz)
                )

    def object_centers(self) -> dict[str, np.ndarray]:
        return {
            instance_id: actor.pose.p[0].detach().cpu().numpy().copy()
            for instance_id, actor in self.semantic_actors.items()
        }

    def is_actor_in_bin(self, instance_id: str, bin_id: str) -> torch.Tensor:
        if instance_id not in self.semantic_actors:
            raise KeyError(f"unknown actor: {instance_id}")
        if bin_id not in self.bin_inner_aabbs:
            raise KeyError(f"unknown bin: {bin_id}")
        position = self.semantic_actors[instance_id].pose.p
        low, high = self.bin_inner_aabbs[bin_id]
        low_t = torch.as_tensor(low, dtype=position.dtype, device=position.device)
        high_t = torch.as_tensor(high, dtype=position.dtype, device=position.device)
        return torch.all((position >= low_t) & (position <= high_t), dim=1)

    def evaluate(self) -> dict[str, torch.Tensor]:
        all_placed = (
            self.is_actor_in_bin("tomato_can_1", "gray_bin")
            & self.is_actor_in_bin("tomato_can_2", "gray_bin")
            & (
                self.is_actor_in_bin("banana_1", "purple_bin")
                | self.is_actor_in_bin("banana_2", "purple_bin")
            )
        )
        return {"success": all_placed}

    def _get_obs_extra(self, info: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {
            f"{instance_id}_pose": actor.pose.raw_pose
            for instance_id, actor in self.semantic_actors.items()
        }

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return info["success"].float()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info)
