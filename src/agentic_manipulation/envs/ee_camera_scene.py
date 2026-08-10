"""Panda wrist-camera sorting scene with six objects and two open bins."""

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


GRASPABLE_INSTANCE_IDS = (
    "red_cube",
    "blue_cube",
    "yellow_block",
    "purple_block",
    "green_cylinder",
    "orange_cylinder",
)
DESTINATION_INSTANCE_IDS = ("white_bin", "pink_bin")
SEMANTIC_LABELS = {
    instance_id: instance_id
    for instance_id in GRASPABLE_INSTANCE_IDS + DESTINATION_INSTANCE_IDS
}

_ROBOT_BASE_POSE = sapien.Pose(p=[-0.615, 0.0, 0.0])
_PANDA_OBSERVATION_QPOS = np.array(
    [
        -0.0003732363,
        0.46138695,
        0.0000882985,
        -1.3107156,
        0.0010112347,
        1.7559249,
        0.78576607,
        0.04,
        0.04,
    ],
    dtype=np.float32,
)

_BIN_HALF_X = 0.12
_BIN_HALF_Y = 0.16
_BIN_WALL_HALF_THICKNESS = 0.005
_BIN_FLOOR_HALF_HEIGHT = 0.005
_BIN_WALL_HALF_HEIGHT = 0.015
_BIN_CENTERS = {
    "white_bin": np.array([0, 0.3, 0.0], dtype=np.float64),
    "pink_bin": np.array([0.0, -0.3, 0.0], dtype=np.float64),
}

_OBJECT_CONFIGS: dict[str, tuple[str, np.ndarray, np.ndarray, list[float]]] = {
    "red_cube": (
        "cube",
        np.array([0.02, 0.035, 0.025]),
        np.array([0.025, 0.025, 0.025]),
        [0.90, 0.12, 0.10, 1.0],
    ),
    "blue_cube": (
        "cube",
        np.array([0.06, -0.035, 0.022]),
        np.array([0.022, 0.022, 0.022]),
        [0.10, 0.30, 0.90, 1.0],
    ),
    "yellow_block": (
        "box",
        np.array([0.12, 0.045, 0.018]),
        np.array([0.035, 0.020, 0.018]),
        [0.95, 0.75, 0.08, 1.0],
    ),
    "purple_block": (
        "box",
        np.array([0.16, -0.040, 0.020]),
        np.array([0.025, 0.032, 0.020]),
        [0.55, 0.20, 0.75, 1.0],
    ),
    # "green_cylinder": (
    #     "cylinder",
    #     np.array([0.045, 0.105, 0.030]),
    #     np.array([0.024, 0.024, 0.030]),
    #     [0.12, 0.70, 0.28, 1.0],
    # ),
    # "orange_cylinder": (
    #     "cylinder",
    #     np.array([0.135, -0.105, 0.027]),
    #     np.array([0.021, 0.021, 0.027]),
    #     [0.95, 0.38, 0.08, 1.0],
    # ),
}


@register_env("EECameraScene-v1", max_episode_steps=2000)
class EECameraSceneEnv(BaseEnv):
    """Two-bin sorting sandbox observed by a Panda wrist RGBD camera."""

    SUPPORTED_ROBOTS = [
        "panda_wristcam",
        "panda",
        "xarm6_robotiq_wristcam",
        "xarm6_robotiq",
    ]

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "panda_wristcam",
        robot_init_qpos_noise: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.semantic_actors: dict[str, Any] = {}
        self.instance_segmentation_ids: dict[int, str] = {}
        self.bin_inner_aabbs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.object_half_heights = {
            name: float(config[2][2]) for name, config in _OBJECT_CONFIGS.items()
        }
        self.observation_home_pose = np.eye(4, dtype=np.float64)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        pose = sapien_utils.look_at(
            eye=[0.52, -0.48, 0.68], target=[0.08, 0.0, 0.03]
        )
        return [
            CameraConfig(
                "scene_camera",
                pose,
                width=320,
                height=240,
                fov=np.pi / 2,
                near=0.01,
                far=10.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(
            eye=[0.72, -0.78, 0.72], target=[0.08, 0.0, 0.10]
        )
        return CameraConfig("render_camera", pose, 640, 480, 1.0, 0.01, 20.0)

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, _ROBOT_BASE_POSE)

    def _build_open_bin(
        self, name: str, center: np.ndarray, color: list[float]
    ) -> Any:
        builder = self.scene.create_actor_builder()
        material = sapien.render.RenderMaterial(base_color=color)
        t = _BIN_WALL_HALF_THICKNESS
        floor_hz = _BIN_FLOOR_HALF_HEIGHT
        wall_hz = _BIN_WALL_HALF_HEIGHT
        wall_z = 2.0 * floor_hz + wall_hz
        parts = [
            (sapien.Pose(p=[0.0, 0.0, floor_hz]), [_BIN_HALF_X, _BIN_HALF_Y, floor_hz]),
            (sapien.Pose(p=[-_BIN_HALF_X + t, 0.0, wall_z]), [t, _BIN_HALF_Y, wall_hz]),
            (sapien.Pose(p=[_BIN_HALF_X - t, 0.0, wall_z]), [t, _BIN_HALF_Y, wall_hz]),
            (sapien.Pose(p=[0.0, -_BIN_HALF_Y + t, wall_z]), [_BIN_HALF_X - 2 * t, t, wall_hz]),
            (sapien.Pose(p=[0.0, _BIN_HALF_Y - t, wall_z]), [_BIN_HALF_X - 2 * t, t, wall_hz]),
        ]
        for pose, half_size in parts:
            builder.add_box_collision(pose=pose, half_size=half_size)
            builder.add_box_visual(
                pose=pose, half_size=half_size, material=material
            )
        builder.initial_pose = sapien.Pose(p=center)
        return builder.build_kinematic(name=name)

    def _build_object(
        self,
        name: str,
        shape: str,
        center: np.ndarray,
        half_sizes: np.ndarray,
        color: list[float],
    ) -> Any:
        pose = sapien.Pose(p=center)
        if shape == "cube":
            return actors.build_cube(
                self.scene,
                half_size=float(half_sizes[0]),
                color=color,
                name=name,
                initial_pose=pose,
            )
        if shape == "box":
            return actors.build_box(
                self.scene,
                half_sizes=half_sizes,
                color=color,
                name=name,
                initial_pose=pose,
            )
        if shape == "cylinder":
            return actors.build_cylinder(
                self.scene,
                radius=float(half_sizes[0]),
                half_length=float(half_sizes[2]),
                color=color,
                name=name,
                initial_pose=pose,
            )
        raise ValueError(f"unsupported object shape: {shape}")

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        objects = {
            name: self._build_object(name, *config)
            for name, config in _OBJECT_CONFIGS.items()
        }
        bins = {
            "white_bin": self._build_open_bin(
                "white_bin", _BIN_CENTERS["white_bin"], [0.95, 0.95, 0.95, 1.0]
            ),
            "pink_bin": self._build_open_bin(
                "pink_bin", _BIN_CENTERS["pink_bin"], [1.0, 0.42, 0.65, 1.0]
            ),
        }
        self.semantic_actors = {**objects, **bins}
        for name, actor in self.semantic_actors.items():
            setattr(self, name, actor)
        inset = np.array(
            [_BIN_WALL_HALF_THICKNESS * 2, _BIN_WALL_HALF_THICKNESS * 2, 0.0],
            dtype=np.float64,
        )
        half = np.array(
            [_BIN_HALF_X, _BIN_HALF_Y, 2 * _BIN_FLOOR_HALF_HEIGHT + 2 * _BIN_WALL_HALF_HEIGHT],
            dtype=np.float64,
        )
        for name, center in _BIN_CENTERS.items():
            low = center - half + inset
            high = center + half - inset
            low[2] = center[2] + 2 * _BIN_FLOOR_HALF_HEIGHT
            self.bin_inner_aabbs[name] = (low, high)
        self.instance_segmentation_ids = {
            int(actor.per_scene_id[0].item()): name
            for name, actor in self.semantic_actors.items()
        }

    def _initialize_episode(
        self, env_idx: torch.Tensor, options: dict[str, Any]
    ) -> None:
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            if self.robot_uids == "panda_wristcam":
                qpos = torch.as_tensor(
                    _PANDA_OBSERVATION_QPOS,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                qpos = self.agent.keyframes["rest"].qpos
            self.agent.reset(qpos)
            self.agent.robot.set_pose(_ROBOT_BASE_POSE)
            count = len(env_idx)
            for name, (_, center, _, _) in _OBJECT_CONFIGS.items():
                xyz = torch.as_tensor(
                    np.repeat(center[None, :], count, axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.semantic_actors[name].set_pose(Pose.create_from_pq(xyz))
            home = self.agent.tcp_pose.to_transformation_matrix()[0]
            self.observation_home_pose = home.detach().cpu().numpy().astype(
                np.float64, copy=True
            )

    def object_centers(self) -> dict[str, np.ndarray]:
        return {
            name: actor.pose.p[0].detach().cpu().numpy().copy()
            for name, actor in self.semantic_actors.items()
        }

    def _get_obs_extra(self, info: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            **{
                f"{name}_pose": actor.pose.raw_pose
                for name, actor in self.semantic_actors.items()
            },
        }

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.zeros(action.shape[0], device=self.device, dtype=torch.float32)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info)
