from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Table-top sorting scene
#
# Coordinate convention used by ManiSkill tabletop tasks:
#   +x : forward from the robot into the table
#   +y : robot's left
#   +z : up
#
# Layout (top view):
#
#                    +x
#                     ^
#                     |
#        WHITE BIN    |      PINK BIN
#          (+y)       |        (-y)
#             \       |       /
#              \  objects   /
#               \   ...    /
#                 ROBOT
#
# The table top created by TableSceneBuilder is z = 0.
# ---------------------------------------------------------------------------


@register_env("EECameraScene-v1", max_episode_steps=1000)
class EECameraScene(BaseEnv):
    """Wrist-camera tabletop sorting scene.

    Scene:
      - one ManiSkill table
      - robot mounted behind the table
      - six graspable objects in the central workspace
      - white open bin on robot-left (+y)
      - pink open bin on robot-right (-y)
      - fixed scene camera in addition to the robot wrist camera
    """

    SUPPORTED_ROBOTS = [
        "panda",
        "panda_wristcam",
        "xarm6_robotiq",
        "xarm6_robotiq_wristcam",
    ]

    # Robot base is behind the table, matching ManiSkill tabletop tasks.
    ROBOT_BASE_POSE = sapien.Pose(p=[-0.615, 0.0, 0.0])

    # Bin centers.  From the robot looking in +x:
    # +y is left, -y is right.
    WHITE_BIN_CENTER = (0, +0.28)
    PINK_BIN_CENTER = (0, -0.28)

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.0,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    @property
    def _default_sensor_configs(self):
        # A fixed overview RGB-D camera.
        # The wrist camera ("hand_camera") is supplied by the wristcam robot.
        pose = sapien_utils.look_at(
            eye=[0.60, 0.00, 0.72],
            target=[0.10, 0.00, 0.05],
        )
        return [
            CameraConfig(
                uid="scene_camera",
                pose=pose,
                width=640,
                height=480,
                fov=np.pi / 2,
                near=0.01,
                far=10.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # Human viewer: oblique view that clearly shows robot + both bins.
        pose = sapien_utils.look_at(
            eye=[0.72, -0.78, 0.72],
            target=[0.08, 0.00, 0.10],
        )
        return CameraConfig(
            uid="render_camera",
            pose=pose,
            width=640,
            height=480,
            fov=1.0,
            near=0.01,
            far=20.0,
        )

    # ------------------------------------------------------------------
    # Robot
    # ------------------------------------------------------------------

    def _load_agent(self, options: dict):
        super()._load_agent(options, self.ROBOT_BASE_POSE)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _build_open_bin(
        self,
        name: str,
        center_xy: tuple[float, float],
        color: list[float],
    ):
        """Build a static open-top receptacle from five collidable boxes."""

        # Outer footprint = 0.22 m x 0.18 m
        hx = 0.11
        hy = 0.09

        # Wall/floor thickness are represented as half-sizes.
        t = 0.006
        floor_hz = 0.006
        wall_hz = 0.040

        wall_z = 2.0 * floor_hz + wall_hz

        builder = self.scene.create_actor_builder()
        material = sapien.render.RenderMaterial(base_color=color)

        # Bottom.
        floor_pose = sapien.Pose(p=[0.0, 0.0, floor_hz])
        builder.add_box_collision(
            pose=floor_pose,
            half_size=[hx, hy, floor_hz],
        )
        builder.add_box_visual(
            pose=floor_pose,
            half_size=[hx, hy, floor_hz],
            material=material,
        )

        # Front/back walls in x.
        for x in (-hx + t, hx - t):
            p = sapien.Pose(p=[x, 0.0, wall_z])
            builder.add_box_collision(
                pose=p,
                half_size=[t, hy, wall_hz],
            )
            builder.add_box_visual(
                pose=p,
                half_size=[t, hy, wall_hz],
                material=material,
            )

        # Left/right walls in y.
        for y in (-hy + t, hy - t):
            p = sapien.Pose(p=[0.0, y, wall_z])
            builder.add_box_collision(
                pose=p,
                half_size=[hx - 2 * t, t, wall_hz],
            )
            builder.add_box_visual(
                pose=p,
                half_size=[hx - 2 * t, t, wall_hz],
                material=material,
            )

        builder.initial_pose = sapien.Pose(
            p=[center_xy[0], center_xy[1], 0.0]
        )
        return builder.build_static(name=name)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _load_scene(self, options: dict):
        # Standard ManiSkill tabletop + robot initialization.
        self.table_scene = TableSceneBuilder(
            self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        # -------------------------
        # Sorting receptacles
        # -------------------------
        self.white_bin = self._build_open_bin(
            name="white_bin",
            center_xy=self.WHITE_BIN_CENTER,
            color=[0.95, 0.95, 0.95, 1.0],
        )
        self.pink_bin = self._build_open_bin(
            name="pink_bin",
            center_xy=self.PINK_BIN_CENTER,
            color=[1.00, 0.42, 0.65, 1.0],
        )

        # -------------------------
        # Central graspable objects
        # -------------------------
        # Initial poses are also explicitly restored in _initialize_episode.
        self.objects = []

        self.red_cube = actors.build_cube(
            self.scene,
            half_size=0.025,
            color=[0.90, 0.12, 0.10, 1.0],
            name="red_cube",
            initial_pose=sapien.Pose(p=[0.03, +0.04, 0.025]),
        )
        self.objects.append(self.red_cube)

        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=0.022,
            color=[0.10, 0.30, 0.90, 1.0],
            name="blue_cube",
            initial_pose=sapien.Pose(p=[0.10, -0.05, 0.022]),
        )
        self.objects.append(self.blue_cube)

        self.yellow_block = actors.build_box(
            self.scene,
            half_sizes=[0.035, 0.020, 0.018],
            color=[0.95, 0.75, 0.08, 1.0],
            name="yellow_block",
            initial_pose=sapien.Pose(p=[0.17, +0.06, 0.018]),
        )
        self.objects.append(self.yellow_block)

        self.purple_block = actors.build_box(
            self.scene,
            half_sizes=[0.025, 0.032, 0.020],
            color=[0.55, 0.20, 0.75, 1.0],
            name="purple_block",
            initial_pose=sapien.Pose(p=[0.20, -0.055, 0.020]),
        )
        self.objects.append(self.purple_block)

        self.green_cylinder = actors.build_cylinder(
            self.scene,
            radius=0.024,
            half_length=0.030,
            color=[0.12, 0.70, 0.28, 1.0],
            name="green_cylinder",
            initial_pose=sapien.Pose(p=[0.075, +0.12, 0.030]),
        )
        self.objects.append(self.green_cylinder)

        self.orange_cylinder = actors.build_cylinder(
            self.scene,
            radius=0.021,
            half_length=0.027,
            color=[0.95, 0.38, 0.08, 1.0],
            name="orange_cylinder",
            initial_pose=sapien.Pose(p=[0.145, -0.13, 0.027]),
        )
        self.objects.append(self.orange_cylinder)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Restore a deterministic sorting layout on every reset."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Fixed locations make debugging VLA / external control easier.
            # Each tuple = (x, y, z, yaw).
            configs = [
                (self.red_cube,        0.03,  +0.04, 0.025, +0.15),
                (self.blue_cube,       0.10,  -0.05, 0.022, -0.25),
                (self.yellow_block,    0.17,  +0.06, 0.018, +0.45),
                (self.purple_block,    0.20,  -0.055,0.020, -0.35),
                (self.green_cylinder,  0.075, +0.12, 0.030,  0.00),
                (self.orange_cylinder, 0.145, -0.13, 0.027,  0.00),
            ]

            for actor, x, y, z, yaw in configs:
                p = torch.tensor(
                    [[x, y, z]],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(b, 1)

                # Quaternion [w, x, y, z] for yaw about +z.
                q = torch.tensor(
                    [[
                        np.cos(yaw / 2.0),
                        0.0,
                        0.0,
                        np.sin(yaw / 2.0),
                    ]],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(b, 1)

                actor.set_pose(Pose.create_from_pq(p=p, q=q))

    def evaluate(self):
        # This scene is currently intended as an externally controlled
        # manipulation sandbox rather than a reward-defined RL task.
        return {
            "success": torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        }
