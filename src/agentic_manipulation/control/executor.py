"""Bounded pick/place state machine and ManiSkill xlerobot backend."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from agentic_manipulation.errors import ExecutionError
from agentic_manipulation.types import ExecutionReport, GraspCandidate


class MotionBackend(Protocol):
    def move_ee(self, target_pose: np.ndarray, steps: int, stage: str) -> None: ...

    def set_gripper(self, closed: bool, steps: int, stage: str) -> None: ...

    def is_grasping(self, instance_id: str) -> bool: ...

    def settle(self, steps: int) -> None: ...


def placement_pose(
    bin_inner_aabb: tuple[np.ndarray, np.ndarray],
    object_half_height: float,
    tool_orientation: np.ndarray,
    *,
    clearance: float = 0.01,
) -> np.ndarray:
    low = np.asarray(bin_inner_aabb[0], dtype=np.float64)
    high = np.asarray(bin_inner_aabb[1], dtype=np.float64)
    orientation = np.asarray(tool_orientation, dtype=np.float64)
    if low.shape != (3,) or high.shape != (3,) or np.any(low >= high):
        raise ExecutionError("bin_inner_aabb must contain ordered 3-vectors")
    if orientation.shape != (3, 3) or not np.isfinite(orientation).all():
        raise ExecutionError("tool_orientation must be a finite 3x3 matrix")
    if object_half_height <= 0:
        raise ExecutionError("object_half_height must be positive")
    # The destination is open at the top, so a tall object may safely protrude
    # above the walls. Its center still has to lie within the state-check AABB.
    required_center_height = object_half_height + clearance
    if required_center_height >= high[2] - low[2]:
        raise ExecutionError("object does not fit inside destination bin")
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = orientation
    pose[:2, 3] = ((low[:2] + high[:2]) / 2).astype(np.float32)
    pose[2, 3] = float(low[2] + object_half_height + clearance)
    return pose


class MotionExecutor:
    def __init__(
        self,
        backend: MotionBackend,
        *,
        motion_steps: int = 25,
        gripper_steps: int = 12,
        settle_steps: int = 30,
        approach_distance_m: float = 0.03,
        lift_distance_m: float = 0.03,
        preplace_height_m: float = 0.12,
    ) -> None:
        self.backend = backend
        self.motion_steps = motion_steps
        self.gripper_steps = gripper_steps
        self.settle_steps = settle_steps
        self.approach_distance_m = approach_distance_m
        self.lift_distance_m = lift_distance_m
        self.preplace_height_m = preplace_height_m

    def can_reach(self, pose: np.ndarray) -> bool:
        target = np.asarray(pose)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            return False
        backend_check = getattr(self.backend, "can_reach", None)
        return bool(backend_check(target)) if callable(backend_check) else True

    def execute(
        self,
        instance_id: str,
        grasp: GraspCandidate,
        placement: np.ndarray,
    ) -> ExecutionReport:
        grasp_pose = grasp.world_from_gripper.astype(np.float32, copy=True)
        placement = np.asarray(placement, dtype=np.float32)
        if placement.shape != (4, 4) or not np.isfinite(placement).all():
            raise ExecutionError("placement pose must be a finite 4x4 matrix")
        approach_axis = grasp_pose[:3, 0]
        if float(np.linalg.norm(approach_axis)) < 1e-6:
            raise ExecutionError("grasp approach axis must be nonzero")
        approach_axis = approach_axis / np.linalg.norm(approach_axis)

        pregrasp = grasp_pose.copy()
        pregrasp[:3, 3] -= approach_axis * self.approach_distance_m
        lift = grasp_pose.copy()
        lift[2, 3] += self.lift_distance_m
        preplace = placement.copy()
        preplace[2, 3] += self.preplace_height_m
        retreat = preplace.copy()

        completed: list[str] = []
        self.backend.move_ee(pregrasp, self.motion_steps, "pregrasp")
        completed.append("pregrasp")
        self.backend.move_ee(grasp_pose, self.motion_steps, "approach")
        completed.append("approach")
        self.backend.set_gripper(True, self.gripper_steps, "close")
        completed.append("close")
        if not self.backend.is_grasping(instance_id):
            return ExecutionReport(
                False,
                tuple(completed),
                "target is not held after closing gripper",
            )
        self.backend.move_ee(lift, self.motion_steps, "lift")
        completed.append("lift")
        self.backend.move_ee(preplace, self.motion_steps, "preplace")
        completed.append("preplace")
        self.backend.move_ee(placement, self.motion_steps, "place")
        completed.append("place")
        self.backend.set_gripper(False, self.gripper_steps, "open")
        completed.append("open")
        self.backend.move_ee(retreat, self.motion_steps, "retreat")
        completed.append("retreat")
        self.backend.settle(self.settle_steps)
        completed.append("settle")
        return ExecutionReport(True, tuple(completed))


class ManiSkillXlerobotBackend:
    """Joint-position/IK backend for xlerobot's first arm."""

    control_mode = "pd_joint_pos_dual_arm"
    right_arm_joint_names = (
        "Rotation",
        "Pitch",
        "Elbow",
        "Wrist_Pitch",
        "Wrist_Roll",
    )
    left_arm_joint_names = (
        "Rotation_2",
        "Pitch_2",
        "Elbow_2",
        "Wrist_Pitch_2",
        "Wrist_Roll_2",
    )

    def __init__(self, env: Any) -> None:
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.agent = self.base_env.agent
        if self.base_env.robot_uids != "xlerobot":
            raise ExecutionError("ManiSkillXlerobotBackend requires robot_uids='xlerobot'")
        if self.control_mode not in self.agent.supported_control_modes:
            raise ExecutionError(
                f"xlerobot does not support control mode {self.control_mode}"
            )
        self.agent.set_control_mode(self.control_mode)
        self._joint_indices = {
            joint.name: index
            for index, joint in enumerate(self.agent.robot.active_joints)
        }
        missing = [
            name
            for name in (*self.right_arm_joint_names, *self.left_arm_joint_names)
            if name not in self._joint_indices
        ]
        if missing:
            raise ExecutionError(f"xlerobot joint names are missing: {missing}")
        try:
            import torch
            from mani_skill.agents.controllers.utils.kinematics import Kinematics

            active_indices = torch.tensor(
                [self._joint_indices[name] for name in self.right_arm_joint_names],
                dtype=torch.int64,
                device=self.agent.robot.device,
            )
            self._kinematics = Kinematics(
                self.agent.urdf_path,
                self.agent.ee_link_name,
                self.agent.robot,
                active_indices,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise ExecutionError(f"failed to initialize xlerobot IK: {exc}") from exc
        current_tcp = (
            self.agent.tcp_pose.to_transformation_matrix()[0]
            .detach()
            .cpu()
            .numpy()
        )
        current_end_link = (
            self._kinematics.end_link.pose.to_transformation_matrix()[0]
            .detach()
            .cpu()
            .numpy()
        )
        self._tcp_from_end_link = np.linalg.inv(current_tcp) @ current_end_link
        self._end_link_from_tcp = np.linalg.inv(self._tcp_from_end_link)
        limits = []
        for name in self.right_arm_joint_names:
            joint = self.agent.robot.active_joints[self._joint_indices[name]]
            limits.append(joint.limits[0].detach().cpu().numpy())
        self._right_joint_limits = np.asarray(limits, dtype=np.float64)
        self._right_target = self._current(self.right_arm_joint_names)
        self._right_gripper_closed = False

    def _current(self, names: tuple[str, ...]):
        indices = [self._joint_indices[name] for name in names]
        return self.agent.robot.get_qpos()[:, indices]

    def _controller_action(self, right_arm_target: Any):
        import torch

        controller = self.agent.controller
        qpos = self.agent.robot.get_qpos()
        batch = qpos.shape[0]
        action_dict = {
            "arm1": right_arm_target,
            "arm2": self._current(self.left_arm_joint_names),
            "gripper1": torch.full(
                (batch, 1),
                -1.0 if self._right_gripper_closed else 1.0,
                device=qpos.device,
            ),
            "gripper2": torch.ones((batch, 1), device=qpos.device),
            "body": torch.zeros((batch, 2), device=qpos.device),
            "base": torch.zeros((batch, 2), device=qpos.device),
        }
        return controller.from_action_dict(action_dict)

    def _step(self, action: Any) -> None:
        array = action.detach().cpu().numpy()
        self.env.step(array[0] if len(array) == 1 else array)

    def _end_link_target(self, tcp_target: np.ndarray) -> np.ndarray:
        world_from_end_link = (
            np.asarray(tcp_target, dtype=np.float32) @ self._tcp_from_end_link
        )
        world_from_root = (
            self.agent.robot.root.pose.to_transformation_matrix()[0]
            .detach()
            .cpu()
            .numpy()
        )
        return np.linalg.inv(world_from_root) @ world_from_end_link

    def _position_ik(self, tcp_target: np.ndarray):
        """Solve TCP translation when a 5-DOF arm cannot satisfy a 6D pose."""

        if self._kinematics.use_gpu_ik:
            return None
        import torch

        kinematics = self._kinematics
        qpos = self.agent.robot.get_qpos()
        q_model = (
            qpos[0, kinematics.pmodel_active_joint_indices]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
        )
        controlled = (
            kinematics.pmodel_controlled_joint_indices
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        world_from_root = (
            self.agent.robot.root.pose.to_transformation_matrix()[0]
            .detach()
            .cpu()
            .numpy()
        )
        target_world = np.ones(4, dtype=np.float64)
        target_world[:3] = np.asarray(tcp_target, dtype=np.float64)[:3, 3]
        target_root = (np.linalg.inv(world_from_root) @ target_world)[:3]

        def forward_position(q_value: np.ndarray) -> np.ndarray:
            kinematics.pmodel.compute_forward_kinematics(q_value)
            root_from_end = kinematics.pmodel.get_link_pose(
                kinematics.end_link_idx
            ).to_transformation_matrix()
            root_from_tcp = root_from_end @ self._end_link_from_tcp
            return np.asarray(root_from_tcp[:3, 3], dtype=np.float64)

        epsilon = 1e-4
        for _ in range(120):
            position = forward_position(q_model)
            error = target_root - position
            if float(np.linalg.norm(error)) <= 0.003:
                return torch.as_tensor(
                    q_model[controlled][None],
                    dtype=torch.float32,
                    device=self.agent.robot.device,
                )
            jacobian = np.zeros((3, len(controlled)), dtype=np.float64)
            for column, joint_index in enumerate(controlled):
                perturbed = q_model.copy()
                perturbed[joint_index] += epsilon
                jacobian[:, column] = (
                    forward_position(perturbed) - position
                ) / epsilon
            try:
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + 1e-3 * np.eye(3), error
                )
            except np.linalg.LinAlgError:
                return None
            step_scale = max(1.0, float(np.max(np.abs(delta))) / 0.15)
            next_controlled = q_model[controlled] + delta / step_scale
            q_model[controlled] = np.clip(
                next_controlled,
                self._right_joint_limits[:, 0],
                self._right_joint_limits[:, 1],
            )
        if float(np.linalg.norm(target_root - forward_position(q_model))) <= 0.005:
            return torch.as_tensor(
                q_model[controlled][None],
                dtype=torch.float32,
                device=self.agent.robot.device,
            )
        return None

    def _solve_ik(self, tcp_target: np.ndarray):
        import sapien
        from mani_skill.utils.structs.pose import Pose

        end_link_target = self._end_link_target(tcp_target)
        result = self._kinematics.compute_ik(
            Pose.create(
                sapien.Pose(end_link_target), device=self.agent.robot.device
            ),
            self.agent.robot.get_qpos(),
        )
        return result if result is not None else self._position_ik(tcp_target)

    def can_reach(self, target_pose: np.ndarray) -> bool:
        target = np.asarray(target_pose, dtype=np.float32)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            return False
        try:
            result = self._solve_ik(target)
        except (RuntimeError, ValueError):
            return False
        return result is not None

    def move_ee(self, target_pose: np.ndarray, steps: int, stage: str) -> None:
        target = np.asarray(target_pose, dtype=np.float32)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ExecutionError(f"{stage} target pose must be a finite 4x4 matrix")
        target_qpos = self._solve_ik(target)
        if target_qpos is None:
            raise ExecutionError(f"xlerobot IK failed during {stage}")
        start = self._current(self.right_arm_joint_names)
        for index in range(1, steps + 1):
            fraction = index / steps
            interpolated = start + (target_qpos - start) * fraction
            self._step(self._controller_action(interpolated))
        self._right_target = target_qpos

    def set_gripper(self, closed: bool, steps: int, stage: str) -> None:
        self._right_gripper_closed = closed
        for _ in range(steps):
            self._step(self._controller_action(self._right_target))

    def is_grasping(self, instance_id: str) -> bool:
        try:
            actor = self.base_env.semantic_actors[instance_id]
        except KeyError as exc:
            raise ExecutionError(f"unknown semantic actor: {instance_id}") from exc
        value = self.agent.is_grasping(actor, arm_id=1)
        return bool(value[0].item())

    def settle(self, steps: int) -> None:
        for _ in range(steps):
            self._step(self._controller_action(self._right_target))
