"""Atomic Panda pick/place motions driven by ``pd_ee_delta_pose``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from agentic_manipulation.errors import ExecutionError, MotionStageError


class PandaMotionBackend(Protocol):
    """Small backend boundary used by the reusable Panda atomic skill."""

    home_pose: np.ndarray

    def can_reach(self, target_pose: np.ndarray) -> bool: ...

    def move_ee(self, target_pose: np.ndarray, steps: int, stage: str) -> None: ...

    def set_gripper(self, value: float, steps: int, stage: str) -> None: ...

    def settle(self, steps: int) -> None: ...


@dataclass(frozen=True)
class AtomicPickReport:
    success: bool
    stages: tuple[str, ...]
    failure_reason: str | None = None


@dataclass(frozen=True)
class AtomicPlaceReport:
    success: bool
    stages: tuple[str, ...]
    failure_reason: str | None = None


@dataclass(frozen=True)
class AtomicTransferPlan:
    """Six prevalidated TCP waypoints frozen before robot motion begins."""

    pregrasp: np.ndarray
    grasp: np.ndarray
    lift: np.ndarray
    preplace: np.ndarray
    release: np.ndarray
    home: np.ndarray

    def __post_init__(self) -> None:
        for field in (
            "pregrasp",
            "grasp",
            "lift",
            "preplace",
            "release",
            "home",
        ):
            value = _finite_pose(getattr(self, field), field)
            value.setflags(write=False)
            object.__setattr__(self, field, value)

    @property
    def waypoints(self) -> tuple[np.ndarray, ...]:
        return (
            self.pregrasp,
            self.grasp,
            self.lift,
            self.preplace,
            self.release,
            self.home,
        )


@dataclass(frozen=True)
class AtomicTransferReport:
    success: bool
    stages: tuple[str, ...]
    failure_reason: str | None = None


def _finite_pose(value: object, field: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ExecutionError(f"{field} must be a finite 4x4 matrix") from exc
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ExecutionError(f"{field} must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ExecutionError(f"{field} must be a finite 4x4 matrix")
    return pose.copy()


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ExecutionError(f"{field} must be a positive integer")
    return int(value)


class PandaAtomicPickPlaceSkill:
    """Bounded top-down Panda pick and selected-bin placement sequence."""

    def __init__(
        self,
        backend: PandaMotionBackend,
        *,
        motion_steps: int = 25,
        gripper_steps: int = 12,
        settle_steps: int = 30,
        pregrasp_height_m: float = 0.08,
        lift_height_m: float = 0.10,
        preplace_height_m: float = 0.10,
        placement_clearance_m: float = 0.01,
        grasp_depth_offset_m: float = 0.01,
    ) -> None:
        self.backend = backend
        self.motion_steps = _positive_int(motion_steps, "motion_steps")
        self.gripper_steps = _positive_int(gripper_steps, "gripper_steps")
        self.settle_steps = _positive_int(settle_steps, "settle_steps")
        for field, value in (
            ("pregrasp_height_m", pregrasp_height_m),
            ("lift_height_m", lift_height_m),
            ("preplace_height_m", preplace_height_m),
            ("placement_clearance_m", placement_clearance_m),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ExecutionError(f"{field} must be positive and finite")
        self.pregrasp_height_m = float(pregrasp_height_m)
        self.lift_height_m = float(lift_height_m)
        self.preplace_height_m = float(preplace_height_m)
        self.placement_clearance_m = float(placement_clearance_m)
        if not np.isfinite(grasp_depth_offset_m) or grasp_depth_offset_m < 0:
            raise ExecutionError(
                "grasp_depth_offset_m must be non-negative and finite"
            )
        self.grasp_depth_offset_m = float(grasp_depth_offset_m)
        self._tool_orientation: np.ndarray | None = None

    def _pick_waypoints(
        self, world_from_ee: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grasp = _finite_pose(world_from_ee, "world_from_ee").copy()
        grasp[:3, 3] += grasp[:3, 2] * self.grasp_depth_offset_m
        pregrasp = grasp.copy()
        pregrasp[:3, 3] -= grasp[:3, 2] * self.pregrasp_height_m
        lift = grasp.copy()
        lift[2, 3] += self.lift_height_m
        return pregrasp, grasp, lift

    def _place_waypoints(
        self,
        world_position: object,
        tool_orientation: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            position = np.asarray(world_position, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ExecutionError(
                "world_position must be a finite 3-vector"
            ) from exc
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ExecutionError("world_position must be a finite 3-vector")

        if tool_orientation is None:
            rotation = (
                np.diag([1.0, -1.0, -1.0])
                if self._tool_orientation is None
                else self._tool_orientation
            )
        else:
            try:
                rotation = np.asarray(tool_orientation, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ExecutionError(
                    "tool_orientation must be a finite 3x3 rotation"
                ) from exc
            if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
                raise ExecutionError(
                    "tool_orientation must be a finite 3x3 rotation"
                )

        placement = np.eye(4, dtype=np.float64)
        placement[:3, :3] = rotation
        placement[:3, 3] = position
        preplace = placement.copy()
        preplace[2, 3] += self.preplace_height_m
        return preplace, placement

    def can_pick(self, world_from_ee: object) -> bool:
        """Return whether pregrasp, grasp, and lift all have Panda IK solutions."""

        return all(
            self.backend.can_reach(pose)
            for pose in self._pick_waypoints(world_from_ee)
        )

    def can_place(
        self,
        world_position: object,
        tool_orientation: object | None = None,
    ) -> bool:
        """Return whether preplace and placement both have Panda IK solutions."""

        return all(
            self.backend.can_reach(pose)
            for pose in self._place_waypoints(world_position, tool_orientation)
        )

    def plan_transfer(
        self,
        world_from_ee: object,
        nominal_world_from_release: object,
    ) -> AtomicTransferPlan:
        """Freeze and IK-check the complete grasp-to-release motion plan."""

        nominal_grasp = _finite_pose(world_from_ee, "world_from_ee")
        pregrasp, grasp, lift = self._pick_waypoints(nominal_grasp)
        release = _finite_pose(
            nominal_world_from_release, "nominal_world_from_release"
        )
        # The grasp-depth correction changes the actual TCP/object offset. Apply
        # the same correction to release so the planned object center is preserved.
        release[:3, 3] += grasp[:3, 3] - nominal_grasp[:3, 3]
        preplace = release.copy()
        preplace[2, 3] += self.preplace_height_m
        home = _finite_pose(self.backend.home_pose, "home_pose")
        plan = AtomicTransferPlan(
            pregrasp, grasp, lift, preplace, release, home
        )
        if not all(self.backend.can_reach(pose) for pose in plan.waypoints):
            raise ExecutionError("IK-unreachable atomic transfer trajectory")
        return plan

    def execute(
        self,
        instance_id: str,
        plan: AtomicTransferPlan,
        confirm_grasp: Callable[[], bool],
    ) -> AtomicTransferReport:
        """Execute one preplanned pick-check-transfer-release transaction."""

        del instance_id
        if not isinstance(plan, AtomicTransferPlan):
            raise ExecutionError("plan must be an AtomicTransferPlan")
        if not callable(confirm_grasp):
            raise ExecutionError("confirm_grasp must be callable")
        if not all(self.backend.can_reach(pose) for pose in plan.waypoints):
            raise ExecutionError("IK-unreachable atomic transfer trajectory")

        completed = ["home_observe"]
        self.backend.set_gripper(1.0, self.gripper_steps, "open")
        completed.append("open")
        self.backend.move_ee(plan.pregrasp, self.motion_steps, "pregrasp")
        completed.append("pregrasp")
        self.backend.move_ee(plan.grasp, self.motion_steps, "approach")
        completed.append("approach")
        self.backend.set_gripper(-1.0, self.gripper_steps, "close")
        completed.append("close")
        self.backend.move_ee(plan.lift, self.motion_steps, "lift")
        completed.append("lift")
        held = confirm_grasp()
        if not isinstance(held, bool):
            raise ExecutionError("confirm_grasp must return a boolean")
        completed.append("grasp_check")
        if not held:
            self.backend.set_gripper(
                1.0, self.gripper_steps, "open_recover"
            )
            completed.append("open_recover")
            self._move_home(plan.home)
            completed.append("home_return")
            self.backend.settle(self.settle_steps)
            completed.append("settle")
            return AtomicTransferReport(
                False,
                tuple(completed),
                "VLM did not confirm a held object",
            )

        self.backend.move_ee(plan.preplace, self.motion_steps, "preplace")
        completed.append("preplace")
        self.backend.move_ee(plan.release, self.motion_steps, "place")
        completed.append("place")
        self.backend.set_gripper(
            1.0, self.gripper_steps * 2, "open_release"
        )
        completed.append("open_release")
        self.backend.move_ee(plan.preplace, self.motion_steps, "retreat")
        completed.append("retreat")
        self._move_home(plan.home)
        completed.append("home_return")
        self.backend.settle(self.settle_steps)
        completed.append("settle")
        return AtomicTransferReport(True, tuple(completed))

    def pick(self, instance_id: str, world_from_ee: object) -> AtomicPickReport:
        del instance_id
        pregrasp, grasp, lift = self._pick_waypoints(world_from_ee)
        if not all(
            self.backend.can_reach(pose) for pose in (pregrasp, grasp, lift)
        ):
            raise ExecutionError("IK-unreachable pick trajectory")

        completed = ["home_observe"]
        self.backend.set_gripper(1.0, self.gripper_steps, "open")
        completed.append("open")
        self.backend.move_ee(pregrasp, self.motion_steps, "pregrasp")
        completed.append("pregrasp")
        self.backend.move_ee(grasp, self.motion_steps, "approach")
        completed.append("approach")
        self.backend.set_gripper(-1.0, self.gripper_steps, "close")
        completed.append("close")
        self.backend.move_ee(lift, self.motion_steps, "lift")
        completed.append("lift")
        self._tool_orientation = grasp[:3, :3].copy()
        return AtomicPickReport(True, tuple(completed))

    def place(self, instance_id: str, world_position: object) -> AtomicPlaceReport:
        del instance_id
        preplace, placement = self._place_waypoints(world_position)
        if not all(
            self.backend.can_reach(pose) for pose in (preplace, placement)
        ):
            raise ExecutionError("IK-unreachable place trajectory")

        completed: list[str] = []
        self.backend.move_ee(preplace, self.motion_steps, "preplace")
        completed.append("preplace")
        self.backend.move_ee(placement, self.motion_steps, "place")
        completed.append("place")
        self.backend.set_gripper(1.0, self.gripper_steps * 2, "open_release")
        completed.append("open_release")
        self.backend.move_ee(preplace, self.motion_steps, "retreat")
        completed.append("retreat")
        self.return_home()
        completed.append("home_return")
        self.backend.settle(self.settle_steps)
        completed.append("settle")
        return AtomicPlaceReport(True, tuple(completed))

    def return_home(self) -> None:
        home = _finite_pose(self.backend.home_pose, "home_pose")
        self._move_home(home)

    def _move_home(self, home: np.ndarray) -> None:
        self.backend.move_ee(home, self.motion_steps * 2, "home_return")

    def recover_after_failed_pick(self) -> None:
        """Open the gripper before returning to the observation-home pose."""

        self.backend.set_gripper(1.0, self.gripper_steps, "open_recover")
        self.return_home()
        self.backend.settle(self.settle_steps)


def _as_pose_matrix(pose: object) -> np.ndarray:
    value: Any
    if hasattr(pose, "to_transformation_matrix"):
        value = pose.to_transformation_matrix()
    elif hasattr(pose, "raw_pose"):
        value = pose.raw_pose
    else:
        value = pose
    if hasattr(value, "detach"):
        value = value[0].detach().cpu().numpy()
    else:
        value = np.asarray(value)
        if value.shape == (1, 4, 4):
            value = value[0]
    return _finite_pose(value, "TCP pose")


def _euler_xyz(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rotation).as_euler("XYZ")


def normalized_delta_pose_action(
    current_pose: object,
    target_pose: object,
    *,
    gripper: float,
    position_scale: object,
    rotation_scale: object,
) -> np.ndarray:
    """Encode a metric pose delta for ManiSkill's normalized Panda controller."""

    current = _finite_pose(current_pose, "current_pose")
    target = _finite_pose(target_pose, "target_pose")
    position = np.broadcast_to(
        np.asarray(position_scale, dtype=np.float64), (3,)
    )
    rotation = np.broadcast_to(
        np.asarray(rotation_scale, dtype=np.float64), (3,)
    )
    if (
        not np.isfinite(position).all()
        or not np.isfinite(rotation).all()
        or np.any(position == 0)
        or np.any(rotation == 0)
    ):
        raise ExecutionError("controller action scales must be finite and nonzero")
    if not np.isfinite(gripper):
        raise ExecutionError("gripper must be finite")
    relative_rotation = target[:3, :3] @ current[:3, :3].T
    action = np.empty(7, dtype=np.float32)
    action[:3] = (target[:3, 3] - current[:3, 3]) / position
    action[3:6] = _euler_xyz(relative_rotation) / rotation
    action[:6] = np.clip(action[:6], -1.0, 1.0)
    action[6] = float(np.clip(gripper, -1.0, 1.0))
    return action


def _interpolate_pose(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    fraction = float(np.clip(fraction, 0.0, 1.0))
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = start[:3, 3] + fraction * (target[:3, 3] - start[:3, 3])
    rotations = Rotation.from_matrix([start[:3, :3], target[:3, :3]])
    result[:3, :3] = Slerp([0.0, 1.0], rotations)([fraction]).as_matrix()[0]
    return result


class PandaDeltaPoseBackend:
    """Real ManiSkill Panda backend with numeric open/close semantics."""

    control_mode = "pd_ee_delta_pose"

    def __init__(
        self,
        env: Any,
        recorder: Any | None = None,
        *,
        translation_tolerance_m: float = 0.01,
        rotation_tolerance_rad: float = 0.05,
        transport_tool_axis_tolerance_rad: float = 0.15,
        placement_xy_tolerance_m: float = 0.02,
        placement_vertical_tolerance_m: float = 0.04,
        render_callback: Callable[[], object] | None = None,
    ) -> None:
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.agent = self.base_env.agent
        if self.base_env.robot_uids not in {"panda", "panda_wristcam"}:
            raise ExecutionError("PandaDeltaPoseBackend requires a Panda robot")
        if self.control_mode not in self.agent.supported_control_modes:
            raise ExecutionError(f"Panda does not support {self.control_mode}")
        self.agent.set_control_mode(self.control_mode)
        self.home_pose = _finite_pose(
            self.base_env.observation_home_pose, "observation_home_pose"
        )
        self.recorder = recorder
        self.render_callback = render_callback
        self.gripper_value = 1.0
        if not np.isfinite(translation_tolerance_m) or translation_tolerance_m <= 0:
            raise ExecutionError("translation_tolerance_m must be positive and finite")
        if not np.isfinite(rotation_tolerance_rad) or rotation_tolerance_rad <= 0:
            raise ExecutionError("rotation_tolerance_rad must be positive and finite")
        if (
            not np.isfinite(transport_tool_axis_tolerance_rad)
            or transport_tool_axis_tolerance_rad <= 0
        ):
            raise ExecutionError(
                "transport_tool_axis_tolerance_rad must be positive and finite"
            )
        for field, value in (
            ("placement_xy_tolerance_m", placement_xy_tolerance_m),
            ("placement_vertical_tolerance_m", placement_vertical_tolerance_m),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ExecutionError(f"{field} must be positive and finite")
        self.translation_tolerance_m = float(translation_tolerance_m)
        self.rotation_tolerance_rad = float(rotation_tolerance_rad)
        self.transport_tool_axis_tolerance_rad = float(
            transport_tool_axis_tolerance_rad
        )
        self.placement_xy_tolerance_m = float(placement_xy_tolerance_m)
        self.placement_vertical_tolerance_m = float(
            placement_vertical_tolerance_m
        )
        try:
            arm_config = self.agent.controller.controllers["arm"].config
            self._position_scale = np.broadcast_to(
                np.asarray(arm_config.pos_upper, dtype=np.float64), (3,)
            ).copy()
            # ManiSkill 3.0.1 multiplies the normalized rotation action by
            # rot_lower, which is -0.1 for Panda pd_ee_delta_pose.
            self._rotation_scale = np.broadcast_to(
                np.asarray(arm_config.rot_lower, dtype=np.float64), (3,)
            ).copy()
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ExecutionError(
                f"failed to read Panda delta-pose controller scales: {exc}"
            ) from exc

    def _capture(self, stage: str) -> None:
        if self.recorder is None:
            return
        observation = self.base_env.get_obs()
        self.recorder.capture(stage, observation)
        self.recorder.record_motion(
            stage, _as_pose_matrix(self.agent.tcp_pose), self.gripper_value
        )

    def _step(self, action: np.ndarray, stage: str) -> None:
        self.env.step(action)
        self._capture(stage)
        if self.render_callback is not None:
            self.render_callback()

    def can_reach(self, target_pose: np.ndarray) -> bool:
        """Check Panda IK using only its current qpos, root pose, and URDF model."""

        try:
            import sapien
            from mani_skill.utils.structs.pose import Pose

            target = _finite_pose(target_pose, "IK target pose")
            world_from_root = _as_pose_matrix(self.agent.robot.root.pose)
            root_from_target = np.linalg.inv(world_from_root) @ target
            arm_controller = self.agent.controller.controllers["arm"]
            result = arm_controller.kinematics.compute_ik(
                pose=Pose.create(
                    sapien.Pose(root_from_target),
                    device=self.agent.robot.device,
                ),
                q0=self.agent.robot.get_qpos(),
            )
        except (
            AttributeError,
            ExecutionError,
            ImportError,
            KeyError,
            np.linalg.LinAlgError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return False
        if result is None:
            return False
        if hasattr(result, "detach"):
            value = result.detach().cpu().numpy()
        else:
            value = np.asarray(result)
        return bool(value.size and np.isfinite(value).all())

    def move_ee(self, target_pose: np.ndarray, steps: int, stage: str) -> None:
        target = _finite_pose(target_pose, f"{stage} target pose")
        count = _positive_int(steps, "steps")
        start = _as_pose_matrix(self.agent.tcp_pose)
        for index in range(1, count + 1):
            waypoint = _interpolate_pose(start, target, index / count)
            current = _as_pose_matrix(self.agent.tcp_pose)
            action = normalized_delta_pose_action(
                current,
                waypoint,
                gripper=self.gripper_value,
                position_scale=self._position_scale,
                rotation_scale=self._rotation_scale,
            )
            self._step(action, stage)
        actual = _as_pose_matrix(self.agent.tcp_pose)
        translation_delta = target[:3, 3] - actual[:3, 3]
        translation_error = float(np.linalg.norm(translation_delta))
        xy_error = float(np.linalg.norm(translation_delta[:2]))
        vertical_residual = float(actual[2, 3] - target[2, 3])
        rotation_error = float(
            np.linalg.norm(
                _euler_xyz(target[:3, :3] @ actual[:3, :3].T)
            )
        )
        target_tool_axis = target[:3, 2]
        actual_tool_axis = actual[:3, 2]
        tool_axis_error = float(
            np.arccos(
                np.clip(
                    np.dot(target_tool_axis, actual_tool_axis)
                    / (
                        np.linalg.norm(target_tool_axis)
                        * np.linalg.norm(actual_tool_axis)
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        translation_reached = translation_error <= self.translation_tolerance_m
        if stage == "preplace":
            translation_reached = (
                translation_error <= self.placement_xy_tolerance_m
            )
        elif stage == "place":
            translation_reached = (
                xy_error
                <= min(
                    self.placement_xy_tolerance_m,
                    self.translation_tolerance_m,
                )
                and abs(vertical_residual)
                <= min(
                    self.placement_vertical_tolerance_m,
                    self.translation_tolerance_m,
                )
            )
        allow_pick_rotation_residual = stage in {"pregrasp", "approach"}
        vertical_transport_stage = stage in {
            "lift",
            "preplace",
            "place",
            "retreat",
        }
        orientation_error = (
            tool_axis_error if vertical_transport_stage else rotation_error
        )
        orientation_tolerance = (
            self.transport_tool_axis_tolerance_rad
            if vertical_transport_stage
            else self.rotation_tolerance_rad
        )
        if not translation_reached or (
            not allow_pick_rotation_residual
            and orientation_error > orientation_tolerance
        ):
            raise MotionStageError(
                stage,
                f"{stage} failed to reach target: translation error "
                f"{translation_error:.6f} m (xy {xy_error:.6f} m, "
                f"z residual {vertical_residual:.6f} m), rotation error "
                f"{rotation_error:.6f} rad, tool-axis error "
                f"{tool_axis_error:.6f} rad"
            )

    def set_gripper(self, value: float, steps: int, stage: str) -> None:
        if value not in {-1.0, 1.0}:
            raise ExecutionError("Panda gripper value must be -1 (close) or 1 (open)")
        self.gripper_value = float(value)
        for _ in range(_positive_int(steps, "steps")):
            action = np.zeros(7, dtype=np.float32)
            action[6] = self.gripper_value
            self._step(action, stage)

    def settle(self, steps: int) -> None:
        for _ in range(_positive_int(steps, "steps")):
            action = np.zeros(7, dtype=np.float32)
            action[6] = self.gripper_value
            self._step(action, "settle")
