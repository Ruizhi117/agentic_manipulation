"""Atomic Panda pick/place motions driven by ``pd_ee_delta_pose``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from agentic_manipulation.errors import ExecutionError


class PandaMotionBackend(Protocol):
    """Small backend boundary used by the reusable Panda atomic skill."""

    home_pose: np.ndarray

    def move_ee(self, target_pose: np.ndarray, steps: int, stage: str) -> None: ...

    def set_gripper(self, value: float, steps: int, stage: str) -> None: ...

    def is_grasping(self, instance_id: str) -> bool: ...

    def bin_inner_aabb(self, bin_id: str) -> tuple[np.ndarray, np.ndarray]: ...

    def object_half_height(self, instance_id: str) -> float: ...

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

    def pick(self, instance_id: str, world_from_ee: object) -> AtomicPickReport:
        grasp = _finite_pose(world_from_ee, "world_from_ee").copy()
        grasp[:3, 3] += grasp[:3, 2] * self.grasp_depth_offset_m
        pregrasp = grasp.copy()
        pregrasp[:3, 3] -= grasp[:3, 2] * self.pregrasp_height_m
        lift = grasp.copy()
        lift[2, 3] += self.lift_height_m

        completed = ["home_observe"]
        self.backend.set_gripper(1.0, self.gripper_steps, "open")
        completed.append("open")
        self.backend.move_ee(pregrasp, self.motion_steps, "pregrasp")
        completed.append("pregrasp")
        self.backend.move_ee(grasp, self.motion_steps, "approach")
        completed.append("approach")
        self.backend.set_gripper(-1.0, self.gripper_steps, "close")
        completed.append("close")
        if not self.backend.is_grasping(instance_id):
            return AtomicPickReport(
                False,
                tuple(completed),
                "target is not held after closing gripper",
            )
        self.backend.move_ee(lift, self.motion_steps, "lift")
        completed.append("lift")
        self._tool_orientation = grasp[:3, :3].copy()
        return AtomicPickReport(True, tuple(completed))

    def place(self, instance_id: str, bin_id: str) -> AtomicPlaceReport:
        low_raw, high_raw = self.backend.bin_inner_aabb(bin_id)
        low = np.asarray(low_raw, dtype=np.float64)
        high = np.asarray(high_raw, dtype=np.float64)
        if (
            low.shape != (3,)
            or high.shape != (3,)
            or not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or np.any(low >= high)
        ):
            raise ExecutionError("bin_inner_aabb must contain ordered finite 3-vectors")
        half_height = float(self.backend.object_half_height(instance_id))
        if not np.isfinite(half_height) or half_height <= 0:
            raise ExecutionError("object_half_height must be positive and finite")

        placement = np.eye(4, dtype=np.float64)
        if self._tool_orientation is not None:
            placement[:3, :3] = self._tool_orientation
        else:
            placement[:3, :3] = np.diag([1.0, -1.0, -1.0])
        placement[:2, 3] = (low[:2] + high[:2]) / 2.0
        placement[2, 3] = low[2] + half_height + self.placement_clearance_m
        slot_planner = getattr(self.backend, "placement_object_center", None)
        if callable(slot_planner):
            center = np.asarray(
                slot_planner(instance_id, bin_id, placement[:3, 3].copy()),
                dtype=np.float64,
            )
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ExecutionError(
                    "placement_object_center must return a finite 3-vector"
                )
            placement[:3, 3] = center
        offset_planner = getattr(self.backend, "placement_ee_pose", None)
        if callable(offset_planner):
            placement = _finite_pose(
                offset_planner(instance_id, bin_id, placement[:3, 3].copy()),
                "offset-aware placement pose",
            )
        preplace = placement.copy()
        preplace[2, 3] += self.preplace_height_m

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
        self.backend.move_ee(home, self.motion_steps * 2, "home_return")

    def recover_after_failed_pick(self) -> None:
        """Open the gripper before returning to the observation-home pose."""

        self.backend.set_gripper(1.0, self.gripper_steps, "open_recover")
        self.return_home()


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
        for field, value in (
            ("placement_xy_tolerance_m", placement_xy_tolerance_m),
            ("placement_vertical_tolerance_m", placement_vertical_tolerance_m),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ExecutionError(f"{field} must be positive and finite")
        self.translation_tolerance_m = float(translation_tolerance_m)
        self.rotation_tolerance_rad = float(rotation_tolerance_rad)
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
        translation_reached = translation_error <= self.translation_tolerance_m
        if stage == "preplace":
            translation_reached = (
                translation_error <= self.placement_xy_tolerance_m
            )
        elif stage == "place":
            translation_reached = (
                xy_error <= self.placement_xy_tolerance_m
                and -self.translation_tolerance_m
                <= vertical_residual
                <= self.placement_vertical_tolerance_m
            )
        allow_pick_rotation_residual = stage in {"pregrasp", "approach"}
        if not translation_reached or (
            not allow_pick_rotation_residual
            and rotation_error > self.rotation_tolerance_rad
        ):
            raise ExecutionError(
                f"{stage} failed to reach target: translation error "
                f"{translation_error:.6f} m (xy {xy_error:.6f} m, "
                f"z residual {vertical_residual:.6f} m), rotation error "
                f"{rotation_error:.6f} rad"
            )

    def set_gripper(self, value: float, steps: int, stage: str) -> None:
        if value not in {-1.0, 1.0}:
            raise ExecutionError("Panda gripper value must be -1 (close) or 1 (open)")
        self.gripper_value = float(value)
        for _ in range(_positive_int(steps, "steps")):
            action = np.zeros(7, dtype=np.float32)
            action[6] = self.gripper_value
            self._step(action, stage)

    def is_grasping(self, instance_id: str) -> bool:
        try:
            actor = self.base_env.semantic_actors[instance_id]
        except KeyError as exc:
            raise ExecutionError(f"unknown semantic actor: {instance_id}") from exc
        value = self.agent.is_grasping(actor)
        if hasattr(value, "detach"):
            return bool(value[0].detach().cpu().item())
        return bool(np.asarray(value).reshape(-1)[0])

    def bin_inner_aabb(self, bin_id: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            low, high = self.base_env.bin_inner_aabbs[bin_id]
        except KeyError as exc:
            raise ExecutionError(f"unknown destination bin: {bin_id}") from exc
        return np.asarray(low).copy(), np.asarray(high).copy()

    def object_half_height(self, instance_id: str) -> float:
        try:
            return float(self.base_env.object_half_heights[instance_id])
        except KeyError as exc:
            raise ExecutionError(f"unknown graspable object: {instance_id}") from exc

    def placement_ee_pose(
        self,
        instance_id: str,
        bin_id: str,
        desired_object_center: np.ndarray,
    ) -> np.ndarray:
        del bin_id
        desired = np.asarray(desired_object_center, dtype=np.float64)
        if desired.shape != (3,) or not np.isfinite(desired).all():
            raise ExecutionError("desired_object_center must be a finite 3-vector")
        try:
            actor_center = self.base_env.semantic_actors[instance_id].pose.p[0]
        except KeyError as exc:
            raise ExecutionError(f"unknown graspable object: {instance_id}") from exc
        if hasattr(actor_center, "detach"):
            actor_center = actor_center.detach().cpu().numpy()
        current_center = np.asarray(actor_center, dtype=np.float64)
        target = _as_pose_matrix(self.agent.tcp_pose)
        target[:3, 3] += desired - current_center
        return target

    def placement_object_center(
        self,
        instance_id: str,
        bin_id: str,
        default_center: np.ndarray,
    ) -> np.ndarray:
        """Choose a separated XY slot when the destination already has objects."""

        desired = np.asarray(default_center, dtype=np.float64)
        if desired.shape != (3,) or not np.isfinite(desired).all():
            raise ExecutionError("default_center must be a finite 3-vector")
        low, high = self.bin_inner_aabb(bin_id)
        span_xy = high[:2] - low[:2]
        bin_center_xy = (low[:2] + high[:2]) / 2.0
        offsets = np.asarray(
            [
                [0.0, 0.0],
                [-0.28, -0.28],
                [0.28, -0.28],
                [-0.28, 0.28],
                [0.28, 0.28],
            ],
            dtype=np.float64,
        )
        slots = bin_center_xy + offsets * span_xy
        occupied: list[np.ndarray] = []
        for other_id, actor in self.base_env.semantic_actors.items():
            if other_id in {instance_id, bin_id}:
                continue
            center = actor.pose.p[0]
            if hasattr(center, "detach"):
                center = center.detach().cpu().numpy()
            point = np.asarray(center, dtype=np.float64)
            if (
                point.shape == (3,)
                and np.all(point[:2] >= low[:2])
                and np.all(point[:2] <= high[:2])
                and point[2] >= low[2]
            ):
                occupied.append(point[:2])
        result = desired.copy()
        if not occupied:
            result[:2] = slots[0]
            return result
        occupied_xy = np.asarray(occupied)
        clearances = np.min(
            np.linalg.norm(slots[:, None, :] - occupied_xy[None, :, :], axis=2),
            axis=1,
        )
        result[:2] = slots[int(np.argmax(clearances))]
        return result

    def settle(self, steps: int) -> None:
        for _ in range(_positive_int(steps, "steps")):
            action = np.zeros(7, dtype=np.float32)
            action[6] = self.gripper_value
            self._step(action, "settle")
