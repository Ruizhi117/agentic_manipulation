"""Versioned transform from the GraspNet gripper frame to Panda TCP."""

from __future__ import annotations

import numpy as np

from agentic_manipulation.demo.grasp_component import compose_world_ee
from agentic_manipulation.demo.protocol import homogeneous_matrix
from agentic_manipulation.errors import ConfigurationError, GraspNetUnavailableError
from agentic_manipulation.types import GraspCandidate


PANDA_GRASP_FROM_EE = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def compose_panda_world_ee(
    world_from_camera: object,
    camera_from_grasp: object,
    calibration: object = PANDA_GRASP_FROM_EE,
) -> np.ndarray:
    """Return a validated Panda TCP pose from a camera-frame grasp pose."""

    grasp_from_ee = homogeneous_matrix(calibration, "grasp_from_ee")
    return compose_world_ee(
        world_from_camera, camera_from_grasp, grasp_from_ee
    )


def calibrated_camera_from_grasp(
    world_from_camera: object,
    camera_grasp_position: object,
    desired_world_ee_rotation: object,
    calibration: object = PANDA_GRASP_FROM_EE,
) -> np.ndarray:
    """Construct a camera-frame grasp that yields the requested Panda TCP rotation.

    This is used by the deterministic motion safety gate. A top-down rotation
    cannot be hard-coded in camera coordinates because the wrist camera itself
    rotates with the robot.
    """

    world_camera = homogeneous_matrix(world_from_camera, "world_from_camera")
    grasp_ee = homogeneous_matrix(calibration, "grasp_from_ee")
    try:
        position = np.asarray(camera_grasp_position, dtype=np.float64)
        desired = np.asarray(desired_world_ee_rotation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("calibrated grasp inputs must be numeric") from exc
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ConfigurationError("camera_grasp_position must be a finite 3-vector")
    if desired.shape != (3, 3) or not np.isfinite(desired).all():
        raise ConfigurationError("desired_world_ee_rotation must be a finite 3x3 matrix")
    if not np.allclose(desired.T @ desired, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(desired), 1.0, atol=1e-6
    ):
        raise ConfigurationError("desired_world_ee_rotation must be a proper rotation")

    world_grasp_rotation = desired @ grasp_ee[:3, :3].T
    camera_grasp_rotation = world_camera[:3, :3].T @ world_grasp_rotation
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = camera_grasp_rotation
    result[:3, 3] = position
    return result


class CalibratedTopDownPandaGraspProvider:
    """Deterministic safety provider with camera-aware Panda TCP rotation."""

    provider_name = "deterministic-calibrated-top-down"

    def __init__(
        self,
        world_from_camera: object,
        desired_world_ee_rotation: object,
    ) -> None:
        self.world_from_camera = homogeneous_matrix(
            world_from_camera, "world_from_camera"
        )
        self.desired_world_ee_rotation = np.asarray(
            desired_world_ee_rotation, dtype=np.float64
        )
        # Reuse full validation, including proper-rotation checks.
        calibrated_camera_from_grasp(
            self.world_from_camera,
            np.zeros(3),
            self.desired_world_ee_rotation,
        )

    def predict(
        self, target_points: np.ndarray, workspace_points: np.ndarray
    ) -> tuple[GraspCandidate, ...]:
        target = np.asarray(target_points, dtype=np.float32)
        workspace = np.asarray(workspace_points, dtype=np.float32)
        if (
            target.ndim != 2
            or target.shape[1:] != (3,)
            or len(target) == 0
            or not np.isfinite(target).all()
        ):
            raise GraspNetUnavailableError(
                "target point cloud must be a finite nonempty (N, 3) array"
            )
        if (
            workspace.ndim != 2
            or workspace.shape[1:] != (3,)
            or len(workspace) == 0
            or not np.isfinite(workspace).all()
        ):
            raise GraspNetUnavailableError(
                "workspace point cloud must be a finite nonempty (N, 3) array"
            )
        pose = calibrated_camera_from_grasp(
            self.world_from_camera,
            np.mean(target, axis=0),
            self.desired_world_ee_rotation,
        ).astype(np.float32)
        span = np.ptp(target[:, :2], axis=0)
        width = max(0.01, min(0.08, float(np.min(span)) + 0.01))
        return (
            GraspCandidate(
                world_from_gripper=pose,
                width_m=width,
                score=1.0,
                collision_free=True,
                provider_name=self.provider_name,
                metadata={"strategy": "camera_calibrated_top_down_centroid"},
            ),
        )
