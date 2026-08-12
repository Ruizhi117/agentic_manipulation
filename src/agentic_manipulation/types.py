"""Validated data contracts for the closed-loop manipulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

import numpy as np


def _require_finite_matrix(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class BBox:
    """Normalized inclusive-exclusive image bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
            raise ValueError("bbox coordinates must be finite values in [0, 1]")
        if not (self.x1 < self.x2 and self.y1 < self.y2):
            raise ValueError("bbox coordinates must be ordered with positive area")

    def as_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        if width <= 0 or height <= 0:
            raise ValueError("image width and height must be positive")
        x1 = max(0, min(width, math.floor(self.x1 * width)))
        y1 = max(0, min(height, math.floor(self.y1 * height)))
        x2 = max(0, min(width, math.ceil(self.x2 * width)))
        y2 = max(0, min(height, math.ceil(self.y2 * height)))
        return x1, y1, x2, y2


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsic: np.ndarray
    world_from_camera: np.ndarray
    segmentation: np.ndarray | None
    timestamp: float

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"rgb shape must be (H, W, 3), got {self.rgb.shape}")
        if self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb dtype must be uint8, got {self.rgb.dtype}")
        expected_hw = self.rgb.shape[:2]
        if self.depth_m.shape != expected_hw:
            raise ValueError(
                f"depth_m shape must match rgb height/width {expected_hw}, got {self.depth_m.shape}"
            )
        if not np.issubdtype(self.depth_m.dtype, np.floating):
            raise ValueError("depth_m must use a floating dtype")
        if self.segmentation is not None and self.segmentation.shape != expected_hw:
            raise ValueError(
                "segmentation shape must match rgb height/width "
                f"{expected_hw}, got {self.segmentation.shape}"
            )
        _require_finite_matrix(self.intrinsic, (3, 3), "intrinsic")
        _require_finite_matrix(
            self.world_from_camera, (4, 4), "world_from_camera"
        )
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")


@dataclass(frozen=True)
class DoableEval:
    type: str
    thought: str
    status: bool

    def __post_init__(self) -> None:
        if self.type != "doable":
            raise ValueError("type must be 'doable'")
        if not isinstance(self.status, bool):
            raise ValueError("status must be boolean")


@dataclass(frozen=True)
class PlannedTask:
    step: int
    action: str
    checker: str

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be a positive integer")
        if not self.action.strip():
            raise ValueError("action must not be empty")
        if not self.checker.strip():
            raise ValueError("checker must not be empty")


@dataclass(frozen=True)
class PlanningEval:
    type: str
    tasks: tuple[PlannedTask, ...]

    def __post_init__(self) -> None:
        if self.type != "planning":
            raise ValueError("type must be 'planning'")
        expected = tuple(range(1, len(self.tasks) + 1))
        actual = tuple(task.step for task in self.tasks)
        if actual != expected:
            raise ValueError("planning steps must be contiguous and start at 1")


@dataclass(frozen=True)
class GroundedAction:
    type: str
    task_step: int
    target_label: str
    target_bbox: BBox
    destination_label: str
    destination_bbox: BBox | None

    def __post_init__(self) -> None:
        if self.type != "action":
            raise ValueError("type must be 'action'")
        if self.task_step <= 0:
            raise ValueError("task_step must be a positive integer")
        if not self.target_label.strip():
            raise ValueError("target_label must not be empty")
        if not self.destination_label.strip():
            raise ValueError("destination_label must not be empty")


@dataclass(frozen=True)
class RegionClassification:
    """VLM color/category judgment for one depth-segmented image region."""

    region_id: int
    color: str
    kind: str
    image_location: str
    label: str
    vlm_color: str | None = None
    vlm_kind: str | None = None
    color_source: str = "vlm"
    rgb_median: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.region_id, bool)
            or not isinstance(self.region_id, int)
            or self.region_id < 0
        ):
            raise ValueError("region_id must be a nonnegative integer")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.color, self.kind, self.image_location, self.label)
        ):
            raise ValueError("region classification strings must not be blank")
        if self.vlm_color is not None and (
            not isinstance(self.vlm_color, str) or not self.vlm_color.strip()
        ):
            raise ValueError("vlm_color must be nonblank when provided")
        if self.vlm_kind is not None and (
            not isinstance(self.vlm_kind, str) or not self.vlm_kind.strip()
        ):
            raise ValueError("vlm_kind must be nonblank when provided")
        if not isinstance(self.color_source, str) or not self.color_source.strip():
            raise ValueError("color_source must not be blank")
        if self.rgb_median is not None and (
            len(self.rgb_median) != 3
            or not all(math.isfinite(value) for value in self.rgb_median)
        ):
            raise ValueError("rgb_median must contain three finite values")


@dataclass(frozen=True)
class CheckerEval:
    type: str
    thought: str
    status: bool

    def __post_init__(self) -> None:
        if self.type != "checker":
            raise ValueError("type must be 'checker'")
        if not isinstance(self.status, bool):
            raise ValueError("status must be boolean")


@dataclass(frozen=True)
class GraspCandidate:
    world_from_gripper: np.ndarray
    width_m: float
    score: float
    collision_free: bool
    provider_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_finite_matrix(
            self.world_from_gripper, (4, 4), "world_from_gripper"
        )
        if not math.isfinite(self.width_m) or self.width_m <= 0:
            raise ValueError("width_m must be finite and positive")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if not self.provider_name.strip():
            raise ValueError("provider_name must not be empty")


@dataclass(frozen=True)
class ExecutionReport:
    success: bool
    completed_states: tuple[str, ...]
    failure_reason: str | None = None


@dataclass(frozen=True)
class TaskResult:
    step: int
    success: bool
    failures: tuple[str, ...] = ()
    attempts: int = 1


class RuntimeEventType(str, Enum):
    OBSERVING = "observing"
    DOABLE = "doable"
    PLANNING = "planning"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    ACTION = "action"
    GRASPING = "grasping"
    EXECUTING = "executing"
    CHECKING = "checking"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class RuntimeEvent:
    type: RuntimeEventType
    message: str
    task_step: int | None = None


@dataclass(frozen=True)
class RuntimeResult:
    success: bool
    task_results: tuple[TaskResult, ...]
    message: str
    is_mock: bool
