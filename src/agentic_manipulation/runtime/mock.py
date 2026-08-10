"""Deterministic in-memory adapters for an explicitly labelled mock demo."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from agentic_manipulation.config import RuntimeConfig
from agentic_manipulation.models.graspnet import DeterministicTopDownGraspProvider
from agentic_manipulation.models.qwen_vl import DeterministicVisionLanguageModel
from agentic_manipulation.runtime.agent import AgentRuntime
from agentic_manipulation.runtime.checker import CompositeChecker
from agentic_manipulation.types import (
    BBox,
    CameraFrame,
    ExecutionReport,
    GroundedAction,
    PlannedTask,
    PlanningEval,
    RuntimeEvent,
)


COMPOUND_COMMAND = (
    "请将所有的西红柿罐头放到灰色箱子中，然后选择离紫色箱子最近的香蕉放到紫色箱子中"
)

INSTANCE_IDS = {
    1: "tomato_can_1",
    2: "tomato_can_2",
    3: "banana_1",
    4: "banana_2",
    5: "gray_bin",
    6: "purple_bin",
}
LABELS = {
    "tomato_can_1": "tomato_can",
    "tomato_can_2": "tomato_can",
    "banana_1": "banana",
    "banana_2": "banana",
    "gray_bin": "gray_bin",
    "purple_bin": "purple_bin",
}


def _column_bbox(column: int) -> BBox:
    return BBox(column / 6, 0.0, (column + 1) / 6, 1.0)


def canonical_plan() -> PlanningEval:
    return PlanningEval(
        "planning",
        (
            PlannedTask(
                1,
                "把西红柿罐头1移动到灰色箱子中",
                "检查西红柿罐头1是否在灰色箱子中",
            ),
            PlannedTask(
                2,
                "把西红柿罐头2移动到灰色箱子中",
                "检查西红柿罐头2是否在灰色箱子中",
            ),
            PlannedTask(
                3,
                "把离紫色箱子最近的香蕉移动到紫色箱子中",
                "检查香蕉是否在紫色箱子中",
            ),
        ),
    )


def canonical_groundings() -> dict[int, GroundedAction]:
    return {
        1: GroundedAction(
            "action", 1, "tomato_can", _column_bbox(0), "gray_bin", _column_bbox(4)
        ),
        2: GroundedAction(
            "action", 2, "tomato_can", _column_bbox(1), "gray_bin", _column_bbox(4)
        ),
        3: GroundedAction(
            "action", 3, "banana", _column_bbox(2), "purple_bin", _column_bbox(5)
        ),
    }


class MockScene:
    """Small RGB-D/segmentation scene whose state changes after mock execution."""

    def __init__(self) -> None:
        self.locations: dict[str, str | None] = {
            instance_id: None for instance_id in LABELS
        }
        self._centers = {
            "tomato_can_1": np.array([0.0, -0.2, 0.05]),
            "tomato_can_2": np.array([0.0, 0.0, 0.05]),
            "banana_1": np.array([0.1, 0.25, 0.03]),
            "banana_2": np.array([0.0, -0.2, 0.03]),
            "gray_bin": np.array([0.3, -0.3, 0.05]),
            "purple_bin": np.array([0.3, 0.3, 0.05]),
        }
        self.capture_count = 0

    def capture(self) -> CameraFrame:
        self.capture_count += 1
        segmentation = np.tile(np.arange(1, 7, dtype=np.int32), (2, 1))
        return CameraFrame(
            rgb=np.zeros((2, 6, 3), dtype=np.uint8),
            depth_m=np.ones((2, 6), dtype=np.float32),
            intrinsic=np.eye(3, dtype=np.float32),
            world_from_camera=np.eye(4, dtype=np.float32),
            segmentation=segmentation,
            timestamp=float(self.capture_count),
        )

    def visible_instances(self) -> Mapping[str, str]:
        return LABELS

    def segmentation_ids(self) -> Mapping[int, str]:
        return INSTANCE_IDS

    def centers(self) -> Mapping[str, np.ndarray]:
        return self._centers

    def workspace_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-1.0, -1.0, 0.0]), np.array([10.0, 10.0, 2.0])

    def bin_inner_aabb(self, bin_id: str) -> tuple[np.ndarray, np.ndarray]:
        center = self._centers[bin_id]
        return center - [0.1, 0.1, 0.04], center + [0.1, 0.1, 0.16]

    def object_half_height(self, instance_id: str) -> float:
        return 0.025 if "banana" in instance_id else 0.045

    def is_in_bin(self, instance_id: str, bin_id: str) -> bool:
        return self.locations[instance_id] == bin_id

    def is_released(self, instance_id: str) -> bool:
        return self.locations[instance_id] is not None

    def is_stable(self, instance_id: str) -> bool:
        return self.locations[instance_id] is not None

    def close(self) -> None:
        return None


@dataclass
class MockPickPlaceExecutor:
    scene: MockScene

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    def can_reach(self, pose: np.ndarray) -> bool:
        return np.asarray(pose).shape == (4, 4)

    def execute(self, instance_id, grasp, placement) -> ExecutionReport:
        self.calls.append(instance_id)
        destination = min(
            ("gray_bin", "purple_bin"),
            key=lambda bin_id: np.linalg.norm(
                placement[:2, 3] - self.scene.centers()[bin_id][:2]
            ),
        )
        self.scene.locations[instance_id] = destination
        return ExecutionReport(True, ("mock-move",))


def build_mock_runtime(
    config: RuntimeConfig,
    *,
    event_callback: Callable[[RuntimeEvent], None] | None = None,
) -> AgentRuntime:
    if config.mode != "mock":
        raise ValueError("build_mock_runtime requires mode='mock'")
    scene = MockScene()
    return AgentRuntime(
        config=config,
        scene=scene,
        vlm=DeterministicVisionLanguageModel(
            groundings=canonical_groundings(), planning=canonical_plan()
        ),
        grasp_provider=DeterministicTopDownGraspProvider(),
        executor=MockPickPlaceExecutor(scene),
        checker=CompositeChecker(),
        event_callback=event_callback,
    )

