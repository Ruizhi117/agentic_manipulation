"""Bounded doable/planning/action/checker manipulation runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Protocol

import numpy as np

from agentic_manipulation.config import RuntimeConfig
from agentic_manipulation.control.executor import placement_pose
from agentic_manipulation.errors import AgenticManipulationError, SemanticValidationError
from agentic_manipulation.models.graspnet import GraspProvider, select_grasp
from agentic_manipulation.models.qwen_vl import VisionLanguageModel
from agentic_manipulation.perception.pointcloud import backproject, nearest_instance
from agentic_manipulation.runtime.artifacts import ArtifactWriter
from agentic_manipulation.runtime.checker import CompositeChecker, SceneState
from agentic_manipulation.runtime.semantics import (
    ResolvedAction,
    resolve_grounding,
    validate_compound_demo,
)
from agentic_manipulation.types import (
    CameraFrame,
    ExecutionReport,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeResult,
    TaskResult,
)


class RuntimeScene(SceneState, Protocol):
    def capture(self) -> CameraFrame: ...

    def visible_instances(self) -> Mapping[str, str]: ...

    def segmentation_ids(self) -> Mapping[int, str]: ...

    def centers(self) -> Mapping[str, np.ndarray]: ...

    def workspace_bounds(self) -> tuple[np.ndarray, np.ndarray]: ...

    def bin_inner_aabb(self, bin_id: str) -> tuple[np.ndarray, np.ndarray]: ...

    def object_half_height(self, instance_id: str) -> float: ...


class PickPlaceExecutor(Protocol):
    def can_reach(self, pose: np.ndarray) -> bool: ...

    def execute(
        self, instance_id: str, grasp: object, placement: np.ndarray
    ) -> ExecutionReport: ...


class AgentRuntime:
    def __init__(
        self,
        *,
        config: RuntimeConfig,
        scene: RuntimeScene,
        vlm: VisionLanguageModel,
        grasp_provider: GraspProvider,
        executor: PickPlaceExecutor,
        checker: CompositeChecker,
        artifacts: ArtifactWriter | None = None,
        event_callback: Callable[[RuntimeEvent], None] | None = None,
    ) -> None:
        self.config = config
        self.scene = scene
        self.vlm = vlm
        self.grasp_provider = grasp_provider
        self.executor = executor
        self.checker = checker
        self.artifacts = artifacts or ArtifactWriter(None)
        self._event_callback = event_callback

    def _emit(
        self, event_type: RuntimeEventType, message: str, step: int | None = None
    ) -> None:
        if self._event_callback is not None:
            self._event_callback(RuntimeEvent(event_type, message, step))

    def _failure(
        self, message: str, results: list[TaskResult] | None = None
    ) -> RuntimeResult:
        self._emit(RuntimeEventType.FAILED, message)
        return RuntimeResult(
            success=False,
            task_results=tuple(results or ()),
            message=message,
            is_mock=self.config.mode == "mock",
        )

    @staticmethod
    def _validate_step(
        resolved: ResolvedAction,
        used_targets: set[str],
        visible: Mapping[str, str],
        initial_centers: Mapping[str, np.ndarray],
    ) -> None:
        if resolved.target_instance_id in used_targets:
            raise SemanticValidationError(
                f"duplicate target instance: {resolved.target_instance_id}"
            )
        if resolved.target_label == "tomato_can":
            if resolved.destination_instance_id != "gray_bin":
                raise SemanticValidationError("tomato can must target gray_bin")
            return
        if resolved.target_label == "banana":
            if resolved.destination_instance_id != "purple_bin":
                raise SemanticValidationError("banana must target purple_bin")
            banana_ids = tuple(
                instance_id
                for instance_id, label in visible.items()
                if label == "banana"
            )
            nearest = nearest_instance(
                banana_ids, initial_centers, "purple_bin"
            )
            if resolved.target_instance_id != nearest:
                raise SemanticValidationError(
                    f"selected banana is not nearest: expected {nearest}"
                )
            return
        raise SemanticValidationError(
            f"unsupported target label: {resolved.target_label}"
        )

    def run(self, command: str) -> RuntimeResult:
        visible = dict(self.scene.visible_instances())
        segmentation_ids = dict(self.scene.segmentation_ids())
        initial_centers = {
            key: np.asarray(value).copy() for key, value in self.scene.centers().items()
        }
        used_targets: set[str] = set()
        results: list[TaskResult] = []

        self._emit(RuntimeEventType.OBSERVING, "capturing scene for doable evaluation")
        initial_frame = self.scene.capture()
        self.artifacts.write_frame(initial_frame)
        self.artifacts.write_json("request", {"command": command})
        try:
            doable = self.vlm.evaluate_doable(command, initial_frame)
        except AgenticManipulationError as exc:
            return self._failure(str(exc))
        self._emit(RuntimeEventType.DOABLE, doable.thought)
        if not doable.status:
            return self._failure(f"任务不可执行：{doable.thought}")

        try:
            planning_frame = self.scene.capture()
            plan = self.vlm.plan(command, planning_frame)
            validate_compound_demo(plan, visible)
        except (AgenticManipulationError, ValueError) as exc:
            return self._failure(f"规划失败：{exc}")
        self.artifacts.write_json(
            "plan",
            {
                "type": plan.type,
                "tasks": [
                    {
                        "step": task.step,
                        "action": task.action,
                        "checker": task.checker,
                    }
                    for task in plan.tasks
                ],
            },
        )
        self._emit(RuntimeEventType.PLANNING, f"planned {len(plan.tasks)} tasks")

        for task in plan.tasks:
            last_failure = "unknown failure"
            task_succeeded = False
            for attempt in range(1, self.config.max_retries + 2):
                try:
                    self._emit(
                        RuntimeEventType.ACTION,
                        f"grounding step {task.step}: {task.action}",
                        task.step,
                    )
                    frame = self.scene.capture()
                    grounded = self.vlm.ground(command, task, frame)
                    if grounded.task_step != task.step:
                        raise SemanticValidationError(
                            f"grounding step {grounded.task_step} does not match task {task.step}"
                        )
                    label_candidates = {
                        instance_id
                        for instance_id, label in visible.items()
                        if label == grounded.target_label
                    }
                    if grounded.target_label == "tomato_can":
                        remaining_targets = sorted(
                            label_candidates - used_targets
                        )
                        allowed_targets = (
                            {remaining_targets[0]} if remaining_targets else set()
                        )
                    elif grounded.target_label == "banana":
                        banana_ids = tuple(sorted(label_candidates))
                        allowed_targets = {
                            nearest_instance(
                                banana_ids, initial_centers, "purple_bin"
                            )
                        }
                    else:
                        allowed_targets = label_candidates
                    resolved = resolve_grounding(
                        grounded,
                        frame,
                        segmentation_ids,
                        visible,
                        allowed_target_ids=allowed_targets,
                    )
                    self._validate_step(
                        resolved, used_targets, visible, initial_centers
                    )
                    instance_to_seg = {
                        instance_id: seg_id
                        for seg_id, instance_id in segmentation_ids.items()
                    }
                    target_points = backproject(
                        frame,
                        segmentation_id=instance_to_seg[
                            resolved.target_instance_id
                        ],
                    )
                    workspace_points = backproject(frame)
                    self._emit(
                        RuntimeEventType.GRASPING,
                        f"requesting grasp for {resolved.target_instance_id}",
                        task.step,
                    )
                    candidates = self.grasp_provider.predict(
                        target_points, workspace_points
                    )
                    self.artifacts.write_grasps(candidates)
                    grasp = select_grasp(
                        candidates,
                        self.scene.workspace_bounds(),
                        max_width_m=0.08,
                        reachable=self.executor.can_reach,
                    )
                    placement = placement_pose(
                        self.scene.bin_inner_aabb(
                            resolved.destination_instance_id
                        ),
                        self.scene.object_half_height(
                            resolved.target_instance_id
                        ),
                        grasp.world_from_gripper[:3, :3],
                    )
                    self._emit(
                        RuntimeEventType.EXECUTING,
                        task.action,
                        task.step,
                    )
                    execution = self.executor.execute(
                        resolved.target_instance_id, grasp, placement
                    )
                    if not execution.success:
                        raise SemanticValidationError(
                            execution.failure_reason or "motion execution failed"
                        )
                    check_frame = self.scene.capture()
                    visual = self.vlm.check(
                        command, task, grounded, check_frame
                    )
                    self._emit(
                        RuntimeEventType.CHECKING,
                        task.checker,
                        task.step,
                    )
                    checked = replace(
                        self.checker.check(visual, self.scene, resolved),
                        attempts=attempt,
                    )
                    if not checked.success:
                        raise SemanticValidationError("; ".join(checked.failures))
                    results.append(checked)
                    used_targets.add(resolved.target_instance_id)
                    task_succeeded = True
                    break
                except (AgenticManipulationError, KeyError, ValueError) as exc:
                    last_failure = str(exc)
                    if attempt <= self.config.max_retries:
                        self._emit(
                            RuntimeEventType.RETRYING,
                            f"step {task.step} attempt {attempt} failed: {last_failure}",
                            task.step,
                        )
                        continue
                    results.append(
                        TaskResult(
                            step=task.step,
                            success=False,
                            failures=(last_failure,),
                            attempts=attempt,
                        )
                    )
            if not task_succeeded:
                return self._failure(
                    f"步骤 {task.step} 失败：{last_failure}", results
                )

        message = "任务完成，还需要什么？"
        self._emit(RuntimeEventType.SUCCEEDED, message)
        return RuntimeResult(
            success=True,
            task_results=tuple(results),
            message=message,
            is_mock=self.config.mode == "mock",
        )
