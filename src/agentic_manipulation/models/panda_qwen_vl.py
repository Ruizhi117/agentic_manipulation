"""General Qwen3-VL prompts for the Panda six-object/two-bin scene."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Protocol

from agentic_manipulation.errors import ModelResponseError
from agentic_manipulation.models.qwen_vl import PHASE_SCHEMAS, object_bbox
from agentic_manipulation.types import (
    CameraFrame,
    CheckerEval,
    DoableEval,
    GroundedAction,
    PlannedTask,
    PlanningEval,
)


PANDA_OBJECT_LABELS = (
    "red_cube",
    "blue_cube",
    "yellow_block",
    "purple_block",
    "green_cylinder",
    "orange_cylinder",
)
PANDA_DESTINATION_LABELS = ("white_bin", "pink_bin")


class StructuredPhaseClient(Protocol):
    def run_structured_phase(
        self,
        phase: str,
        prompt: str,
        frame: CameraFrame,
        *,
        schema: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


TraceCallback = Callable[[str, str, Mapping[str, object]], None]


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelResponseError(f"{field} must be an object")
    return value


def _result_type(data: Mapping[str, Any], expected: str) -> None:
    if data.get("type") != expected:
        raise ModelResponseError(
            f"expected {expected} response, got {data.get('type')!r}"
        )


class PandaVisionLanguageModel:
    """Typed Panda VLM adapter; geometry and robot control remain outside Qwen."""

    provider_name = "ollama-qwen-vl-panda"

    def __init__(
        self,
        client: StructuredPhaseClient,
        *,
        trace: TraceCallback | None = None,
    ) -> None:
        self.client = client
        self.trace = trace

    @staticmethod
    def _scene_vocabulary() -> str:
        objects = ", ".join(PANDA_OBJECT_LABELS)
        bins = ", ".join(PANDA_DESTINATION_LABELS)
        return (
            f"Canonical graspable object labels: {objects}. "
            f"Canonical destination bin labels: {bins}. "
            "Aliases: 红色方块=red_cube, 蓝色方块=blue_cube, "
            "黄色方块=yellow_block, 紫色方块=purple_block, "
            "绿色圆柱=green_cylinder, 橙色圆柱=orange_cylinder, "
            "白色盒子=white_bin, 粉色或紫色盒子=pink_bin. "
            "Words 方块/cube/block include all four cube/block objects; "
            "圆柱/cylinder includes both cylinders."
        )

    def _emit(self, phase: str, prompt: str, result: object) -> None:
        if self.trace is not None:
            self.trace(phase, prompt, asdict(result))

    def _run(
        self,
        phase: str,
        prompt: str,
        frame: CameraFrame,
        *,
        schema: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        result = self.client.run_structured_phase(
            phase, prompt, frame, schema=schema
        )
        if not isinstance(result, Mapping):
            raise ModelResponseError(f"{phase} result must be an object")
        _result_type(result, phase)
        return result

    def evaluate_doable(self, command: str, frame: CameraFrame) -> DoableEval:
        prompt = (
            f"Human command: {command}\n{self._scene_vocabulary()}\n"
            "Judge whether every requested target and destination is visibly present "
            "and the complete sorting command is feasible. Use visible evidence only. "
            "Before execution, write thought in the same language as the human command "
            "using at most three short sentences: report detected object names and counts, "
            "the current scene state, and the requested action sequence. Keep it concise."
        )
        data = self._run("doable", prompt, frame)
        try:
            result = DoableEval(
                type=str(data["type"]),
                thought=str(data["thought"]),
                status=data["status"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelResponseError(f"invalid Panda doable response: {exc}") from exc
        self._emit("doable", prompt, result)
        return result

    def plan(self, command: str, frame: CameraFrame) -> PlanningEval:
        prompt = (
            f"Human command: {command}\n{self._scene_vocabulary()}\n"
            "Create the complete ordered execution queue now: one atomic task per "
            "physical object that must move. Each action "
            "must name exactly one canonical object instance and exactly one of the "
            "two destination bins. Expand words such as all into separate tasks. "
            "Do not refer to earlier tasks, later tasks, or grouped objects inside any "
            "single task. Order steps from 1 without gaps. Never produce grasp poses "
            "or robot commands."
        )
        planning_schema = deepcopy(PHASE_SCHEMAS["planning"])
        planning_schema["properties"]["tasks"]["maxItems"] = len(
            PANDA_OBJECT_LABELS
        )
        data = self._run("planning", prompt, frame, schema=planning_schema)
        try:
            raw_tasks = data["tasks"]
            if not isinstance(raw_tasks, list):
                raise TypeError("tasks must be a list")
            tasks = tuple(
                PlannedTask(
                    step=int(_mapping(item, "task")["step"]),
                    action=str(_mapping(item, "task")["action"]),
                    checker=str(_mapping(item, "task")["checker"]),
                )
                for item in raw_tasks
            )
            result = PlanningEval(type=str(data["type"]), tasks=tasks)
        except ModelResponseError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelResponseError(f"invalid Panda planning response: {exc}") from exc
        self._emit("planning", prompt, result)
        return result

    def ground(
        self, command: str, task: PlannedTask, frame: CameraFrame
    ) -> GroundedAction:
        prompt = (
            f"Human command: {command}\nCurrent task {task.step}: {task.action}\n"
            f"Checker intent: {task.checker}\n{self._scene_vocabulary()}\n"
            "Ground exactly one visible target object and the specified destination "
            "bin. Return canonical labels and normalized bbox_xyxy_norm bounding "
            "boxes [x1, y1, x2, y2] for both target and destination. A destination bounding box is "
            "mandatory even though the simulator later supplies its metric geometry. "
            "The image may contain authoritative simulator rectangles and canonical "
            "text labels. The current task may also contain an authoritative "
            "bbox_xyxy_norm catalog. Select the commanded labels and copy their "
            "catalog coordinates exactly; do not estimate or convert new boxes."
        )
        action_schema = deepcopy(PHASE_SCHEMAS["action"])
        action_schema["properties"]["task_step"] = {
            "type": "integer",
            "const": task.step,
        }
        data = self._run("action", prompt, frame, schema=action_schema)
        try:
            target = _mapping(data["target"], "target")
            destination = _mapping(data["destination"], "destination")
            target_label = str(target["label"])
            destination_label = str(destination["label"])
            if target_label not in PANDA_OBJECT_LABELS:
                raise ModelResponseError(f"unknown Panda target label: {target_label}")
            if destination_label not in PANDA_DESTINATION_LABELS:
                raise ModelResponseError(
                    f"unknown Panda destination label: {destination_label}"
                )
            destination_bbox = object_bbox(destination, "destination bounding box")
            result = GroundedAction(
                type=str(data["type"]),
                task_step=int(data["task_step"]),
                target_label=target_label,
                target_bbox=object_bbox(target, "target bounding box"),
                destination_label=destination_label,
                destination_bbox=destination_bbox,
            )
            if result.task_step != task.step:
                raise ModelResponseError(
                    f"action task_step {result.task_step} does not match {task.step}"
                )
        except ModelResponseError as exc:
            if "destination bounding box" in str(exc) and "missing" in str(exc):
                raise ModelResponseError(
                    "destination bounding box is required for Panda action"
                ) from exc
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelResponseError(f"invalid Panda action response: {exc}") from exc
        self._emit("action", prompt, result)
        return result

    def _check(
        self,
        trace_phase: str,
        evidence_rule: str,
        command: str,
        task: PlannedTask,
        grounded: GroundedAction,
        frame: CameraFrame,
    ) -> CheckerEval:
        prompt = (
            f"Human command: {command}\nTask {task.step}: {task.action}\n"
            f"Target: {grounded.target_label}; destination: "
            f"{grounded.destination_label}.\n{evidence_rule}\n"
            "Return status=true only when the requested visual condition is clearly met."
        )
        data = self._run("checker", prompt, frame)
        try:
            result = CheckerEval(
                type=str(data["type"]),
                thought=str(data["thought"]),
                status=data["status"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelResponseError(f"invalid Panda checker response: {exc}") from exc
        self._emit(trace_phase, prompt, result)
        return result

    def check_grasp(
        self,
        command: str,
        task: PlannedTask,
        grounded: GroundedAction,
        frame: CameraFrame,
    ) -> CheckerEval:
        return self._check(
            "grasp_checker",
            "Verify the target is held between the Panda fingers and visibly lifted from the table.",
            command,
            task,
            grounded,
            frame,
        )

    def check_place(
        self,
        command: str,
        task: PlannedTask,
        grounded: GroundedAction,
        frame: CameraFrame,
    ) -> CheckerEval:
        return self._check(
            "place_checker",
            "Verify the target is inside the specified open-top bin and no longer "
            "held by the gripper. other objects already inside are allowed; overlap "
            "or partial occlusion does not invalidate placement.",
            command,
            task,
            grounded,
            frame,
        )

    def check(
        self,
        command: str,
        task: PlannedTask,
        grounded: GroundedAction,
        frame: CameraFrame,
    ) -> CheckerEval:
        """Compatibility boundary: the generic final checker is the place check."""

        return self.check_place(command, task, grounded, frame)
