"""Validate VLM plans and bind visual boxes to stable simulator instances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from agentic_manipulation.errors import PerceptionError, SemanticValidationError
from agentic_manipulation.perception.pointcloud import match_instance, nearest_instance
from agentic_manipulation.types import BBox, CameraFrame, GroundedAction, PlanningEval


@dataclass(frozen=True)
class ResolvedAction:
    task_step: int
    target_instance_id: str
    target_label: str
    destination_instance_id: str
    destination_label: str
    target_bbox: BBox
    destination_bbox: BBox | None

    def __post_init__(self) -> None:
        if self.task_step <= 0:
            raise ValueError("task_step must be positive")


def validate_plan(
    plan: PlanningEval, visible_instances: Mapping[str, str]
) -> None:
    if not plan.tasks:
        raise SemanticValidationError("planning returned no tasks")
    if not visible_instances:
        raise SemanticValidationError("no visible instances are available")
    steps = tuple(task.step for task in plan.tasks)
    if steps != tuple(range(1, len(steps) + 1)):
        raise SemanticValidationError("plan steps must be contiguous")


def validate_compound_demo(
    plan: PlanningEval, visible_instances: Mapping[str, str]
) -> None:
    """Validate the accepted tomato-can/nearest-banana demonstration plan."""

    validate_plan(plan, visible_instances)
    tomato_ids = {
        instance_id
        for instance_id, label in visible_instances.items()
        if label == "tomato_can"
    }
    expected_tasks = len(tomato_ids) + 1
    if len(plan.tasks) != expected_tasks:
        raise SemanticValidationError(
            f"compound demo requires {expected_tasks} tasks for visible objects"
        )
    tomato_tasks = [task for task in plan.tasks if "西红柿" in task.action]
    if len(tomato_tasks) != len(tomato_ids):
        raise SemanticValidationError("plan must expand every visible tomato can")
    if any("灰色" not in task.action for task in tomato_tasks):
        raise SemanticValidationError("every tomato can task must target the gray bin")
    banana_tasks = [task for task in plan.tasks if "香蕉" in task.action]
    if len(banana_tasks) != 1:
        raise SemanticValidationError("plan must contain one banana task")
    banana_action = banana_tasks[0].action
    if "紫色" not in banana_action or "最近" not in banana_action:
        raise SemanticValidationError(
            "banana task must select the nearest banana for the purple bin"
        )


def resolve_grounding(
    grounded: GroundedAction,
    frame: CameraFrame,
    seg_id_to_instance: Mapping[int, str],
    instance_labels: Mapping[str, str],
    allowed_target_ids: set[str] | None = None,
) -> ResolvedAction:
    if frame.segmentation is None:
        raise PerceptionError("segmentation is required to resolve grounding")
    try:
        target_id = match_instance(
            grounded.target_bbox, frame.segmentation, seg_id_to_instance
        )
    except SemanticValidationError:
        target_id = ""
    if allowed_target_ids is not None:
        allowed_matching = {
            instance_id
            for instance_id in allowed_target_ids
            if instance_labels.get(instance_id) == grounded.target_label
        }
        if target_id not in allowed_matching:
            if len(allowed_matching) == 1:
                target_id = next(iter(allowed_matching))
            else:
                raise SemanticValidationError(
                    "target bbox does not identify an allowed semantic instance"
                )
    elif not target_id:
        raise SemanticValidationError("bbox does not overlap a known instance")
    destination_candidates = [
        instance_id
        for instance_id, label in instance_labels.items()
        if label == grounded.destination_label
    ]
    if len(destination_candidates) == 1:
        destination_id = destination_candidates[0]
    elif grounded.destination_bbox is not None:
        destination_id = match_instance(
            grounded.destination_bbox, frame.segmentation, seg_id_to_instance
        )
    else:
        raise SemanticValidationError(
            f"destination label {grounded.destination_label!r} is not unique"
        )
    if instance_labels.get(target_id) != grounded.target_label:
        raise SemanticValidationError(
            f"target label {grounded.target_label!r} does not match instance {target_id!r}"
        )
    if instance_labels.get(destination_id) != grounded.destination_label:
        raise SemanticValidationError(
            "destination label "
            f"{grounded.destination_label!r} does not match instance {destination_id!r}"
        )
    return ResolvedAction(
        task_step=grounded.task_step,
        target_instance_id=target_id,
        target_label=grounded.target_label,
        destination_instance_id=destination_id,
        destination_label=grounded.destination_label,
        target_bbox=grounded.target_bbox,
        destination_bbox=grounded.destination_bbox,
    )


def validate_resolved_actions(
    actions: Sequence[ResolvedAction],
    visible_instances: Mapping[str, str],
    centers: Mapping[str, np.ndarray],
) -> None:
    if tuple(action.task_step for action in actions) != tuple(
        range(1, len(actions) + 1)
    ):
        raise SemanticValidationError("resolved action steps must be contiguous")
    targets = [action.target_instance_id for action in actions]
    if len(targets) != len(set(targets)):
        raise SemanticValidationError("duplicate target instance in resolved actions")

    expected_tomatoes = {
        instance_id
        for instance_id, label in visible_instances.items()
        if label == "tomato_can"
    }
    tomato_actions = [action for action in actions if action.target_label == "tomato_can"]
    if {action.target_instance_id for action in tomato_actions} != expected_tomatoes:
        raise SemanticValidationError("resolved actions must cover every tomato can once")
    if any(action.destination_instance_id != "gray_bin" for action in tomato_actions):
        raise SemanticValidationError("every resolved tomato can must target gray_bin")

    banana_actions = [action for action in actions if action.target_label == "banana"]
    if len(banana_actions) != 1:
        raise SemanticValidationError("resolved actions must contain one banana")
    banana_action = banana_actions[0]
    if banana_action.destination_instance_id != "purple_bin":
        raise SemanticValidationError("resolved banana must target purple_bin")
    banana_ids = tuple(
        instance_id
        for instance_id, label in visible_instances.items()
        if label == "banana"
    )
    nearest = nearest_instance(banana_ids, centers, "purple_bin")
    if banana_action.target_instance_id != nearest:
        raise SemanticValidationError(
            f"selected banana is not the nearest banana: expected {nearest}"
        )
