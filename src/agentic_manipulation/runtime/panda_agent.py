"""Closed-loop Panda runtime with per-attempt perception and dual checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
import re
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from agentic_manipulation.config import RuntimeConfig
from agentic_manipulation.control.panda_atomic import (
    AtomicPickReport,
    AtomicPlaceReport,
)
from agentic_manipulation.demo.panda_calibration import compose_panda_world_ee
from agentic_manipulation.envs.ee_camera_scene import (
    DESTINATION_INSTANCE_IDS,
    GRASPABLE_INSTANCE_IDS,
)
from agentic_manipulation.errors import (
    AgenticManipulationError,
    GraspNetUnavailableError,
    SemanticValidationError,
)
from agentic_manipulation.models.graspnet import GraspProvider
from agentic_manipulation.perception.pointcloud import (
    backproject_camera,
    crop_local_workspace_camera,
    match_instance,
)
from agentic_manipulation.perception.depth import depth_grayscale_rgb
from agentic_manipulation.runtime.artifacts import ArtifactWriter
from agentic_manipulation.runtime.semantics import ResolvedAction
from agentic_manipulation.types import (
    CameraFrame,
    GroundedAction,
    PlannedTask,
    PlanningEval,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeResult,
    TaskResult,
)


_OBJECT_ALIASES = {
    "red_cube": (
        "red cube",
        "红色方块",
        "红方块",
        "红色立方体",
        "红色cube",
    ),
    "blue_cube": (
        "blue cube",
        "蓝色方块",
        "蓝方块",
        "蓝色立方体",
        "蓝色cube",
    ),
    "yellow_block": (
        "yellow block",
        "yellow cube",
        "黄色方块",
        "黄方块",
        "黄色长方体",
    ),
    "purple_block": (
        "purple block",
        "purple cube",
        "紫色方块",
        "紫方块",
        "紫色长方体",
    ),
    "green_cylinder": (
        "green cylinder",
        "绿色圆柱",
        "绿圆柱",
        "绿色圆柱体",
    ),
    "orange_cylinder": (
        "orange cylinder",
        "橙色圆柱",
        "橙圆柱",
        "橙色圆柱体",
    ),
}
_DESTINATION_ALIASES = {
    "white_bin": (
        "white bin",
        "white box",
        "白色盒子",
        "白盒子",
        "白色箱子",
        "白色框子",
        "灰色盒子",
        "灰色箱子",
        "gray bin",
        "grey bin",
    ),
    "pink_bin": (
        "pink bin",
        "pink box",
        "粉色盒子",
        "粉红色盒子",
        "粉色箱子",
        "紫色盒子",
        "紫色箱子",
        "purple bin",
    ),
}
_ALL_QUANTIFIERS = ("all", "every", "所有", "全部", "每个", "每一个")
_ALL_OBJECT_GROUPS = ("all objects", "every object", "所有物体", "全部物体", "所有对象")
_BLOCK_GROUPS = ("cubes", "cube", "blocks", "block", "方块", "立方体", "长方体")
_CYLINDER_GROUPS = ("cylinders", "cylinder", "圆柱", "圆柱体")


def _semantic_text(value: object) -> str:
    return re.sub(r"[\s_-]+", " ", str(value).casefold()).strip()


def _has_any(text: str, aliases: Sequence[str]) -> bool:
    return any(_semantic_text(alias) in text for alias in aliases)


def _labels_in_text(
    value: object,
    aliases_by_label: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    text = _semantic_text(value)
    labels = []
    for label, aliases in aliases_by_label.items():
        if _semantic_text(label) in text or _has_any(text, aliases):
            labels.append(label)
    return tuple(labels)


def _first_alias_position(text: str, aliases: Sequence[str]) -> int | None:
    positions = [
        position
        for alias in aliases
        if (position := text.find(_semantic_text(alias))) >= 0
    ]
    return min(positions) if positions else None


def _destination_mentions(command: str) -> tuple[tuple[int, str], ...]:
    text = _semantic_text(command)
    mentions = set()
    for label, aliases in _DESTINATION_ALIASES.items():
        for alias in (label, *aliases):
            normalized = _semantic_text(alias)
            start = 0
            while (position := text.find(normalized, start)) >= 0:
                mentions.add((position, label))
                start = position + len(normalized)
    return tuple(sorted(mentions))


def _target_mention_position(command: str, target: str) -> int | None:
    text = _semantic_text(command)
    specific = _first_alias_position(
        text, (target, *_OBJECT_ALIASES[target])
    )
    if specific is not None:
        return specific
    if target in GRASPABLE_INSTANCE_IDS[:4]:
        grouped = _first_alias_position(text, _BLOCK_GROUPS)
        if grouped is not None:
            return grouped
    if target in GRASPABLE_INSTANCE_IDS[4:]:
        grouped = _first_alias_position(text, _CYLINDER_GROUPS)
        if grouped is not None:
            return grouped
    return _first_alias_position(text, _ALL_OBJECT_GROUPS)


def _requested_targets(
    command: str,
    visible_graspables: set[str],
) -> tuple[str, ...]:
    text = _semantic_text(command)
    requested = set(_labels_in_text(command, _OBJECT_ALIASES))
    if _has_any(text, _ALL_QUANTIFIERS):
        if _has_any(text, _ALL_OBJECT_GROUPS):
            requested.update(GRASPABLE_INSTANCE_IDS)
        else:
            if _has_any(text, _BLOCK_GROUPS):
                requested.update(GRASPABLE_INSTANCE_IDS[:4])
            if _has_any(text, _CYLINDER_GROUPS):
                requested.update(GRASPABLE_INSTANCE_IDS[4:])
    return tuple(
        label
        for label in GRASPABLE_INSTANCE_IDS
        if label in requested and label in visible_graspables
    )


def _canonical_task(step: int, target: str, destination: str) -> PlannedTask:
    return PlannedTask(
        step,
        f"move {target} to {destination}",
        f"check {target} is in {destination}",
    )


def canonicalize_panda_plan(
    command: str,
    plan: PlanningEval,
    visible_instances: Mapping[str, str],
) -> PlanningEval:
    """Normalize common Chinese/English aliases and expand quantified groups."""

    visible_graspables = {
        label for label in GRASPABLE_INSTANCE_IDS if label in visible_instances
    }
    requested = _requested_targets(command, visible_graspables)
    destination_mentions = _destination_mentions(command)
    command_destinations = tuple(dict.fromkeys(label for _, label in destination_mentions))
    if requested and destination_mentions:
        assignments = []
        for target in requested:
            target_position = _target_mention_position(command, target)
            following = [
                (position, label)
                for position, label in destination_mentions
                if target_position is None or position >= target_position
            ]
            if following:
                assignments.append((target, min(following)[1]))
            elif len(command_destinations) == 1:
                assignments.append((target, command_destinations[0]))
            else:
                assignments = []
                break
    else:
        assignments = []
    if assignments:
        return PlanningEval(
            "planning",
            tuple(
                _canonical_task(step, target, destination)
                for step, (target, destination) in enumerate(
                    assignments, start=1
                )
            ),
        )

    normalized = []
    for task in plan.tasks:
        text = f"{task.action}\n{task.checker}"
        targets = tuple(
            label
            for label in _labels_in_text(text, _OBJECT_ALIASES)
            if label in visible_graspables
        )
        destinations = _labels_in_text(text, _DESTINATION_ALIASES)
        normalized.append(
            _canonical_task(task.step, targets[0], destinations[0])
            if len(targets) == 1 and len(destinations) == 1
            else task
        )
    return PlanningEval("planning", tuple(normalized))


class PandaScene(Protocol):
    def capture(self) -> CameraFrame: ...

    def visible_instances(self) -> Mapping[str, str]: ...

    def segmentation_ids(self) -> Mapping[int, str]: ...

    def is_grasping(self, instance_id: str) -> bool: ...

    def is_in_bin(self, instance_id: str, bin_id: str) -> bool: ...

    def is_released(self, instance_id: str) -> bool: ...

    def is_stable(self, instance_id: str) -> bool: ...


class PandaVLM(Protocol):
    def evaluate_doable(self, command: str, frame: CameraFrame): ...

    def plan(self, command: str, frame: CameraFrame): ...

    def ground(self, command: str, task: object, frame: CameraFrame): ...

    def check_grasp(
        self, command: str, task: object, grounded: GroundedAction, frame: CameraFrame
    ): ...

    def check_place(
        self, command: str, task: object, grounded: GroundedAction, frame: CameraFrame
    ): ...


class PandaAtomicSkill(Protocol):
    def pick(self, instance_id: str, world_from_ee: np.ndarray) -> AtomicPickReport: ...

    def recover_after_failed_pick(self) -> None: ...

    def place(self, instance_id: str, bin_id: str) -> AtomicPlaceReport: ...

    def return_home(self) -> None: ...


def annotated_grounding_frame(
    frame: CameraFrame, segmentation_ids: Mapping[int, str]
) -> CameraFrame:
    """Overlay authoritative simulator instance labels for VLM grounding."""

    if frame.segmentation is None:
        raise SemanticValidationError("grounding overlay requires segmentation")
    image = Image.fromarray(frame.rgb.copy(), mode="RGB")
    draw = ImageDraw.Draw(image)
    colors = (
        (255, 230, 40),
        (40, 255, 100),
        (255, 90, 220),
        (80, 220, 255),
        (255, 140, 40),
        (180, 100, 255),
        (255, 255, 255),
        (255, 80, 120),
    )
    for index, (segmentation_id, instance_id) in enumerate(
        sorted(segmentation_ids.items())
    ):
        rows, columns = np.nonzero(frame.segmentation == segmentation_id)
        if len(rows) == 0:
            continue
        box = (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
        )
        color = colors[index % len(colors)]
        draw.rectangle(box, outline=color, width=2)
        text_box = draw.textbbox((box[0], box[1]), instance_id)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((box[0], box[1]), instance_id, fill=color)
    return CameraFrame(
        rgb=np.asarray(image, dtype=np.uint8),
        depth_m=frame.depth_m,
        intrinsic=frame.intrinsic,
        world_from_camera=frame.world_from_camera,
        segmentation=frame.segmentation,
        timestamp=frame.timestamp,
    )


def grounding_selection_frame(
    frame: CameraFrame, grounded: GroundedAction
) -> np.ndarray:
    """Draw the VLM-selected target and destination boxes on the wrist RGB."""

    image = Image.fromarray(frame.rgb.copy(), mode="RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def draw_box(box, label: str, color: tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = box.as_pixels(width, height)
        coordinates = (x1, y1, max(x1, x2 - 1), max(y1, y2 - 1))
        draw.rectangle(coordinates, outline=color, width=3)
        text_box = draw.textbbox((x1, y1), label)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x1, y1), label, fill=color)

    draw_box(grounded.target_bbox, f"target: {grounded.target_label}", (40, 255, 100))
    if grounded.destination_bbox is not None:
        draw_box(
            grounded.destination_bbox,
            f"destination: {grounded.destination_label}",
            (255, 150, 30),
        )
    return np.asarray(image, dtype=np.uint8)


def _project_camera_point(
    point: object, intrinsic: np.ndarray
) -> tuple[float, float] | None:
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all() or value[2] <= 1e-6:
        return None
    projected = intrinsic @ value
    uv = projected[:2] / projected[2]
    if not np.isfinite(uv).all():
        return None
    return float(uv[0]), float(uv[1])


def grasp_prediction_frame(
    frame: CameraFrame,
    grounded: GroundedAction,
    candidates: Sequence[object],
    selected: object,
) -> np.ndarray:
    """Project GraspNet candidates and the selected 6-DoF pose into wrist RGB."""

    image = Image.fromarray(grounding_selection_frame(frame, grounded), mode="RGB")
    draw = ImageDraw.Draw(image)
    intrinsic = np.asarray(frame.intrinsic, dtype=np.float64)
    width, height = image.size
    for candidate in candidates:
        center = _project_camera_point(candidate.world_from_gripper[:3, 3], intrinsic)
        if center is None:
            continue
        u, v = center
        if 0 <= u < width and 0 <= v < height:
            draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(170, 170, 170))

    pose = np.asarray(selected.world_from_gripper, dtype=np.float64)
    center = _project_camera_point(pose[:3, 3], intrinsic)
    if center is not None:
        u, v = center
        draw.ellipse((u - 7, v - 7, u + 7, v + 7), outline=(255, 255, 0), width=3)
        draw.line((u - 9, v, u + 9, v), fill=(255, 255, 0), width=2)
        draw.line((u, v - 9, u, v + 9), fill=(255, 255, 0), width=2)
        axis_colors = ((255, 80, 80), (80, 255, 80), (80, 160, 255))
        for axis, color in enumerate(axis_colors):
            endpoint = _project_camera_point(
                pose[:3, 3] + pose[:3, axis] * 0.035, intrinsic
            )
            if endpoint is not None:
                draw.line((u, v, endpoint[0], endpoint[1]), fill=color, width=3)
        label = f"GraspNet score={selected.score:.3f} width={selected.width_m:.3f}m"
        label_y = max(0, min(height - 12, int(v) + 10))
        text_box = draw.textbbox((2, label_y), label)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((2, label_y), label, fill=(255, 255, 0))
    return np.asarray(image, dtype=np.uint8)


def grounding_bbox_catalog(
    frame: CameraFrame, segmentation_ids: Mapping[int, str]
) -> str:
    """Return exact visible instance boxes for the VLM to select and echo."""

    if frame.segmentation is None:
        raise SemanticValidationError("grounding catalog requires segmentation")
    height, width = frame.segmentation.shape
    entries = []
    for segmentation_id, instance_id in sorted(segmentation_ids.items()):
        rows, columns = np.nonzero(frame.segmentation == segmentation_id)
        if len(rows) == 0:
            continue
        coordinates = (
            columns.min() / width,
            rows.min() / height,
            (columns.max() + 1) / width,
            (rows.max() + 1) / height,
        )
        entries.append(
            f"{instance_id}=[{coordinates[0]:.4f},{coordinates[1]:.4f},"
            f"{coordinates[2]:.4f},{coordinates[3]:.4f}]"
        )
    return "Authoritative simulator bbox_xyxy_norm catalog: " + "; ".join(entries)


def validate_panda_plan(
    plan: PlanningEval, visible_instances: Mapping[str, str]
) -> None:
    if not plan.tasks:
        raise SemanticValidationError("planning returned no tasks")
    steps = tuple(task.step for task in plan.tasks)
    if steps != tuple(range(1, len(steps) + 1)):
        raise SemanticValidationError("Panda plan steps must be contiguous")
    visible_graspables = {
        instance_id
        for instance_id in GRASPABLE_INSTANCE_IDS
        if instance_id in visible_instances
    }
    if len(plan.tasks) > len(visible_graspables):
        raise SemanticValidationError(
            "planning returned more tasks than visible graspable objects"
        )
    used_targets: set[str] = set()
    for task in plan.tasks:
        targets = {
            label
            for label in visible_graspables
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])", task.action)
        }
        destinations = {
            label
            for label in DESTINATION_INSTANCE_IDS
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])", task.action)
        }
        if len(targets) != 1:
            raise SemanticValidationError(
                f"planning task {task.step} must name exactly one canonical object"
            )
        if len(destinations) != 1:
            raise SemanticValidationError(
                f"planning task {task.step} must name exactly one canonical destination"
            )
        target = next(iter(targets))
        if target in used_targets:
            raise SemanticValidationError(
                f"planning task {task.step} repeats target {target}"
            )
        used_targets.add(target)


def atomic_subtask_context(task: object) -> str:
    """Return the only command context used after compound planning completes."""

    return (
        f"Atomic subtask {task.step}: {task.action}\n"
        f"Completion criterion: {task.checker}"
    )


def resolve_panda_action(
    grounded: GroundedAction,
    frame: CameraFrame,
    segmentation_ids: Mapping[int, str],
    used_targets: set[str],
) -> ResolvedAction:
    if grounded.destination_label not in DESTINATION_INSTANCE_IDS:
        raise SemanticValidationError(
            f"destination must be one of {DESTINATION_INSTANCE_IDS}"
        )
    if grounded.target_label not in GRASPABLE_INSTANCE_IDS:
        raise SemanticValidationError(
            f"target must be one of {GRASPABLE_INSTANCE_IDS}"
        )
    if grounded.destination_bbox is None:
        raise SemanticValidationError("Panda destination bbox is required")
    if frame.segmentation is None:
        raise SemanticValidationError("Panda grounding requires segmentation")
    target = match_instance(grounded.target_bbox, frame.segmentation, segmentation_ids)
    if target != grounded.target_label:
        raise SemanticValidationError(
            f"target bbox resolves to {target}, not {grounded.target_label}"
        )
    destination = match_instance(
        grounded.destination_bbox, frame.segmentation, segmentation_ids
    )
    if destination != grounded.destination_label:
        raise SemanticValidationError(
            f"destination bbox resolves to {destination}, not {grounded.destination_label}"
        )
    if target in used_targets:
        raise SemanticValidationError(f"target was already completed: {target}")
    return ResolvedAction(
        task_step=grounded.task_step,
        target_instance_id=target,
        target_label=grounded.target_label,
        destination_instance_id=destination,
        destination_label=grounded.destination_label,
        target_bbox=grounded.target_bbox,
        destination_bbox=grounded.destination_bbox,
    )


def _select_camera_grasp(
    candidates: object,
    target_points_camera: object,
    world_from_camera: object,
    *,
    target_instance_id: str | None = None,
    candidate_index: int = 0,
):
    target = np.asarray(target_points_camera, dtype=np.float64)
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
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index < 0
    ):
        raise GraspNetUnavailableError("candidate_index must be nonnegative")
    association_margin_m = 0.035
    boxlike_ids = {"red_cube", "blue_cube", "yellow_block", "purple_block"}
    max_tilt_from_vertical_deg = (
        60.0 if target_instance_id in boxlike_ids else 75.0
    )
    preferred_vertical_deg = 20.0
    min_down_alignment = float(
        np.cos(np.deg2rad(max_tilt_from_vertical_deg))
    )
    target_low = np.min(target, axis=0) - association_margin_m
    target_high = np.max(target, axis=0) + association_margin_m
    ranked = []
    for candidate in candidates:
        position = np.asarray(candidate.world_from_gripper[:3, 3], dtype=np.float64)
        if not candidate.collision_free or candidate.width_m > 0.081:
            continue
        if not np.all((position >= target_low) & (position <= target_high)):
            continue
        world_from_ee = compose_panda_world_ee(
            world_from_camera, candidate.world_from_gripper
        )
        tool_axis = world_from_ee[:3, 2]
        norm = float(np.linalg.norm(tool_axis))
        if norm == 0:
            continue
        down_alignment = -float(tool_axis[2] / norm)
        if down_alignment < min_down_alignment:
            continue
        tilt_angle_deg = float(
            np.rad2deg(np.arccos(np.clip(down_alignment, -1.0, 1.0)))
        )
        target_center_distance = float(
            np.linalg.norm(position - np.mean(target, axis=0))
        )
        ranked.append(
            (
                0 if tilt_angle_deg <= preferred_vertical_deg else 1,
                -float(candidate.score),
                tilt_angle_deg,
                target_center_distance,
                candidate,
            )
        )
    ranked.sort(key=lambda item: item[:4])
    if not ranked:
        raise GraspNetUnavailableError(
            "no target-associated, collision-free Panda grasp satisfies the "
            "0.081 m width, relaxed 0.035 m association, and "
            f"{max_tilt_from_vertical_deg:.0f} degree vertical-approach limits"
        )
    # A new point cloud is inferred on every retry. If it exposes fewer
    # candidates than the requested rank, retry its last remaining candidate;
    # otherwise advance to a genuinely different pose.
    return ranked[min(candidate_index, len(ranked) - 1)][4]


class PandaAgentRuntime:
    def __init__(
        self,
        *,
        config: RuntimeConfig,
        scene: PandaScene,
        vlm: PandaVLM,
        grasp_provider: GraspProvider,
        skill: PandaAtomicSkill,
        artifacts: ArtifactWriter | None = None,
        event_callback: Callable[[RuntimeEvent], None] | None = None,
        image_callback: Callable[[str, np.ndarray, Path | None], None] | None = None,
        grasp_3d_callback: Callable[
            [np.ndarray, np.ndarray, Sequence[object], object, Path | None], None
        ]
        | None = None,
    ) -> None:
        self.config = config
        self.scene = scene
        self.vlm = vlm
        self.grasp_provider = grasp_provider
        self.skill = skill
        self.artifacts = artifacts or ArtifactWriter(None)
        self.event_callback = event_callback
        self.image_callback = image_callback
        self.grasp_3d_callback = grasp_3d_callback

    def _present_image(self, kind: str, name: str, rgb: np.ndarray) -> None:
        path = self.artifacts.write_rgb(name, rgb)
        if self.image_callback is not None:
            self.image_callback(kind, rgb, path)

    def _emit(
        self, kind: RuntimeEventType, message: str, step: int | None = None
    ) -> None:
        if self.event_callback is not None:
            self.event_callback(RuntimeEvent(kind, message, step))

    def _result(
        self, success: bool, tasks: list[TaskResult], message: str
    ) -> RuntimeResult:
        self._emit(
            RuntimeEventType.SUCCEEDED if success else RuntimeEventType.FAILED,
            message,
        )
        return RuntimeResult(
            success=success,
            task_results=tuple(tasks),
            message=message,
            is_mock=self.config.mode == "mock",
        )

    def _recover_pick_failure(self, failure: str) -> tuple[str, bool]:
        try:
            self.skill.recover_after_failed_pick()
        except (AgenticManipulationError, KeyError, TypeError, ValueError) as exc:
            return f"{failure}; recovery failed: {exc}", False
        return failure, True

    def _return_home_after_failure(self, failure: str) -> str:
        try:
            self.skill.return_home()
        except (AgenticManipulationError, KeyError, TypeError, ValueError) as exc:
            return f"{failure}; home recovery failed: {exc}"
        return failure

    def _attempt(
        self,
        command: str,
        task: object,
        used_targets: set[str],
        attempt: int,
    ) -> tuple[ResolvedAction, GroundedAction, np.ndarray]:
        self._emit(RuntimeEventType.ACTION, f"grounding attempt {attempt}", task.step)
        frame = self.scene.capture()
        segmentation_ids = self.scene.segmentation_ids()
        artifact_prefix = f"step_{task.step}_attempt_{attempt}"
        annotated = annotated_grounding_frame(frame, segmentation_ids)
        self.artifacts.write_frame(frame, name=f"{artifact_prefix}_grounding_raw")
        self.artifacts.write_frame(
            annotated, name=f"{artifact_prefix}_grounding_annotated"
        )
        catalog = grounding_bbox_catalog(frame, segmentation_ids)
        grounding_task = replace(
            task,
            action=f"{task.action}\n{catalog}",
        )
        grounded = self.vlm.ground(
            command,
            grounding_task,
            annotated,
        )
        self.artifacts.write_json(
            f"{artifact_prefix}_action", asdict(grounded)
        )
        if grounded.task_step != task.step:
            raise SemanticValidationError(
                f"action step {grounded.task_step} does not match task {task.step}"
            )
        resolved = resolve_panda_action(
            grounded, frame, segmentation_ids, used_targets
        )
        self._present_image(
            "grounding",
            f"{artifact_prefix}_grounding_selection",
            grounding_selection_frame(frame, grounded),
        )
        self._present_image(
            "depth",
            f"{artifact_prefix}_depth_gray",
            depth_grayscale_rgb(frame.depth_m),
        )
        instance_to_seg = {
            instance_id: segmentation_id
            for segmentation_id, instance_id in segmentation_ids.items()
        }
        segmentation_id = instance_to_seg[resolved.target_instance_id]
        target_points = backproject_camera(
            frame, resolved.target_bbox, segmentation_id
        )
        full_workspace_points = backproject_camera(frame)
        if len(target_points) == 0 or len(full_workspace_points) == 0:
            raise SemanticValidationError("grounded point cloud is empty")
        workspace_points = crop_local_workspace_camera(
            full_workspace_points,
            target_points,
            frame.world_from_camera,
        )
        self.artifacts.write_array(
            f"{artifact_prefix}_target_points_camera", target_points
        )
        self.artifacts.write_array(
            f"{artifact_prefix}_workspace_points_camera", workspace_points
        )
        self._emit(
            RuntimeEventType.GRASPING,
            f"GraspNet for {resolved.target_instance_id}",
            task.step,
        )
        candidates = self.grasp_provider.predict(target_points, workspace_points)
        self.artifacts.write_grasps(
            candidates, name=f"{artifact_prefix}_grasps"
        )
        selected = _select_camera_grasp(
            candidates,
            target_points,
            frame.world_from_camera,
            target_instance_id=resolved.target_instance_id,
            candidate_index=attempt - 1,
        )
        self._present_image(
            "graspnet",
            f"{artifact_prefix}_grasp_prediction",
            grasp_prediction_frame(frame, grounded, candidates, selected),
        )
        if self.grasp_3d_callback is not None:
            run_dir = self.artifacts.run_dir
            snapshot = (
                None
                if run_dir is None
                else run_dir / f"{artifact_prefix}_grasp_prediction_3d.png"
            )
            self.grasp_3d_callback(
                target_points,
                workspace_points,
                candidates,
                selected,
                snapshot,
            )
        world_from_ee = compose_panda_world_ee(
            frame.world_from_camera, selected.world_from_gripper
        )
        self._emit(
            RuntimeEventType.GRASPING,
            "selected grasp camera position "
            f"{selected.world_from_gripper[:3, 3].round(5).tolist()}; "
            f"world EE position {world_from_ee[:3, 3].round(5).tolist()}",
            task.step,
        )
        self.artifacts.write_json(
            f"step_{task.step}_attempt_{attempt}_grasp",
            {
                "target": resolved.target_instance_id,
                "destination": resolved.destination_instance_id,
                "target_point_count": len(target_points),
                "workspace_point_count": len(workspace_points),
                "camera_from_grasp": selected.world_from_gripper.tolist(),
                "world_from_ee": world_from_ee.tolist(),
                "score": selected.score,
                "width_m": selected.width_m,
            },
        )
        return resolved, grounded, world_from_ee

    def run(self, command: str) -> RuntimeResult:
        if not isinstance(command, str) or not command.strip():
            return self._result(False, [], "command must not be blank")
        results: list[TaskResult] = []
        used_targets: set[str] = set()
        try:
            self._emit(RuntimeEventType.OBSERVING, "capturing doable frame")
            initial = self.scene.capture()
            self.artifacts.write_frame(initial, name="initial")
            doable = self.vlm.evaluate_doable(command, initial)
            self._emit(RuntimeEventType.DOABLE, doable.thought)
            if not doable.status:
                return self._result(False, results, f"task is not doable: {doable.thought}")
            plan = self.vlm.plan(command, self.scene.capture())
            plan = canonicalize_panda_plan(
                command, plan, self.scene.visible_instances()
            )
            validate_panda_plan(plan, self.scene.visible_instances())
            self.artifacts.write_json("plan", asdict(plan))
            self._emit(RuntimeEventType.PLANNING, f"planned {len(plan.tasks)} tasks")
        except (AgenticManipulationError, KeyError, TypeError, ValueError) as exc:
            return self._result(False, results, f"planning failed: {exc}")

        for task in plan.tasks:
            subtask_context = atomic_subtask_context(task)
            self.artifacts.write_json(
                f"step_{task.step}_context",
                {"task_step": task.step, "context": subtask_context},
            )
            try:
                task_scene = self.vlm.evaluate_doable(
                    subtask_context, self.scene.capture()
                )
            except (AgenticManipulationError, KeyError, TypeError, ValueError) as exc:
                failure = f"task scene description failed: {exc}"
                results.append(TaskResult(task.step, False, (failure,), 0))
                return self._result(False, results, f"step {task.step} failed: {failure}")
            self._emit(
                RuntimeEventType.TASK_STARTED,
                f"{task.action}; {task_scene.thought}",
                task.step,
            )
            if not task_scene.status:
                failure = f"task is not currently doable: {task_scene.thought}"
                results.append(TaskResult(task.step, False, (failure,), 0))
                return self._result(False, results, f"step {task.step} failed: {failure}")
            last_failure = "unknown grasp failure"
            completed = False
            for attempt in range(1, self.config.max_retries + 2):
                pick_started = False
                try:
                    resolved, grounded, world_from_ee = self._attempt(
                        subtask_context, task, used_targets, attempt
                    )
                    self._emit(RuntimeEventType.EXECUTING, "atomic pick", task.step)
                    pick_started = True
                    pick = self.skill.pick(
                        resolved.target_instance_id, world_from_ee
                    )
                    if not pick.success:
                        last_failure = pick.failure_reason or "atomic pick failed"
                        last_failure, recovered = self._recover_pick_failure(
                            last_failure
                        )
                        if not recovered:
                            break
                        if attempt <= self.config.max_retries:
                            self._emit(RuntimeEventType.RETRYING, last_failure, task.step)
                            continue
                        break
                    self._emit(
                        RuntimeEventType.EXECUTING,
                        f"atomic place into {resolved.destination_instance_id}",
                        task.step,
                    )
                    try:
                        place = self.skill.place(
                            resolved.target_instance_id,
                            resolved.destination_instance_id,
                        )
                    except (
                        AgenticManipulationError,
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        failure = self._return_home_after_failure(str(exc))
                        results.append(
                            TaskResult(task.step, False, (failure,), attempt)
                        )
                        return self._result(
                            False,
                            results,
                            f"step {task.step} failed during place: {failure}",
                        )
                    if not place.success:
                        failure = self._return_home_after_failure(
                            place.failure_reason or "atomic place failed"
                        )
                        results.append(TaskResult(task.step, False, (failure,), attempt))
                        return self._result(False, results, f"step {task.step} failed: {failure}")
                    place_frame = self.scene.capture()
                    self.artifacts.write_frame(
                        place_frame,
                        name=f"step_{task.step}_attempt_{attempt}_place_check",
                    )
                    place_visual = self.vlm.check_place(
                        subtask_context, task, grounded, place_frame
                    )
                    self._emit(RuntimeEventType.CHECKING, "place checker", task.step)
                    failures = []
                    if not place_visual.status:
                        failures.append(f"visual place false: {place_visual.thought}")
                    if not self.scene.is_in_bin(
                        resolved.target_instance_id, resolved.destination_instance_id
                    ):
                        failures.append("target is not inside destination bin")
                    if not self.scene.is_released(resolved.target_instance_id):
                        failures.append("target is still held")
                    if failures:
                        results.append(
                            TaskResult(task.step, False, tuple(failures), attempt)
                        )
                        return self._result(
                            False,
                            results,
                            f"step {task.step} place check failed: {'; '.join(failures)}",
                        )
                    results.append(TaskResult(task.step, True, (), attempt))
                    used_targets.add(resolved.target_instance_id)
                    self._emit(
                        RuntimeEventType.TASK_COMPLETED,
                        f"{place_visual.thought}; {resolved.target_instance_id} "
                        f"is released inside {resolved.destination_instance_id}",
                        task.step,
                    )
                    completed = True
                    break
                except (AgenticManipulationError, KeyError, TypeError, ValueError) as exc:
                    last_failure = str(exc)
                    if pick_started:
                        last_failure, recovered = self._recover_pick_failure(
                            last_failure
                        )
                        if not recovered:
                            break
                    if attempt <= self.config.max_retries:
                        self._emit(RuntimeEventType.RETRYING, last_failure, task.step)
                        continue
                    break
            if not completed:
                results.append(
                    TaskResult(
                        task.step,
                        False,
                        (last_failure,),
                        self.config.max_retries + 1,
                    )
                )
                return self._result(
                    False, results, f"step {task.step} failed: {last_failure}"
                )
        return self._result(True, results, "任务完成，还需要什么？")
