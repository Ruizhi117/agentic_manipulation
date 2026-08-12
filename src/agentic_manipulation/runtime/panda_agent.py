"""Closed-loop Panda runtime with per-attempt perception and dual checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from agentic_manipulation.config import RuntimeConfig
from agentic_manipulation.control.panda_atomic import (
    AtomicTransferPlan,
    AtomicTransferReport,
)
from agentic_manipulation.demo.panda_calibration import compose_panda_world_ee
from agentic_manipulation.envs.ee_camera_scene import (
    DESTINATION_INSTANCE_IDS,
    GRASPABLE_INSTANCE_IDS,
)
from agentic_manipulation.errors import (
    AgenticManipulationError,
    GraspNetUnavailableError,
    MotionStageError,
    SemanticValidationError,
)
from agentic_manipulation.models.graspnet import GraspProvider
from agentic_manipulation.perception.depth_segmentation import DepthRegion, segment_depth
from agentic_manipulation.perception.pointcloud import (
    backproject_camera,
    crop_local_workspace_camera,
)
from agentic_manipulation.perception.depth import depth_grayscale_rgb
from agentic_manipulation.runtime.artifacts import ArtifactWriter
from agentic_manipulation.runtime.semantics import ResolvedAction
from agentic_manipulation.types import (
    BBox,
    CameraFrame,
    GroundedAction,
    PlannedTask,
    PlanningEval,
    RegionClassification,
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
_FOUR_OBJECT_GROUPS = (
    "four objects",
    "four items",
    "4 objects",
    "4 items",
    "四个物体",
    "四个物品",
    "4个物体",
    "4个物品",
)
_ANY_DESTINATION_GROUPS = (
    "any bin",
    "any box",
    "either bin",
    "任意盒子",
    "任意一个盒子",
    "任何盒子",
    "任一盒子",
    "任意箱子",
)
_SCENE_INSTANCE_CATALOG = {
    label: label
    for label in GRASPABLE_INSTANCE_IDS + DESTINATION_INSTANCE_IDS
}
_TARGET_RGB_REFERENCES = {
    "red_cube": np.array([230.0, 31.0, 26.0]),
    "blue_cube": np.array([26.0, 77.0, 230.0]),
    "yellow_block": np.array([242.0, 191.0, 20.0]),
    "purple_block": np.array([140.0, 51.0, 191.0]),
}
_TARGET_RGB_MAX_DISTANCE = {
    "red_cube": 90.0,
    "blue_cube": 90.0,
    "yellow_block": 70.0,
    "purple_block": 90.0,
}
_MIN_TARGET_PLACE_PIXELS = 20
_MIN_TARGET_INSIDE_RATIO = 0.8


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
    if target in GRASPABLE_INSTANCE_IDS:
        grouped = _first_alias_position(text, _BLOCK_GROUPS)
        if grouped is not None:
            return grouped
    return _first_alias_position(text, _ALL_OBJECT_GROUPS)


def _requested_targets(
    command: str,
    visible_graspables: set[str],
) -> tuple[str, ...]:
    text = _semantic_text(command)
    requested = set(_labels_in_text(command, _OBJECT_ALIASES))
    if _has_any(text, _FOUR_OBJECT_GROUPS):
        requested.update(GRASPABLE_INSTANCE_IDS)
    elif _has_any(text, _ALL_QUANTIFIERS):
        if _has_any(text, _ALL_OBJECT_GROUPS):
            requested.update(GRASPABLE_INSTANCE_IDS)
        else:
            if _has_any(text, _BLOCK_GROUPS):
                requested.update(GRASPABLE_INSTANCE_IDS)
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
    perceived_catalog: Mapping[str, str],
) -> PlanningEval:
    """Normalize common Chinese/English aliases and expand quantified groups."""

    visible_graspables = {
        label for label in GRASPABLE_INSTANCE_IDS if label in perceived_catalog
    }
    requested = _requested_targets(command, visible_graspables)
    destination_mentions = _destination_mentions(command)
    command_destinations = tuple(dict.fromkeys(label for _, label in destination_mentions))
    visible_destinations = tuple(
        label
        for label in DESTINATION_INSTANCE_IDS
        if label in perceived_catalog
    )
    if (
        requested
        and _has_any(_semantic_text(command), _ANY_DESTINATION_GROUPS)
        and visible_destinations
    ):
        assignments = [
            (target, visible_destinations[index % len(visible_destinations)])
            for index, target in enumerate(requested)
        ]
    elif requested and destination_mentions:
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

class PandaVLM(Protocol):
    def evaluate_doable(self, command: str, frame: CameraFrame): ...

    def plan(self, command: str, frame: CameraFrame): ...

    def ground(self, command: str, task: object, frame: CameraFrame): ...

    def classify_regions(
        self, frame: CameraFrame, region_bboxes: Sequence[BBox]
    ) -> tuple[RegionClassification, ...]: ...

    def check_grasp(
        self, command: str, task: object, grounded: GroundedAction, frame: CameraFrame
    ): ...

    def check_place(
        self, command: str, task: object, grounded: GroundedAction, frame: CameraFrame
    ): ...


@dataclass(frozen=True)
class _RegionGeometry:
    point_count: int
    center_camera_m: tuple[float, float, float]
    center_world_m: tuple[float, float, float]


@dataclass(frozen=True)
class _RGBPlaceEvidence:
    status: bool | None
    target_pixel_count: int
    inside_pixel_count: int
    inside_ratio: float


def _rgb_target_inside_destination(
    rgb: np.ndarray, target_label: str, destination_bbox: BBox
) -> _RGBPlaceEvidence:
    """Check target-color concentration in a destination using RGB only."""
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise SemanticValidationError("place RGB image must have shape (H, W, 3)")
    try:
        reference = _TARGET_RGB_REFERENCES[target_label]
    except KeyError as exc:
        raise SemanticValidationError(
            f"no RGB reference for target {target_label}"
        ) from exc
    color_distance = np.linalg.norm(image - reference, axis=2)
    target_mask = color_distance <= _TARGET_RGB_MAX_DISTANCE[target_label]
    target_pixel_count = int(np.count_nonzero(target_mask))
    height, width = image.shape[:2]
    x1, y1, x2, y2 = destination_bbox.as_pixels(width, height)
    inside_pixel_count = int(np.count_nonzero(target_mask[y1:y2, x1:x2]))
    inside_ratio = (
        inside_pixel_count / target_pixel_count if target_pixel_count else 0.0
    )
    status = None
    if target_pixel_count >= _MIN_TARGET_PLACE_PIXELS:
        status = inside_ratio >= _MIN_TARGET_INSIDE_RATIO
    return _RGBPlaceEvidence(
        status=status,
        target_pixel_count=target_pixel_count,
        inside_pixel_count=inside_pixel_count,
        inside_ratio=inside_ratio,
    )


def _region_geometry(
    frame: CameraFrame, regions: Sequence[DepthRegion]
) -> tuple[_RegionGeometry, ...]:
    """Reconstruct a calibrated 3D center for every depth region."""
    transform = np.asarray(frame.world_from_camera, dtype=np.float64)
    result = []
    for region in regions:
        points_camera = backproject_camera(
            frame, region.bbox, pixel_mask=region.mask
        ).astype(np.float64, copy=False)
        if len(points_camera) == 0:
            raise SemanticValidationError("depth region point cloud is empty")
        homogeneous = np.column_stack(
            (points_camera, np.ones(len(points_camera), dtype=np.float64))
        )
        points_world = (transform @ homogeneous.T).T[:, :3]
        center_camera = np.median(points_camera, axis=0)
        center_world = np.median(points_world, axis=0)
        result.append(
            _RegionGeometry(
                point_count=len(points_camera),
                center_camera_m=tuple(float(value) for value in center_camera),
                center_world_m=tuple(float(value) for value in center_world),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class _RegionPerception:
    frame: CameraFrame
    regions: tuple[DepthRegion, ...]
    inventory: tuple[RegionClassification, ...]
    geometry: tuple[_RegionGeometry, ...]
    annotated_rgb: np.ndarray

    def vlm_frame(self) -> CameraFrame:
        return CameraFrame(
            rgb=self.annotated_rgb.copy(),
            depth_m=self.frame.depth_m.copy(),
            intrinsic=self.frame.intrinsic.copy(),
            world_from_camera=self.frame.world_from_camera.copy(),
            segmentation=None,
            timestamp=self.frame.timestamp,
        )

    def command_context(self, command: str) -> str:
        rows = []
        for region, item, geometry in zip(
            self.regions, self.inventory, self.geometry, strict=True
        ):
            rows.append(
                f"R{item.region_id}: color={item.color}, kind={item.kind}, "
                f"label={item.label}, image_location={item.image_location}, "
                f"center_xy_norm={np.round(region.centroid_norm, 4).tolist()}, "
                f"center_camera_m={np.round(geometry.center_camera_m, 4).tolist()}, "
                f"center_world_m={np.round(geometry.center_world_m, 4).tolist()}"
            )
        return command + "\nCurrent RGB-D region inventory:\n" + "\n".join(rows)


class PandaAtomicSkill(Protocol):
    def can_pick(self, world_from_ee: np.ndarray) -> bool: ...

    def plan_transfer(
        self,
        world_from_ee: np.ndarray,
        nominal_world_from_release: np.ndarray,
    ) -> AtomicTransferPlan: ...

    def execute(
        self,
        instance_id: str,
        plan: AtomicTransferPlan,
        confirm_grasp: Callable[[], bool],
    ) -> AtomicTransferReport: ...

    def recover_after_failed_pick(self) -> None: ...

    def return_home(self) -> None: ...


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


def validate_panda_plan(
    plan: PlanningEval, perceived_catalog: Mapping[str, str]
) -> None:
    if not plan.tasks:
        raise SemanticValidationError("planning returned no tasks")
    steps = tuple(task.step for task in plan.tasks)
    if steps != tuple(range(1, len(steps) + 1)):
        raise SemanticValidationError("Panda plan steps must be contiguous")
    visible_graspables = {
        instance_id
        for instance_id in GRASPABLE_INSTANCE_IDS
        if instance_id in perceived_catalog
    }
    visible_destinations = {
        instance_id
        for instance_id in DESTINATION_INSTANCE_IDS
        if instance_id in perceived_catalog
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
            for label in visible_destinations
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
    target = grounded.target_label
    destination = grounded.destination_label
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


def _task_labels(task: object) -> tuple[str, str]:
    """Extract canonical target and destination labels from a *task* action.

    The action is assumed to contain exactly one target label from
    ``GRASPABLE_INSTANCE_IDS`` and one destination label from
    ``DESTINATION_INSTANCE_IDS``.
    """
    action_text = getattr(task, "action", "")
    target = ""
    destination = ""
    for label in GRASPABLE_INSTANCE_IDS:
        if label in action_text:
            target = label
            break
    for label in DESTINATION_INSTANCE_IDS:
        if label in action_text:
            destination = label
            break
    if not target or not destination:
        raise SemanticValidationError(
            f"cannot extract target/destination labels from task: {action_text!r}"
        )
    return target, destination


def _draw_depth_regions(
    rgb: np.ndarray, labeled: list[tuple[BBox, str]]
) -> np.ndarray:
    """Draw coloured bounding boxes and labels for depth-segmented regions."""
    image = Image.fromarray(rgb.copy(), mode="RGB")
    draw = ImageDraw.Draw(image)
    colors = (
        (255, 230, 40), (40, 255, 100), (255, 90, 220),
        (80, 220, 255), (255, 140, 40), (180, 100, 255),
        (255, 255, 255), (255, 80, 120),
    )
    height, width = rgb.shape[:2]
    for idx, (bbox, label) in enumerate(labeled):
        color = colors[idx % len(colors)]
        x1, y1, x2, y2 = bbox.as_pixels(width, height)
        draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)),
                       outline=color, width=2)
        text_box = draw.textbbox((x1, y1), label)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x1, y1), label, fill=color)
    return np.asarray(image, dtype=np.uint8)


def _select_camera_grasp(
    candidates: object,
    target_points_camera: object,
    world_from_camera: object,
    *,
    target_bbox: BBox,
    intrinsic: object,
    image_hw: tuple[int, int],
    reachable: Callable[[np.ndarray], bool],
    candidate_index: int = 0,
):
    """Return an in-box, reachable GraspNet candidate, preferring verticality."""
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

    transform = np.asarray(world_from_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise GraspNetUnavailableError(
            "world_from_camera must be a finite 4x4 matrix"
        )
    camera_intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if camera_intrinsic.shape != (3, 3) or not np.isfinite(camera_intrinsic).all():
        raise GraspNetUnavailableError("intrinsic must be a finite 3x3 matrix")
    if (
        not isinstance(image_hw, tuple)
        or len(image_hw) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in image_hw)
    ):
        raise GraspNetUnavailableError("image_hw must contain two positive integers")
    if not isinstance(target_bbox, BBox):
        raise GraspNetUnavailableError("target_bbox must be a normalized BBox")
    if not callable(reachable):
        raise GraspNetUnavailableError("reachable must be callable")
    height, width = image_hw
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = target_bbox.as_pixels(width, height)

    target_low = np.min(target, axis=0) - 0.035
    target_high = np.max(target, axis=0) + 0.035
    target_center = np.median(target, axis=0)
    max_predicted_width_m = 0.085
    max_tilt_from_vertical_deg = 60.0
    min_acceptable = float(np.cos(np.deg2rad(max_tilt_from_vertical_deg)))

    ranked: list[tuple[float, float, float, object]] = []
    for candidate in candidates:
        if (
            not candidate.collision_free
            or candidate.width_m > max_predicted_width_m
        ):
            continue
        camera_position = np.asarray(
            candidate.world_from_gripper[:3, 3], dtype=np.float64
        )
        if not np.all((camera_position >= target_low) & (camera_position <= target_high)):
            continue
        projected = _project_camera_point(camera_position, camera_intrinsic)
        if projected is None:
            continue
        u, v = projected
        projection_epsilon_px = 1e-6
        if not (
            bbox_x1 - projection_epsilon_px <= u < bbox_x2 + projection_epsilon_px
            and bbox_y1 - projection_epsilon_px <= v < bbox_y2 + projection_epsilon_px
        ):
            continue
        world_from_ee = compose_panda_world_ee(
            transform, candidate.world_from_gripper
        )
        if not reachable(world_from_ee):
            continue
        tool_axis = np.asarray(
            world_from_ee[:3, 2], dtype=np.float64
        )
        norm = float(np.linalg.norm(tool_axis))
        if norm == 0:
            continue
        down_alignment = -float(tool_axis[2] / norm)
        if down_alignment < min_acceptable:
            continue
        tilt_deg = float(np.rad2deg(np.arccos(np.clip(down_alignment, -1.0, 1.0))))
        center_distance = float(np.linalg.norm(camera_position - target_center))
        ranked.append((tilt_deg, center_distance, -float(candidate.score), candidate))

    if not ranked:
        raise GraspNetUnavailableError(
            "no detection-box-associated, collision-free, IK-reachable Panda grasp "
            "satisfies the 0.085 m predicted-width and 60° vertical-approach limits"
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[min(candidate_index, len(ranked) - 1)][3]


def _camera_points_world(
    points_camera: object, world_from_camera: object
) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    transform = np.asarray(world_from_camera, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) == 0
        or not np.isfinite(points).all()
    ):
        raise SemanticValidationError(
            "RGB-D point cloud must be a finite nonempty (N, 3) array"
        )
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SemanticValidationError(
            "world_from_camera must be a finite 4x4 matrix"
        )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (transform @ homogeneous.T).T[:, :3]


def _free_destination_xy(
    target_world: np.ndarray,
    destination_world: np.ndarray,
) -> np.ndarray:
    """Choose a low, interior RGB-D slot instead of an occupied bin center."""

    center = np.median(destination_world[:, :2], axis=0)
    if len(destination_world) < 32:
        return center
    low = np.quantile(destination_world[:, :2], 0.05, axis=0)
    high = np.quantile(destination_world[:, :2], 0.95, axis=0)
    span = high - low
    target_span = np.ptp(target_world[:, :2], axis=0)
    wall_margin = target_span / 2.0 + 0.015
    slot_low = low + wall_margin
    slot_high = high - wall_margin
    if np.any(slot_low >= slot_high):
        return center
    center = np.clip(center, slot_low, slot_high)
    max_offset = np.minimum(center - slot_low, slot_high - center)
    offset = np.minimum(span * 0.22, max_offset)
    candidates = np.array(
        [
            center,
            center + [-offset[0], 0.0],
            center + [offset[0], 0.0],
            center + [0.0, -offset[1]],
            center + [0.0, offset[1]],
            center + [-offset[0], -offset[1]],
            center + [-offset[0], offset[1]],
            center + [offset[0], -offset[1]],
            center + [offset[0], offset[1]],
        ],
        dtype=np.float64,
    )
    local_radius = max(0.02, float(np.max(target_span)) / 2.0 + 0.005)
    clear_surface_height = (
        float(np.quantile(destination_world[:, 2], 0.05)) + 0.015
    )
    scores = []
    for index, candidate in enumerate(candidates):
        distances = np.linalg.norm(
            destination_world[:, :2] - candidate, axis=1
        )
        local_heights = destination_world[distances <= local_radius, 2]
        local_peak = (
            float(np.quantile(local_heights, 0.90))
            if len(local_heights) >= 8
            else float("inf")
        )
        obstacle_excess = max(0.0, local_peak - clear_surface_height)
        scores.append(
            (
                obstacle_excess > 0.0,
                obstacle_excess,
                float(np.linalg.norm(candidate - center)),
                index,
            )
        )
    return candidates[min(scores)[3]]


def _placement_world_position(
    target_points_camera: object,
    destination_points_camera: object,
    world_from_camera: object,
    world_from_ee: object,
    *,
    clearance_m: float = 0.01,
) -> np.ndarray:
    """Compute an EE placement position from RGB-D geometry only.

    The target-to-EE translation measured at grasp time is preserved while the
    estimated target center is translated to a low, interior destination slot.
    No simulator object pose, bin AABB, or desired orientation is used.
    """
    target_world = _camera_points_world(target_points_camera, world_from_camera)
    destination_world = _camera_points_world(
        destination_points_camera, world_from_camera
    )
    ee = np.asarray(world_from_ee, dtype=np.float64)
    if ee.shape != (4, 4) or not np.isfinite(ee).all():
        raise SemanticValidationError("world_from_ee must be a finite 4x4 matrix")
    if not np.isfinite(clearance_m) or clearance_m < 0:
        raise SemanticValidationError("clearance_m must be nonnegative and finite")

    target_center = np.median(target_world, axis=0)
    destination_center = np.median(destination_world, axis=0)
    destination_center[:2] = _free_destination_xy(
        target_world, destination_world
    )
    target_height = max(0.01, float(np.ptp(target_world[:, 2])))
    destination_floor = float(np.quantile(destination_world[:, 2], 0.05))
    destination_rim = float(np.quantile(destination_world[:, 2], 0.95))
    target_half_height = target_height / 2.0
    # The release pose is intentionally above the RGB-D-estimated rim. Driving
    # the closed fingers toward the bin floor causes contact before the TCP can
    # reach its target, which previously triggered a premature high release.
    destination_center[2] = max(
        destination_floor + target_half_height + clearance_m,
        destination_rim + target_half_height + clearance_m,
    )
    ee_offset_from_target = ee[:3, 3] - target_center
    return destination_center + ee_offset_from_target


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

    def _perceive_regions(
        self,
        frame: CameraFrame,
        *,
        artifact_prefix: str,
        task_step: int | None,
    ) -> _RegionPerception | None:
        """Create one RGB-D-only inventory shared by reasoning and execution."""
        regions = segment_depth(frame)
        if not regions:
            return None
        inventory = self.vlm.classify_regions(
            frame, tuple(region.bbox for region in regions)
        )
        if len(inventory) != len(regions):
            raise SemanticValidationError(
                "VLM region inventory count does not match depth segmentation"
            )
        geometry = _region_geometry(frame, regions)
        labeled: list[tuple[BBox, str]] = []
        inventory_rows: list[dict[str, object]] = []
        for idx, (region, item, region_geometry) in enumerate(
            zip(regions, inventory, geometry, strict=True)
        ):
            if item.region_id != idx:
                raise SemanticValidationError(
                    "VLM region inventory ids do not match depth region order"
                )
            labeled.append((region.bbox, f"R{idx} {item.label} ({item.color})"))
            inventory_rows.append(
                {
                    **asdict(item),
                    "bbox_xyxy_norm": asdict(region.bbox),
                    "center_xy_norm": list(region.centroid_norm),
                    "area_px": region.area_px,
                    "point_count": region_geometry.point_count,
                    "center_camera_m": list(region_geometry.center_camera_m),
                    "center_world_m": list(region_geometry.center_world_m),
                }
            )
            self._emit(
                RuntimeEventType.ACTION,
                f"depth region {idx}: color={item.color}, kind={item.kind}, "
                f"label={item.label}, location={item.image_location}, "
                f"center_xy_norm={np.round(region.centroid_norm, 4).tolist()}, "
                f"center_world_m="
                f"{np.round(region_geometry.center_world_m, 4).tolist()}",
                task_step,
            )
        self.artifacts.write_json(
            f"{artifact_prefix}_region_inventory", {"regions": inventory_rows}
        )
        annotated_rgb = _draw_depth_regions(frame.rgb, labeled)
        self._present_image(
            "perception", f"{artifact_prefix}_depth_regions", annotated_rgb
        )
        return _RegionPerception(frame, regions, inventory, geometry, annotated_rgb)

    def _attempt(
        self,
        command: str,
        task: object,
        used_targets: set[str],
        attempt: int,
        perception: _RegionPerception | None = None,
    ) -> tuple[ResolvedAction, GroundedAction, AtomicTransferPlan]:
        self._emit(RuntimeEventType.ACTION, f"grounding attempt {attempt}", task.step)
        frame = perception.frame if perception is not None else self.scene.capture()
        artifact_prefix = f"step_{task.step}_attempt_{attempt}"
        self.artifacts.write_frame(frame, name=f"{artifact_prefix}_grounding_raw")
        target_region = None
        destination_region = None

        # ── 1. Depth-based object segmentation ──────────────────────────
        if perception is None:
            perception = self._perceive_regions(
                frame, artifact_prefix=artifact_prefix, task_step=task.step
            )
        if perception is None:
            # Fall back to VLM full-scene grounding when depth segmentation
            # finds nothing (e.g. objects too close together).
            grounded = self.vlm.ground(command, task, frame)
        else:
            # ── 2. VLM full-scene region inventory ──────────────────────
            regions = perception.regions
            inventory = perception.inventory

            # ── 4. Match target / destination by label ──────────────────
            target_label, dest_label = _task_labels(task)
            target_matches = [
                region
                for region, item in zip(regions, inventory, strict=True)
                if item.label == target_label
            ]
            destination_matches = [
                region
                for region, item in zip(regions, inventory, strict=True)
                if item.label == dest_label
            ]
            if len(target_matches) != 1:
                raise SemanticValidationError(
                    f"depth segmentation must find one target {target_label}; "
                    f"classified: {[item.label for item in inventory]}"
                )
            if len(destination_matches) != 1:
                raise SemanticValidationError(
                    f"depth segmentation must find one destination {dest_label}; "
                    f"classified: {[item.label for item in inventory]}"
                )
            target_region = target_matches[0]
            destination_region = destination_matches[0]

            grounded = GroundedAction(
                type="action",
                task_step=task.step,
                target_label=target_label,
                target_bbox=target_region.bbox,
                destination_label=dest_label,
                destination_bbox=destination_region.bbox,
            )

        self.artifacts.write_json(
            f"{artifact_prefix}_action", asdict(grounded)
        )
        if grounded.task_step != task.step:
            raise SemanticValidationError(
                f"action step {grounded.task_step} does not match task {task.step}"
            )
        resolved = resolve_panda_action(grounded, used_targets)
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
        # Extract point cloud using the VLM-confirmed bounding box.
        target_points = backproject_camera(
            frame,
            resolved.target_bbox,
            pixel_mask=None if target_region is None else target_region.mask,
        )
        destination_points = backproject_camera(
            frame,
            resolved.destination_bbox,
            pixel_mask=(
                None if destination_region is None else destination_region.mask
            ),
        )
        full_workspace_points = backproject_camera(frame)
        if (
            len(target_points) == 0
            or len(destination_points) == 0
            or len(full_workspace_points) == 0
        ):
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
        self.artifacts.write_array(
            f"{artifact_prefix}_destination_points_camera", destination_points
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
            target_bbox=resolved.target_bbox,
            intrinsic=frame.intrinsic,
            image_hw=frame.rgb.shape[:2],
            reachable=self.skill.can_pick,
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
        placement_position = _placement_world_position(
            target_points,
            destination_points,
            frame.world_from_camera,
            world_from_ee,
        )
        nominal_world_from_release = np.eye(4, dtype=np.float64)
        nominal_world_from_release[:3, :3] = world_from_ee[:3, :3]
        nominal_world_from_release[:3, 3] = placement_position
        transfer_plan = self.skill.plan_transfer(
            world_from_ee, nominal_world_from_release
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
                "destination_point_count": len(destination_points),
                "camera_from_grasp": selected.world_from_gripper.tolist(),
                "world_from_ee": world_from_ee.tolist(),
                "nominal_world_from_release": nominal_world_from_release.tolist(),
                "planned_world_from_release": transfer_plan.release.tolist(),
                "placement_world_position": transfer_plan.release[:3, 3].tolist(),
                "score": selected.score,
                "width_m": selected.width_m,
            },
        )
        return resolved, grounded, transfer_plan

    def run(self, command: str) -> RuntimeResult:
        if not isinstance(command, str) or not command.strip():
            return self._result(False, [], "command must not be blank")
        results: list[TaskResult] = []
        used_targets: set[str] = set()
        try:
            self._emit(RuntimeEventType.OBSERVING, "capturing doable frame")
            initial = self.scene.capture()
            self.artifacts.write_frame(initial, name="initial")
            initial_perception = self._perceive_regions(
                initial, artifact_prefix="initial", task_step=None
            )
            initial_vlm_frame = (
                initial
                if initial_perception is None
                else initial_perception.vlm_frame()
            )
            initial_command = (
                command
                if initial_perception is None
                else initial_perception.command_context(command)
            )
            doable = self.vlm.evaluate_doable(initial_command, initial_vlm_frame)
            self._emit(RuntimeEventType.DOABLE, doable.thought)
            if not doable.status:
                return self._result(False, results, f"task is not doable: {doable.thought}")
            plan = self.vlm.plan(initial_command, initial_vlm_frame)
            perceived_catalog = (
                _SCENE_INSTANCE_CATALOG
                if initial_perception is None
                else {
                    item.label: item.label
                    for item in initial_perception.inventory
                }
            )
            plan = canonicalize_panda_plan(command, plan, perceived_catalog)
            validate_panda_plan(plan, perceived_catalog)
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
                if task.step == 1 and initial_perception is not None:
                    task_perception = initial_perception
                    task_frame = initial_vlm_frame
                else:
                    raw_task_frame = self.scene.capture()
                    task_perception = self._perceive_regions(
                        raw_task_frame,
                        artifact_prefix=f"step_{task.step}_initial",
                        task_step=task.step,
                    )
                    task_frame = (
                        raw_task_frame
                        if task_perception is None
                        else task_perception.vlm_frame()
                    )
                task_command = (
                    subtask_context
                    if task_perception is None
                    else task_perception.command_context(subtask_context)
                )
                task_scene = self.vlm.evaluate_doable(
                    task_command, task_frame
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
                transfer_started = False
                try:
                    (
                        resolved,
                        grounded,
                        transfer_plan,
                    ) = self._attempt(
                        task_command,
                        task,
                        used_targets,
                        attempt,
                        perception=task_perception if attempt == 1 else None,
                    )
                    grasp_visual = None

                    def confirm_grasp() -> bool:
                        nonlocal grasp_visual
                        grasp_frame = self.scene.capture()
                        self.artifacts.write_frame(
                            grasp_frame,
                            name=(
                                f"step_{task.step}_attempt_{attempt}_grasp_check"
                            ),
                        )
                        grasp_visual = self.vlm.check_grasp(
                            subtask_context, task, grounded, grasp_frame
                        )
                        self._emit(
                            RuntimeEventType.CHECKING,
                            "near-field RGB-D grasp checker",
                            task.step,
                        )
                        return grasp_visual.status

                    self._emit(
                        RuntimeEventType.EXECUTING,
                        "atomic pick-check-transfer-release into "
                        f"{resolved.destination_instance_id}",
                        task.step,
                    )
                    transfer_started = True
                    transfer = self.skill.execute(
                        resolved.target_instance_id,
                        transfer_plan,
                        confirm_grasp,
                    )
                    if "home_return" not in transfer.stages:
                        reset_failure, recovered = self._recover_pick_failure(
                            "atomic transfer did not confirm observation-home reset"
                        )
                        if not recovered:
                            last_failure = reset_failure
                            break
                    self._emit(
                        RuntimeEventType.EXECUTING,
                        "observation-home reset confirmed; capturing fresh RGB-D",
                        task.step,
                    )
                    if not transfer.success:
                        last_failure = (
                            f"visual grasp false: {grasp_visual.thought}"
                            if grasp_visual is not None
                            and not grasp_visual.status
                            else transfer.failure_reason
                            or "atomic transfer failed"
                        )
                        if attempt <= self.config.max_retries:
                            self._emit(
                                RuntimeEventType.RETRYING,
                                last_failure,
                                task.step,
                            )
                            continue
                        break
                    place_frame = self.scene.capture()
                    self.artifacts.write_frame(
                        place_frame,
                        name=f"step_{task.step}_attempt_{attempt}_place_check",
                    )
                    place_visual = self.vlm.check_place(
                        subtask_context, task, grounded, place_frame
                    )
                    rgb_place = _rgb_target_inside_destination(
                        place_frame.rgb,
                        resolved.target_instance_id,
                        resolved.destination_bbox,
                    )
                    self.artifacts.write_json(
                        f"step_{task.step}_attempt_{attempt}_place_rgb_evidence",
                        {
                            **asdict(rgb_place),
                            "vlm_status": place_visual.status,
                            "vlm_thought": place_visual.thought,
                        },
                    )
                    self._emit(
                        RuntimeEventType.CHECKING,
                        "place checker: "
                        f"vlm={place_visual.status}, rgb={rgb_place.status}, "
                        f"inside={rgb_place.inside_pixel_count}/"
                        f"{rgb_place.target_pixel_count}",
                        task.step,
                    )
                    failures = []
                    if rgb_place.status is False:
                        failures.append(
                            "RGB target-color evidence is outside destination: "
                            f"{rgb_place.inside_pixel_count}/"
                            f"{rgb_place.target_pixel_count} pixels"
                        )
                    elif rgb_place.status is None and not place_visual.status:
                        failures.append(f"visual place false: {place_visual.thought}")
                    if failures:
                        last_failure = "; ".join(failures)
                        if attempt <= self.config.max_retries:
                            self._emit(
                                RuntimeEventType.RETRYING,
                                last_failure,
                                task.step,
                            )
                            continue
                        results.append(
                            TaskResult(task.step, False, tuple(failures), attempt)
                        )
                        return self._result(
                            False,
                            results,
                            f"step {task.step} place check failed: {last_failure}",
                        )
                    results.append(TaskResult(task.step, True, (), attempt))
                    used_targets.add(resolved.target_instance_id)
                    completion_evidence = (
                        place_visual.thought
                        if place_visual.status
                        else "RGB target-color evidence confirms placement"
                    )
                    self._emit(
                        RuntimeEventType.TASK_COMPLETED,
                        f"{completion_evidence}; {resolved.target_instance_id} "
                        f"is released inside {resolved.destination_instance_id}",
                        task.step,
                    )
                    completed = True
                    break
                except (AgenticManipulationError, KeyError, TypeError, ValueError) as exc:
                    last_failure = str(exc)
                    safe_to_retry = not transfer_started
                    if transfer_started:
                        safe_to_retry = (
                            isinstance(exc, MotionStageError)
                            and exc.stage in {"pregrasp", "approach", "lift"}
                        )
                        last_failure, recovered = self._recover_pick_failure(
                            last_failure
                        )
                        if not recovered:
                            break
                    if safe_to_retry and attempt <= self.config.max_retries:
                        self._emit(RuntimeEventType.RETRYING, last_failure, task.step)
                        continue
                    break
            if not completed:
                results.append(
                    TaskResult(
                        task.step,
                        False,
                        (last_failure,),
                        attempt,
                    )
                )
                return self._result(
                    False, results, f"step {task.step} failed: {last_failure}"
                )
        return self._result(True, results, "任务完成，还需要什么？")
