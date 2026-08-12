"""Qwen3-VL prompts for the Panda four-block/two-bin RGB-D scene."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agentic_manipulation.errors import ModelResponseError
from agentic_manipulation.models.qwen_vl import PHASE_SCHEMAS, object_bbox
from agentic_manipulation.types import (
    BBox,
    CameraFrame,
    CheckerEval,
    DoableEval,
    GroundedAction,
    PlannedTask,
    PlanningEval,
    RegionClassification,
)


PANDA_OBJECT_LABELS = (
    "red_cube",
    "blue_cube",
    "yellow_block",
    "purple_block",
)
PANDA_DESTINATION_LABELS = ("white_bin", "pink_bin")

_REGION_COLOR_LABELS = {
    ("red", "block"): "red_cube",
    ("blue", "block"): "blue_cube",
    ("yellow", "block"): "yellow_block",
    ("purple", "block"): "purple_block",
    ("white", "bin"): "white_bin",
    ("pink", "bin"): "pink_bin",
}
_REGION_COLORS = ("red", "blue", "yellow", "purple", "white", "pink")
_REGION_KINDS = ("block", "bin")
_REGION_RGB_REFERENCES = {
    "red": np.array([230.0, 31.0, 26.0]),
    "blue": np.array([26.0, 77.0, 230.0]),
    "yellow": np.array([242.0, 191.0, 20.0]),
    "purple": np.array([140.0, 51.0, 191.0]),
    "white": np.array([242.0, 242.0, 242.0]),
    "pink": np.array([255.0, 107.0, 166.0]),
}
_RGB_HINT_MAX_DISTANCE = 120.0


def _bbox_image_location(bbox: BBox) -> str:
    """Derive coarse image position from depth geometry, never VLM guessing."""
    center_x = (float(bbox.x1) + float(bbox.x2)) / 2.0
    center_y = (float(bbox.y1) + float(bbox.y2)) / 2.0
    horizontal = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
    if center_y < 1 / 3:
        return f"upper_{horizontal}"
    if center_y > 2 / 3:
        return f"lower_{horizontal}"
    return horizontal


def _rgb_color_hint(
    frame: CameraFrame, bbox: BBox
) -> tuple[str | None, tuple[float, float, float], float]:
    """Return a conservative scene-palette hint from only bbox RGB pixels."""
    height, width = frame.rgb.shape[:2]
    x1, y1, x2, y2 = bbox.as_pixels(width, height)
    pixels = frame.rgb[y1:y2, x1:x2].reshape(-1, 3).astype(np.float64)
    median = np.median(pixels, axis=0)
    distances = {
        color: float(np.linalg.norm(median - reference))
        for color, reference in _REGION_RGB_REFERENCES.items()
    }
    color = min(distances, key=distances.get)
    distance = distances[color]
    hint = color if distance <= _RGB_HINT_MAX_DISTANCE else None
    return hint, tuple(float(value) for value in median), distance


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
    def _numbered_region_frame(
        frame: CameraFrame, region_bboxes: Sequence[BBox]
    ) -> CameraFrame:
        image = Image.fromarray(frame.rgb.copy(), mode="RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        colors = (
            (255, 230, 40),
            (40, 255, 100),
            (255, 90, 220),
            (80, 220, 255),
            (255, 140, 40),
            (180, 100, 255),
        )
        height, width = frame.rgb.shape[:2]
        for region_id, bbox in enumerate(region_bboxes):
            x1, y1, x2, y2 = bbox.as_pixels(width, height)
            color = colors[region_id % len(colors)]
            draw.rectangle(
                (x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)),
                outline=color,
                width=3,
            )
            text = f"R{region_id}"
            text_bbox = draw.textbbox((x1, y1), text, font=font, stroke_width=1)
            draw.rectangle(text_bbox, fill=(0, 0, 0))
            draw.text(
                (x1, y1),
                text,
                fill=color,
                font=font,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
        return CameraFrame(
            rgb=np.asarray(image, dtype=np.uint8),
            depth_m=frame.depth_m.copy(),
            intrinsic=frame.intrinsic.copy(),
            world_from_camera=frame.world_from_camera.copy(),
            segmentation=None,
            timestamp=frame.timestamp,
        )

    @staticmethod
    def _zoomed_region_frame(frame: CameraFrame, bbox: BBox) -> CameraFrame:
        """Enlarge one tight RGB-D region without color interpolation."""
        height, width = frame.rgb.shape[:2]
        x1, y1, x2, y2 = bbox.as_pixels(width, height)
        rgb_crop = frame.rgb[y1:y2, x1:x2]
        depth_crop = frame.depth_m[y1:y2, x1:x2]
        crop_height, crop_width = rgb_crop.shape[:2]
        max_width = max(1, int(width * 0.75))
        max_height = max(1, int(height * 0.75))
        scale = min(max_width / crop_width, max_height / crop_height)
        zoom_width = max(1, round(crop_width * scale))
        zoom_height = max(1, round(crop_height * scale))
        rgb_zoom = np.asarray(
            Image.fromarray(rgb_crop, mode="RGB").resize(
                (zoom_width, zoom_height), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        depth_zoom = np.asarray(
            Image.fromarray(depth_crop).resize(
                (zoom_width, zoom_height), Image.Resampling.NEAREST
            ),
            dtype=frame.depth_m.dtype,
        )
        rgb_canvas = np.zeros_like(frame.rgb)
        depth_canvas = np.zeros_like(frame.depth_m)
        offset_x = (width - zoom_width) // 2
        offset_y = (height - zoom_height) // 2
        rgb_canvas[
            offset_y : offset_y + zoom_height,
            offset_x : offset_x + zoom_width,
        ] = rgb_zoom
        depth_canvas[
            offset_y : offset_y + zoom_height,
            offset_x : offset_x + zoom_width,
        ] = depth_zoom
        return CameraFrame(
            rgb=rgb_canvas,
            depth_m=depth_canvas,
            intrinsic=frame.intrinsic.copy(),
            world_from_camera=frame.world_from_camera.copy(),
            segmentation=None,
            timestamp=frame.timestamp,
        )

    @staticmethod
    def _near_field_grasp_frame(
        frame: CameraFrame, *, max_depth_m: float = 0.30
    ) -> CameraFrame:
        """Mask distant RGB-D so the grasp checker sees only gripper-range evidence."""

        depth = frame.depth_m
        near = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth_m)
        rgb_near = np.zeros_like(frame.rgb)
        depth_near = np.zeros_like(depth)
        rgb_near[near] = frame.rgb[near]
        depth_near[near] = depth[near]
        return CameraFrame(
            rgb=rgb_near,
            depth_m=depth_near,
            intrinsic=frame.intrinsic.copy(),
            world_from_camera=frame.world_from_camera.copy(),
            segmentation=None,
            timestamp=frame.timestamp,
        )

    def classify_regions(
        self, frame: CameraFrame, region_bboxes: Sequence[BBox]
    ) -> tuple[RegionClassification, ...]:
        """Inventory all depth regions from one numbered full-scene RGB-D view."""
        if not region_bboxes:
            return ()
        annotated_frame = self._numbered_region_frame(frame, region_bboxes)
        descriptors = []
        for region_id, bbox in enumerate(region_bboxes):
            descriptors.append(
                f"region {region_id}: bbox_xyxy_norm="
                f"[{bbox.x1:.4f}, {bbox.y1:.4f}, {bbox.x2:.4f}, {bbox.y2:.4f}]"
            )
        prompt = (
            "This is a full view from a Panda wrist camera mounted beside the "
            "Panda robot gripper. The gripper may be visible in the foreground. "
            "The tabletop scene contains exactly four movable blocks and two "
            "open-top bins: one red block, one blue block, one yellow block, one "
            "purple block, one white bin, and one pink bin. Colored boxes marked "
            "R0, R1, ... are regions found only from Image 2 depth geometry. "
            "Use Image 1 color and open-bin shape to classify every numbered depth "
            "region exactly once. Judge the visible color for every region; do not "
            "classify the robot gripper. Exact region position comes from the listed "
            "depth bbox, so only judge region_id, color, and kind.\n"
            + "\n".join(descriptors)
        )
        item_schema = {
            "type": "object",
            "properties": {
                "region_id": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": len(region_bboxes) - 1,
                },
                "color": {"type": "string", "enum": list(_REGION_COLORS)},
                "kind": {"type": "string", "enum": list(_REGION_KINDS)},
            },
            "required": ["region_id", "color", "kind"],
            "additionalProperties": False,
        }
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "const": "doable"},
                "regions": {
                    "type": "array",
                    "items": item_schema,
                    "minItems": len(region_bboxes),
                    "maxItems": len(region_bboxes),
                },
            },
            "required": ["type", "regions"],
            "additionalProperties": False,
        }
        data = self._run("doable", prompt, annotated_frame, schema=schema)
        raw_regions = data.get("regions")
        if not isinstance(raw_regions, list):
            raise ModelResponseError("region inventory regions must be a list")
        global_rows: list[dict[str, object]] = []
        try:
            for raw in raw_regions:
                item = _mapping(raw, "region")
                region_id = int(item["region_id"])
                color = str(item["color"])
                kind = str(item["kind"])
                if not 0 <= region_id < len(region_bboxes):
                    raise ModelResponseError(
                        f"region_id is outside the depth region range: {region_id}"
                    )
                global_rows.append(
                    {"region_id": region_id, "color": color, "kind": kind}
                )
        except ModelResponseError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelResponseError(f"invalid region inventory: {exc}") from exc
        global_rows.sort(key=lambda item: int(item["region_id"]))
        if [int(item["region_id"]) for item in global_rows] != list(
            range(len(region_bboxes))
        ):
            raise ModelResponseError(
                "region inventory must return every region id exactly once"
            )
        verification_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "const": "doable"},
                "color": {"type": "string", "enum": list(_REGION_COLORS)},
                "kind": {"type": "string", "enum": list(_REGION_KINDS)},
            },
            "required": ["type", "color", "kind"],
            "additionalProperties": False,
        }
        verified: list[RegionClassification] = []
        region_checks: list[dict[str, object]] = []
        for region_id, bbox in enumerate(region_bboxes):
            zoomed = self._zoomed_region_frame(frame, bbox)
            rgb_hint, rgb_median, rgb_distance = _rgb_color_hint(frame, bbox)
            rgb_evidence = (
                f"The measured bbox median RGB is "
                f"{np.round(rgb_median, 1).tolist()}"
                + (
                    f", which is close to the calibrated {rgb_hint} surface color."
                    if rgb_hint is not None
                    else "."
                )
            )
            verification_prompt = (
                "Image 1 is a nearest-neighbor enlarged close-up of original depth "
                f"region {region_id} on black. It contains exactly one scene entity. "
                "Ignore black padding. Judge its actual surface color and whether it "
                "is a block or open-top bin. The scene vocabulary is red, blue, "
                f"yellow, or purple block and white or pink bin. {rgb_evidence}"
            )
            verified_data = self._run(
                "doable",
                verification_prompt,
                zoomed,
                schema=verification_schema,
            )
            vlm_color = str(verified_data.get("color", ""))
            vlm_kind = str(verified_data.get("kind", ""))
            color = vlm_color
            kind = vlm_kind
            color_source = "vlm"
            if rgb_hint is not None:
                color = rgb_hint
                kind = "bin" if rgb_hint in {"white", "pink"} else "block"
                color_source = (
                    "vlm_rgb_agree"
                    if (vlm_color, vlm_kind) == (color, kind)
                    else "rgb_median_correction"
                )
            label = _REGION_COLOR_LABELS.get((color, kind))
            if label is None:
                raise ModelResponseError(
                    f"incompatible verified region color/kind: {color}/{kind}"
                )
            verified.append(
                RegionClassification(
                    region_id=region_id,
                    color=color,
                    kind=kind,
                    image_location=_bbox_image_location(bbox),
                    label=label,
                    vlm_color=vlm_color,
                    vlm_kind=vlm_kind,
                    color_source=color_source,
                    rgb_median=rgb_median,
                )
            )
            region_checks.append(
                {
                    "region_id": region_id,
                    "vlm_color": vlm_color,
                    "vlm_kind": vlm_kind,
                    "rgb_hint": rgb_hint,
                    "rgb_median": list(rgb_median),
                    "rgb_distance": rgb_distance,
                    "final_color": color,
                    "final_kind": kind,
                    "color_source": color_source,
                }
            )
        result = tuple(verified)
        if self.trace is not None:
            self.trace(
                "region_inventory",
                prompt,
                {
                    "type": "region_inventory",
                    "global_regions": global_rows,
                    "region_checks": region_checks,
                    "regions": [asdict(item) for item in result],
                },
            )
        return result

    @staticmethod
    def _scene_vocabulary() -> str:
        objects = ", ".join(PANDA_OBJECT_LABELS)
        bins = ", ".join(PANDA_DESTINATION_LABELS)
        return (
            "The RGB-D view comes from a Panda wrist camera beside the robot "
            "gripper. The tabletop contains exactly four movable blocks and two "
            "open-top bins. "
            f"Canonical graspable object labels: {objects}. "
            f"Canonical destination bin labels: {bins}. "
            "Aliases: 红色方块=red_cube, 蓝色方块=blue_cube, "
            "黄色方块=yellow_block, 紫色方块=purple_block, "
            "白色盒子=white_bin, 粉色或紫色盒子=pink_bin. "
            "Words 方块/cube/block include all four block objects."
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
            "Visually detect exactly one target object and the specified destination "
            "bin from the raw RGB image. The RGB image shows the real scene without "
            "any overlay boxes or text labels — you must locate objects by their "
            "visible color, shape, and context. Image 2 (grayscale depth) provides "
            "relative distance: brighter pixels are closer to the camera.\n"
            "Return canonical labels and your own estimated normalized "
            "bbox_xyxy_norm bounding boxes [x1, y1, x2, y2] for both target and "
            "destination. A destination bounding box is mandatory. "
            "Visual cues for the four graspable objects: red_cube (red, square-ish), "
            "blue_cube (blue, square-ish), yellow_block (yellow, rectangular), "
            "purple_block (purple, rectangular). Destinations: white_bin (gray/white "
            "open-top box), pink_bin (pink/purple open-top box). "
            "Estimate tight bounding boxes that enclose each visible instance. "
            "Do not include the robot arm or gripper inside any object box."
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
        near_frame = self._near_field_grasp_frame(frame)
        return self._check(
            "grasp_checker",
            "This is a post-pick view from a wrist camera that moved together with "
            "the Panda gripper. Verify the target is held between the Panda fingers. "
            "The RGB-D input has been masked to retain only valid points at or closer "
            "than 0.30 m from the wrist camera; black pixels are intentionally removed "
            "far-field evidence. "
            "The destination bins are irrelevant at this phase. A successful grasp "
            "appears large and close to the camera, centered between the two dark "
            "fingers. The wooden table remains visible behind or below it and may "
            "overlap the object in 2D; that expected overlap does not mean the object "
            "rests on the table. Use depth brightness and finger enclosure; return "
            "false only when the target is outside the fingers or clearly left behind "
            "on the distant table.",
            command,
            task,
            grounded,
            near_frame,
        )

    def check_place(
        self,
        command: str,
        task: PlannedTask,
        grounded: GroundedAction,
        frame: CameraFrame,
    ) -> CheckerEval:
        destination_bbox = grounded.destination_bbox
        if destination_bbox is None:
            raise ModelResponseError("place checker requires a destination bbox")
        destination_frame = self._numbered_region_frame(
            frame, (destination_bbox,)
        )
        bbox_text = (
            f"[{destination_bbox.x1:.4f}, {destination_bbox.y1:.4f}, "
            f"{destination_bbox.x2:.4f}, {destination_bbox.y2:.4f}]"
        )
        return self._check(
            "place_checker",
            "Image 1 includes an R0 destination overlay around the pre-action "
            f"{grounded.destination_label} depth region; its normalized destination "
            f"bbox is {bbox_text}, and image x increases from left to right. "
            f"Verify visible {grounded.target_label} color pixels are inside the "
            "specified open-top bin interior and the target is no longer held "
            "by the gripper; other objects already inside are allowed; overlap or "
            "partial occlusion does not invalidate placement. Ignore the target's "
            "pre-action position and return status=false when uncertain.",
            command,
            task,
            grounded,
            destination_frame,
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
