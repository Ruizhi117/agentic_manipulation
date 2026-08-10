"""Four-stage Qwen3-VL adapter for Ollama's local chat API."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from io import BytesIO
import json
from typing import Any, Protocol
from urllib import error, request

from PIL import Image

from agentic_manipulation.config import RuntimeConfig
from agentic_manipulation.errors import ModelResponseError, OllamaUnavailableError
from agentic_manipulation.perception.depth import depth_grayscale_rgb
from agentic_manipulation.types import (
    BBox,
    CameraFrame,
    CheckerEval,
    DoableEval,
    GroundedAction,
    PlannedTask,
    PlanningEval,
)


Transport = Callable[[str, dict[str, Any], float], Mapping[str, Any]]


class VisionLanguageModel(Protocol):
    provider_name: str

    def evaluate_doable(self, command: str, frame: CameraFrame) -> DoableEval: ...

    def plan(self, command: str, frame: CameraFrame) -> PlanningEval: ...

    def ground(
        self, command: str, task: PlannedTask, frame: CameraFrame
    ) -> GroundedAction: ...

    def check(
        self,
        command: str,
        task: PlannedTask,
        grounded: GroundedAction,
        frame: CameraFrame,
    ) -> CheckerEval: ...


SYSTEM_PROMPT = """You are the visual planner for a robot manipulation runtime.
Return one JSON object only. The requested phase is one of doable, planning,
action, or checker. You may identify objects, order tasks, and ground target and
destination boxes. Never output robot joint commands or grasp poses. The
runtime and GraspNet handle geometry and motion. Use the exact key `type` for
the requested phase; never substitute `phase`. Keep thought fields short and
limited to visible evidence. Follow the supplied JSON schema exactly."""


_CENTER_BBOX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cx": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "cy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "width": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
        "height": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
    },
    "required": ["cx", "cy", "width", "height"],
    "additionalProperties": False,
}

_XYXY_BBOX_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 4,
    "maxItems": 4,
    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
}

PHASE_SCHEMAS: dict[str, dict[str, Any]] = {
    "doable": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "const": "doable"},
            "thought": {"type": "string"},
            "status": {"type": "boolean"},
        },
        "required": ["type", "thought", "status"],
        "additionalProperties": False,
    },
    "planning": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "const": "planning"},
            "tasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer", "minimum": 1},
                        "action": {
                            "type": "string",
                            "description": "Chinese action text naming target and destination",
                        },
                        "checker": {
                            "type": "string",
                            "description": "Chinese visual check for the completed action",
                        },
                    },
                    "required": ["step", "action", "checker"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["type", "tasks"],
        "additionalProperties": False,
    },
    "action": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "const": "action"},
            "task_step": {"type": "integer", "minimum": 1},
            "target": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bbox_xyxy_norm": _XYXY_BBOX_SCHEMA,
                },
                "required": ["label", "bbox_xyxy_norm"],
                "additionalProperties": False,
            },
            "destination": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bbox_xyxy_norm": _XYXY_BBOX_SCHEMA,
                },
                "required": ["label", "bbox_xyxy_norm"],
                "additionalProperties": False,
            },
        },
        "required": ["type", "task_step", "target", "destination"],
        "additionalProperties": False,
    },
    "checker": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "const": "checker"},
            "thought": {"type": "string"},
            "status": {"type": "boolean"},
        },
        "required": ["type", "thought", "status"],
        "additionalProperties": False,
    },
}


def _encode_png(frame: CameraFrame) -> str:
    buffer = BytesIO()
    Image.fromarray(frame.rgb, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_depth_png(frame: CameraFrame) -> str:
    buffer = BytesIO()
    Image.fromarray(depth_grayscale_rgb(frame.depth_m), mode="RGB").save(
        buffer, format="PNG"
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        raise ModelResponseError("model response must contain valid JSON")
    return "\n".join(lines[1:-1]).strip()


def _parse_json(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(_strip_json_fence(content))
    except (json.JSONDecodeError, ModelResponseError) as exc:
        raise ModelResponseError("model response must contain valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ModelResponseError("model response JSON must be an object")
    return parsed


def _expect_type(data: Mapping[str, Any], expected: str) -> None:
    if data.get("type") != expected:
        raise ModelResponseError(
            f"expected {expected} response, got {data.get('type')!r}"
        )


def _bbox(value: Any, field: str) -> BBox:
    if not isinstance(value, list) or len(value) != 4:
        raise ModelResponseError(f"{field} must be a four-value list")
    try:
        return BBox(*(float(item) for item in value))
    except (TypeError, ValueError) as exc:
        raise ModelResponseError(f"{field}: {exc}") from exc


def _object_bbox(value: Mapping[str, Any], field: str) -> BBox:
    center = value.get("bbox_center_xywh_norm")
    if isinstance(center, Mapping):
        try:
            cx = float(center["cx"])
            cy = float(center["cy"])
            width = float(center["width"])
            height = float(center["height"])
            return BBox(
                max(0.0, cx - width / 2),
                max(0.0, cy - height / 2),
                min(1.0, cx + width / 2),
                min(1.0, cy + height / 2),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelResponseError(f"{field}: {exc}") from exc
    if "bbox_xyxy_norm" in value:
        return _bbox(value["bbox_xyxy_norm"], field)
    raise ModelResponseError(f"{field} is missing")


def object_bbox(value: Mapping[str, Any], field: str) -> BBox:
    """Public validated conversion for structured object bounding boxes."""

    return _object_bbox(value, field)


class OllamaQwenVLClient:
    provider_name = "ollama-qwen-vl"

    def __init__(
        self,
        config: RuntimeConfig,
        transport: Transport | None = None,
        *,
        timeout: float = 60.0,
    ) -> None:
        self.config = config
        self.timeout = timeout
        self._transport = transport or self._http_transport

    @staticmethod
    def _http_transport(
        url: str, payload: dict[str, Any], timeout: float
    ) -> Mapping[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
            raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc

    def _chat(
        self,
        phase: str,
        phase_prompt: str,
        frame: CameraFrame,
        *,
        schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_schema = PHASE_SCHEMAS[phase] if schema is None else dict(schema)
        payload = {
            "model": self.config.qwen_model,
            "keep_alive": -1,
            "format": active_schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 1024},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Image 1 is RGB. Image 2 is grayscale depth: nearer valid "
                        "pixels are brighter, farther valid pixels are darker, and "
                        "black means invalid depth. Use Image 2 for relative distance "
                        "judgments instead of guessing distance from RGB object size.\n"
                        f"Phase: {phase}\n{phase_prompt}\n"
                        "Return JSON matching this exact schema: "
                        f"{json.dumps(active_schema, ensure_ascii=False)}"
                    ),
                    "images": [_encode_png(frame), _encode_depth_png(frame)],
                },
            ],
        }
        try:
            result = self._transport(
                f"{self.config.ollama_url}/api/chat", payload, self.timeout
            )
        except OllamaUnavailableError:
            raise
        except (OSError, TimeoutError) as exc:
            raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc
        try:
            message = result["message"]
            content = message["content"]
        except (KeyError, TypeError) as exc:
            raise ModelResponseError(
                "Ollama response must contain message.content"
            ) from exc
        if not isinstance(content, str):
            raise ModelResponseError("Ollama message.content must be text")
        if not content.strip() and isinstance(message, Mapping):
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                content = thinking
        data = _parse_json(content)
        _expect_type(data, phase)
        return data

    def run_structured_phase(
        self,
        phase: str,
        phase_prompt: str,
        frame: CameraFrame,
        *,
        schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one schema-constrained phase with an exact caller-owned prompt."""

        if phase not in PHASE_SCHEMAS:
            raise ModelResponseError(f"unsupported structured phase: {phase}")
        return self._chat(phase, phase_prompt, frame, schema=schema)

    def unload(self) -> None:
        """Unload this model and its KV/context allocation from Ollama."""

        payload = {
            "model": self.config.qwen_model,
            "keep_alive": 0,
            "stream": False,
        }
        try:
            self._transport(
                f"{self.config.ollama_url}/api/generate",
                payload,
                self.timeout,
            )
        except OllamaUnavailableError:
            raise
        except (OSError, TimeoutError) as exc:
            raise OllamaUnavailableError(
                f"Ollama unload request failed: {exc}"
            ) from exc

    def evaluate_doable(self, command: str, frame: CameraFrame) -> DoableEval:
        data = self._chat(
            "doable",
            f"Command: {command}\nReport visible evidence and whether the complete command is feasible.",
            frame,
        )
        try:
            return DoableEval(
                type=str(data["type"]),
                thought=str(data["thought"]),
                status=data["status"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelResponseError(f"invalid doable response: {exc}") from exc

    def plan(self, command: str, frame: CameraFrame) -> PlanningEval:
        data = self._chat(
            "planning",
            (
                f"Command: {command}\n"
                "Create exactly one task per visible physical target; never group "
                "multiple objects into one task. For this compound command, return "
                "exactly 3 tasks in this order: 西红柿罐头1到灰色箱子, "
                "西红柿罐头2到灰色箱子, then 离紫色箱子最近的香蕉到紫色箱子. "
                "Write Chinese action/checker text and include the literal words "
                "西红柿, 灰色, 香蕉, 最近, and 紫色 where applicable."
            ),
            frame,
        )
        try:
            raw_tasks = data["tasks"]
            if not isinstance(raw_tasks, list):
                raise TypeError("tasks must be a list")
            tasks = tuple(
                PlannedTask(
                    step=int(item["step"]),
                    action=str(item["action"]),
                    checker=str(item["checker"]),
                )
                for item in raw_tasks
            )
            return PlanningEval(type=str(data["type"]), tasks=tasks)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelResponseError(f"invalid planning response: {exc}") from exc

    def ground(
        self, command: str, task: PlannedTask, frame: CameraFrame
    ) -> GroundedAction:
        visual_hint = {
            1: (
                "For tomato can 1, select the left-hand red cylinder on the tabletop, "
                "not either robot gripper."
            ),
            2: (
                "For tomato can 2, select the right-hand red cylinder that remains on "
                "the tabletop; do not select an object already in the gray bin or either "
                "of the robot grippers."
            ),
            3: (
                "Select the yellow curved banana closest to the purple bin; in this "
                "camera view it is the right-hand banana."
            ),
        }.get(task.step, "Ground the single object named by the current task.")
        data = self._chat(
            "action",
            (
                f"Command: {command}\nCurrent step {task.step}: {task.action}\n"
                f"Visual instance rule: {visual_hint}\n"
                "Return a target label and normalized center/width/height target box. "
                "cx/cy are the object center; width/height are positive full extents. "
                "For the destination return only its canonical label; the simulator owns "
                "the destination geometry. "
                "Use only these canonical labels: tomato_can, banana, gray_bin, "
                "purple_bin. Ground exactly one target instance."
            ),
            frame,
        )
        try:
            target = data["target"]
            destination = data["destination"]
            if not isinstance(target, Mapping) or not isinstance(
                destination, Mapping
            ):
                raise TypeError("target and destination must be objects")
            return GroundedAction(
                type=str(data["type"]),
                task_step=int(data["task_step"]),
                target_label=str(target["label"]),
                target_bbox=_object_bbox(target, "target bounding box"),
                destination_label=str(destination["label"]),
                destination_bbox=(
                    _object_bbox(destination, "destination bounding box")
                    if "bbox_center_xywh_norm" in destination
                    or "bbox_xyxy_norm" in destination
                    else None
                ),
            )
        except ModelResponseError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelResponseError(f"invalid action response: {exc}") from exc

    def check(
        self,
        command: str,
        task: PlannedTask,
        grounded: GroundedAction,
        frame: CameraFrame,
    ) -> CheckerEval:
        visual_hint = {
            ("tomato_can", "gray_bin"): (
                "Look for the red cylinder inside the gray container's inner floor "
                "and surrounding walls."
            ),
            ("banana", "purple_bin"): (
                "Look for the yellow curved banana inside the purple container's "
                "inner floor and surrounding walls."
            ),
        }.get(
            (grounded.target_label, grounded.destination_label),
            "Look for the named target within the destination's inner walls.",
        )
        data = self._chat(
            "checker",
            (
                f"Command: {command}\nCompleted step {task.step}: {task.action}\n"
                f"Check: {task.checker}\nTarget: {grounded.target_label}; "
                f"destination: {grounded.destination_label}.\n"
                "The destination is an open-top bin. Set status=true only when the "
                "target is visibly inside the open-top bin, enclosed by its walls; "
                "inside objects can appear below the wall tops. The target must be "
                "inside, not merely above, below, or next to the bin. "
                "In this simulation, tomato_can always means the red cylinder, never "
                "a yellow object. Yellow curved objects are bananas. Gray and purple "
                "objects with four walls are the destination bins. "
                f"Visual rule: {visual_hint}"
            ),
            frame,
        )
        try:
            return CheckerEval(
                type=str(data["type"]),
                thought=str(data["thought"]),
                status=data["status"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelResponseError(f"invalid checker response: {exc}") from exc


class DeterministicVisionLanguageModel:
    """Explicitly marked non-model implementation for tests and mock demos."""

    provider_name = "deterministic"

    def __init__(
        self,
        *,
        groundings: Mapping[int, GroundedAction],
        planning: PlanningEval | None = None,
        checker_statuses: Mapping[int, bool] | None = None,
    ) -> None:
        self._groundings = dict(groundings)
        self._planning = planning or PlanningEval(
            "planning",
            tuple(
                PlannedTask(
                    step=step,
                    action=f"move {grounding.target_label} to {grounding.destination_label}",
                    checker=f"check {grounding.target_label} in {grounding.destination_label}",
                )
                for step, grounding in sorted(self._groundings.items())
            ),
        )
        self._checker_statuses = dict(checker_statuses or {})
        self.calls: list[str] = []

    def evaluate_doable(self, command: str, frame: CameraFrame) -> DoableEval:
        self.calls.append("doable")
        return DoableEval("doable", "deterministic scene fixture is available", True)

    def plan(self, command: str, frame: CameraFrame) -> PlanningEval:
        self.calls.append("planning")
        return self._planning

    def ground(
        self, command: str, task: PlannedTask, frame: CameraFrame
    ) -> GroundedAction:
        self.calls.append(f"action:{task.step}")
        try:
            return self._groundings[task.step]
        except KeyError as exc:
            raise ModelResponseError(
                f"no deterministic grounding for step {task.step}"
            ) from exc

    def check(
        self,
        command: str,
        task: PlannedTask,
        grounded: GroundedAction,
        frame: CameraFrame,
    ) -> CheckerEval:
        self.calls.append(f"checker:{task.step}")
        status = self._checker_statuses.get(task.step, True)
        return CheckerEval("checker", "deterministic fixture check", status)
