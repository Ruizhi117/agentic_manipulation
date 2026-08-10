"""One-shot structured VLM component driven by prepared prompt and RGB files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from agentic_manipulation.config import RuntimeConfig
from agentic_manipulation.demo.protocol import resolve_project_path
from agentic_manipulation.errors import ConfigurationError, ModelResponseError
from agentic_manipulation.models.qwen_vl import OllamaQwenVLClient
from agentic_manipulation.types import CameraFrame


_PHASES = ("doable", "planning", "action", "checker")


class StructuredPhaseClient(Protocol):
    provider_name: str

    def run_phase(
        self, phase: str, prompt: str, frame: CameraFrame
    ) -> Mapping[str, object]: ...


class OllamaStructuredPhaseClient:
    """Expose Ollama's schema-constrained phase call with a supplied prompt."""

    provider_name = "ollama-qwen-vl"

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self._client = OllamaQwenVLClient(config or RuntimeConfig.from_env())

    def run_phase(
        self, phase: str, prompt: str, frame: CameraFrame
    ) -> Mapping[str, object]:
        return self._client._chat(phase, prompt, frame)


class DeterministicStructuredPhaseClient:
    """Protocol-compatible model double; output is always labelled mock."""

    provider_name = "deterministic-vlm-component"

    _RESULTS: dict[str, dict[str, object]] = {
        "doable": {
            "type": "doable",
            "thought": "deterministic RGB fixture is available",
            "status": True,
        },
        "planning": {
            "type": "planning",
            "tasks": [
                {
                    "step": 1,
                    "action": "抓取红色方块",
                    "checker": "检查夹爪是否抓住红色方块",
                }
            ],
        },
        "action": {
            "type": "action",
            "task_step": 1,
            "target": {
                "label": "red_cube",
                "bbox_center_xywh_norm": {
                    "cx": 0.5,
                    "cy": 0.5,
                    "width": 0.2,
                    "height": 0.2,
                },
            },
            "destination": {"label": "robot_gripper"},
        },
        "checker": {
            "type": "checker",
            "thought": "deterministic fixture reports the target held",
            "status": True,
        },
    }

    def run_phase(
        self, phase: str, prompt: str, frame: CameraFrame
    ) -> Mapping[str, object]:
        del prompt, frame
        return dict(self._RESULTS[phase])


def _positive_request_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError("request_id must be a positive integer")
    return value


def _load_frame(image_path: Path) -> CameraFrame:
    try:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"failed to read RGB image {image_path}: {exc}") from exc
    height, width = rgb.shape[:2]
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = float(width)
    intrinsic[1, 1] = float(height)
    intrinsic[0, 2] = (width - 1) / 2
    intrinsic[1, 2] = (height - 1) / 2
    return CameraFrame(
        rgb=rgb,
        depth_m=np.zeros((height, width), dtype=np.float32),
        intrinsic=intrinsic,
        world_from_camera=np.eye(4, dtype=np.float32),
        segmentation=None,
        timestamp=0.0,
    )


def run_vlm_request(
    request: Mapping[str, object],
    project_root: Path,
    mode: str,
    client: StructuredPhaseClient | None = None,
) -> dict[str, object]:
    """Run one schema-constrained VLM phase from filesystem inputs."""

    request_id = _positive_request_id(request.get("request_id"))
    phase = request.get("phase")
    if phase not in _PHASES:
        raise ConfigurationError(f"phase must be one of {_PHASES}")
    if mode not in ("real", "mock"):
        raise ConfigurationError("mode must be 'real' or 'mock'")

    prompt_value = request.get("prompt_path")
    image_value = request.get("image_path")
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        raise ConfigurationError("prompt_path must be a nonempty string")
    if not isinstance(image_value, str) or not image_value.strip():
        raise ConfigurationError("image_path must be a nonempty string")
    prompt_path = resolve_project_path(project_root, prompt_value)
    image_path = resolve_project_path(project_root, image_value)
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"failed to read prompt {prompt_path}: {exc}") from exc
    if not prompt:
        raise ConfigurationError("prepared prompt must not be blank")
    frame = _load_frame(image_path)

    active_client: StructuredPhaseClient
    if client is not None:
        active_client = client
    elif mode == "mock":
        active_client = DeterministicStructuredPhaseClient()
    else:
        active_client = OllamaStructuredPhaseClient()
    result = active_client.run_phase(str(phase), prompt, frame)
    if not isinstance(result, Mapping) or result.get("type") != phase:
        raise ModelResponseError(
            f"VLM result type must be {phase!r}, got "
            f"{result.get('type') if isinstance(result, Mapping) else type(result).__name__!r}"
        )
    return {
        "request_id": request_id,
        "status": "ok",
        "phase": phase,
        "result": dict(result),
        "provider": active_client.provider_name,
        "is_mock": mode == "mock",
    }
