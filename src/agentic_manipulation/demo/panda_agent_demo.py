"""Human-command entry point for the Panda closed-loop runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib import request

import numpy as np

from agentic_manipulation.config import RuntimeConfig
from agentic_manipulation.control.panda_atomic import (
    AtomicTransferPlan,
    AtomicTransferReport,
    PandaAtomicPickPlaceSkill,
    PandaDeltaPoseBackend,
)
from agentic_manipulation.demo.protocol import atomic_write_json, resolve_project_path
from agentic_manipulation.demo.stage_recorder import StageRecorder
from agentic_manipulation.envs.ee_camera_scene import (
    DESTINATION_INSTANCE_IDS,
    GRASPABLE_INSTANCE_IDS,
)
from agentic_manipulation.envs.panda_runtime_scene import PandaSortingScene
from agentic_manipulation.errors import ConfigurationError
from agentic_manipulation.models.graspnet import (
    DeterministicTopDownGraspProvider,
    GraspNetProvider,
)
from agentic_manipulation.models.panda_qwen_vl import PandaVisionLanguageModel
from agentic_manipulation.models.qwen_vl import OllamaQwenVLClient
from agentic_manipulation.runtime.artifacts import ArtifactWriter
from agentic_manipulation.runtime.panda_agent import PandaAgentRuntime
from agentic_manipulation.types import (
    BBox,
    CameraFrame,
    CheckerEval,
    DoableEval,
    GroundedAction,
    PlannedTask,
    PlanningEval,
    RuntimeEvent,
)


@dataclass(frozen=True)
class AgentDemoOptions:
    mode: str
    command: str
    seed: int
    max_retries: int
    output_root: str | Path
    render_backend: str
    record: bool
    checkpoint: str | Path
    device: str


def ollama_launch_instructions() -> str:
    return "\n".join(
        (
            "Terminal A (Ollama service):",
            '$env:OLLAMA_HOST="http://127.0.0.1:11434"',
            "$env:OLLAMA_CONTEXT_LENGTH=16384",
            "ollama serve",
            "",
            "Terminal B (load qwen3-vl:2b):",
            '$env:OLLAMA_HOST="http://127.0.0.1:11434"',
            "$env:OLLAMA_CONTEXT_LENGTH=16384",
            "ollama run qwen3-vl:2b",
        )
    )


def validate_options(options: AgentDemoOptions, project_root: str | Path) -> Path:
    if options.mode not in {"mock", "real"}:
        raise ConfigurationError("mode must be 'mock' or 'real'")
    if not isinstance(options.command, str) or not options.command.strip():
        raise ConfigurationError("command must not be blank")
    if options.max_retries < 0:
        raise ConfigurationError("max_retries must be non-negative")
    if options.seed < 0:
        raise ConfigurationError("seed must be non-negative")
    if options.render_backend not in {"cpu", "gpu"}:
        raise ConfigurationError("render_backend must be 'cpu' or 'gpu'")
    return resolve_project_path(Path(project_root), options.output_root)


def _ollama_models(url: str) -> set[str]:
    """Return the set of model names available from the Ollama /api/tags endpoint.

    Retries transient connection failures up to 3 times with short back-off,
    and explicitly bypasses system proxy settings so that ``127.0.0.1`` is
    always reached directly.
    """
    import time

    target = f"{url.rstrip('/')}/api/tags"
    # Bypass any system-wide proxy — Ollama listens on localhost.
    proxy_handler = request.ProxyHandler({})
    opener = request.build_opener(proxy_handler)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with opener.open(target, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, TimeoutError, ValueError) as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
            continue
        try:
            return {str(model["name"]) for model in payload["models"]}
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(
                "Ollama /api/tags returned an invalid response"
            ) from exc
    raise ConfigurationError(
        f"Ollama is unreachable at {target} after 3 attempts. "
        f"Last error: {last_exc}. "
        "Make sure Ollama is running:\n" + ollama_launch_instructions()
    ) from last_exc


def check_real_prerequisites(
    options: AgentDemoOptions,
    project_root: str | Path,
    *,
    check_ollama: bool = True,
) -> Path:
    root = Path(project_root).resolve()
    checkpoint = resolve_project_path(root, options.checkpoint)
    if not checkpoint.is_file():
        raise ConfigurationError(f"GraspNet checkpoint does not exist: {checkpoint}")
    baseline = root / "graspnet-baseline"
    for relative in ("models", "dataset", "utils"):
        candidate = str((baseline / relative).resolve())
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    missing = [
        name
        for name in ("graspnet", "graspnetAPI", "collision_detector")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ConfigurationError(
            f"real GraspNet modules are unavailable: {', '.join(missing)}"
        )
    if check_ollama:
        url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        if "qwen3-vl:2b" not in _ollama_models(url):
            raise ConfigurationError("Ollama model qwen3-vl:2b is not installed")
    return checkpoint


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class _MockScene:
    def __init__(self) -> None:
        self.held: str | None = None
        self.location: dict[str, str | None] = {"blue_cube": None}

    def capture(self) -> CameraFrame:
        segmentation = np.tile(np.arange(1, 9, dtype=np.int32), (16, 1))
        return CameraFrame(
            rgb=np.zeros((16, 8, 3), dtype=np.uint8),
            depth_m=np.full((16, 8), 0.4, dtype=np.float32),
            intrinsic=np.array(
                [[20.0, 0.0, 4.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            world_from_camera=np.eye(4, dtype=np.float32),
            segmentation=segmentation,
            timestamp=0.0,
        )

    def visible_instances(self):
        names = (
            "red_cube",
            "blue_cube",
            "yellow_block",
            "purple_block",
            "white_bin",
            "pink_bin",
        )
        return {name: name for name in names}

    def segmentation_ids(self):
        return {index: name for index, name in enumerate(self.visible_instances(), 1)}

    def is_grasping(self, instance_id):
        return self.held == instance_id

    def is_in_bin(self, instance_id, bin_id):
        return self.location.get(instance_id) == bin_id

    def is_released(self, instance_id):
        return self.held != instance_id

    def is_stable(self, instance_id):
        return self.location.get(instance_id) is not None


class _MockVLM:
    provider_name = "deterministic-panda-agent"

    task = PlannedTask(1, "move blue_cube to white_bin", "check blue_cube in white_bin")

    def evaluate_doable(self, command, frame):
        del command, frame
        return DoableEval("doable", "deterministic scene is available", True)

    def plan(self, command, frame):
        del command, frame
        return PlanningEval("planning", (self.task,))

    def ground(self, command, task, frame):
        del command, frame
        target = next(
            label for label in GRASPABLE_INSTANCE_IDS if label in task.action
        )
        destination = next(
            label for label in DESTINATION_INSTANCE_IDS if label in task.action
        )
        columns = {
            "red_cube": 0,
            "blue_cube": 1,
            "yellow_block": 2,
            "purple_block": 3,
            "white_bin": 6,
            "pink_bin": 7,
        }
        return GroundedAction(
            "action",
            task.step,
            target,
            BBox(columns[target] / 8, 0.0, (columns[target] + 1) / 8, 1.0),
            destination,
            BBox(
                columns[destination] / 8,
                0.0,
                (columns[destination] + 1) / 8,
                1.0,
            ),
        )

    def check_grasp(self, command, task, grounded, frame):
        del command, task, grounded, frame
        return CheckerEval("checker", "deterministic held", True)

    def check_place(self, command, task, grounded, frame):
        del command, task, grounded, frame
        return CheckerEval("checker", "deterministic inside", True)


class _MockSkill:
    def __init__(self, scene: _MockScene) -> None:
        self.scene = scene
        self.backend = self

    def can_pick(self, _pose: object) -> bool:
        return True

    def plan_transfer(self, world_from_ee, world_from_release):
        grasp = np.asarray(world_from_ee, dtype=np.float64)
        release = np.asarray(world_from_release, dtype=np.float64)
        return AtomicTransferPlan(
            grasp, grasp, grasp, release, release, grasp
        )

    def execute(self, instance_id, plan, confirm_grasp):
        del plan
        self.scene.held = instance_id
        if not confirm_grasp():
            self.scene.held = None
            return AtomicTransferReport(
                False,
                (
                    "close",
                    "lift",
                    "grasp_check",
                    "open_recover",
                    "home_return",
                    "settle",
                ),
                "not held",
            )
        self.scene.held = None
        return AtomicTransferReport(
            True,
            (
                "pregrasp",
                "approach",
                "close",
                "lift",
                "grasp_check",
                "preplace",
                "place",
                "open_release",
                "retreat",
                "home_return",
                "settle",
            ),
        )

    def recover_after_failed_pick(self):
        self.scene.held = None

    def return_home(self):
        return None


def _event_sink(events: list[dict[str, object]], is_mock: bool):
    def emit(event: RuntimeEvent) -> None:
        row = {
            "type": event.type.value,
            "message": event.message,
            "task_step": event.task_step,
            "is_mock": is_mock,
        }
        events.append(row)
        step = "" if event.task_step is None else f" step={event.task_step}"
        print(f"[{event.type.value}]{step} {event.message}")

    return emit


def _run_mock(
    options: AgentDemoOptions,
    artifacts: ArtifactWriter,
    events: list[dict[str, object]],
):
    scene = _MockScene()
    runtime = PandaAgentRuntime(
        config=RuntimeConfig(mode="mock", max_retries=options.max_retries),
        scene=scene,
        vlm=_MockVLM(),
        grasp_provider=DeterministicTopDownGraspProvider(),
        skill=_MockSkill(scene),
        artifacts=artifacts,
        event_callback=_event_sink(events, True),
    )
    return runtime.run(options.command), None, []


def _run_real(
    options: AgentDemoOptions,
    root: Path,
    run_dir: Path,
    artifacts: ArtifactWriter,
    events: list[dict[str, object]],
):
    checkpoint = check_real_prerequisites(options, root)
    try:
        import gymnasium as gym
        import agentic_manipulation.envs  # noqa: F401
    except ImportError as exc:
        raise ConfigurationError(f"ManiSkill dependencies are unavailable: {exc}") from exc
    env = gym.make(
        "EECameraScene-v1",
        robot_uids="panda_wristcam",
        control_mode="pd_ee_delta_pose",
        obs_mode="rgb+depth",
        render_backend=options.render_backend,
        num_envs=1,
        sensor_configs={
            "hand_camera": {"width": 320, "height": 240},
            "scene_camera": {"width": 320, "height": 240},
        },
    )
    try:
        env.reset(seed=options.seed, options={"reconfigure": True})
        ollama_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        client = OllamaQwenVLClient(
            RuntimeConfig(
                ollama_url=ollama_url,
                qwen_model="qwen3-vl:2b",
                mode="real",
                max_retries=options.max_retries,
            ),
            timeout=120,
        )
        return _run_real_in_env(
            options,
            root,
            run_dir,
            artifacts,
            events,
            env,
            client=client,
            grasp_provider=GraspNetProvider(checkpoint, device=options.device),
        )
    finally:
        env.close()


def _run_real_in_env(
    options: AgentDemoOptions,
    root: Path,
    run_dir: Path,
    artifacts: ArtifactWriter,
    events: list[dict[str, object]],
    env: Any,
    *,
    client: OllamaQwenVLClient,
    grasp_provider: GraspNetProvider,
    render_callback: Callable[[], object] | None = None,
    image_callback: Callable[[str, np.ndarray, Path | None], None] | None = None,
    grasp_3d_callback: Callable[
        [np.ndarray, np.ndarray, object, object, Path | None], None
    ]
    | None = None,
    event_callback: Callable[[RuntimeEvent], None] | None = None,
):
    del root
    traces: list[dict[str, object]] = []
    scene = PandaSortingScene(env)
    recorder = StageRecorder()
    backend = PandaDeltaPoseBackend(
        env, recorder, render_callback=render_callback
    )
    skill = PandaAtomicPickPlaceSkill(
        backend, motion_steps=60, gripper_steps=20, settle_steps=30
    )

    def trace(phase, prompt, result):
        row = {
            "index": len(traces),
            "phase": phase,
            "prompt": prompt,
            "result": dict(result),
            "is_mock": False,
        }
        traces.append(row)
        atomic_write_json(run_dir / f"vlm_{len(traces):02d}_{phase}.json", row)

    vlm = PandaVisionLanguageModel(client, trace=trace)
    persist_event = _event_sink(events, False)

    def emit_event(event: RuntimeEvent) -> None:
        persist_event(event)
        if event_callback is not None:
            event_callback(event)

    runtime = PandaAgentRuntime(
        config=RuntimeConfig(mode="real", max_retries=options.max_retries),
        scene=scene,
        vlm=vlm,
        grasp_provider=grasp_provider,
        skill=skill,
        artifacts=artifacts,
        event_callback=emit_event,
        image_callback=image_callback,
        grasp_3d_callback=grasp_3d_callback,
    )
    result = runtime.run(options.command)
    video_path = None
    if options.record and recorder.frames:
        video = run_dir / "agent.mp4"
        recorder.write_mp4(video)
        recorder.write_motion_json(run_dir / "motion.json")
        video_path = video
    return result, video_path, traces


def _persist_agent_run(
    options: AgentDemoOptions,
    root: Path,
    runner: Callable[
        [Path, ArtifactWriter, list[dict[str, object]]],
        tuple[object, Path | None, list[dict[str, object]]],
    ],
) -> dict[str, object]:
    output = validate_options(options, root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = ArtifactWriter(output, run_id=run_id)
    events: list[dict[str, object]] = []
    result, video, traces = runner(run_dir, artifacts, events)
    events_path = run_dir / "events.json"
    atomic_write_json(events_path, {"events": events})
    status_path = run_dir / "status.json"
    status: dict[str, object] = {
        "status": "ok" if result.success else "error",
        "mode": options.mode,
        "is_mock": options.mode == "mock",
        "command": options.command,
        "result": _jsonable(asdict(result)),
        "event_count": len(events),
        "trace_count": len(traces),
        "events_path": str(events_path.relative_to(root)).replace("\\", "/"),
        "video_path": (
            None if video is None else str(video.relative_to(root)).replace("\\", "/")
        ),
        "status_path": str(status_path.relative_to(root)).replace("\\", "/"),
    }
    atomic_write_json(status_path, status)
    return status


class RealPandaAgentSession:
    """Run repeated commands in one caller-owned ManiSkill environment."""

    def __init__(
        self,
        options: AgentDemoOptions,
        project_root: str | Path,
        env: Any,
        *,
        render_callback: Callable[[], object] | None = None,
        image_callback: Callable[[str, np.ndarray, Path | None], None] | None = None,
        grasp_3d_callback: Callable[
            [np.ndarray, np.ndarray, object, object, Path | None], None
        ]
        | None = None,
    ) -> None:
        if options.mode != "real":
            raise ConfigurationError("RealPandaAgentSession requires real mode")
        self.root = Path(project_root).resolve()
        validate_options(options, self.root)
        checkpoint = check_real_prerequisites(options, self.root)
        self.options = options
        self.env = env
        self.render_callback = render_callback
        self.image_callback = image_callback
        self.grasp_3d_callback = grasp_3d_callback
        PandaSortingScene(env)
        ollama_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        client_config = RuntimeConfig(
            ollama_url=ollama_url,
            qwen_model="qwen3-vl:2b",
            mode="real",
            max_retries=options.max_retries,
        )
        self.client_factory = lambda: OllamaQwenVLClient(
            client_config, timeout=120
        )
        self.grasp_provider = GraspNetProvider(checkpoint, device=options.device)

    def run(
        self,
        command: str,
        *,
        event_callback: Callable[[RuntimeEvent], None] | None = None,
    ) -> dict[str, object]:
        options = replace(self.options, command=command)
        client = self.client_factory()
        return _persist_agent_run(
            options,
            self.root,
            lambda run_dir, artifacts, events: _run_real_in_env(
                options,
                self.root,
                run_dir,
                artifacts,
                events,
                self.env,
                client=client,
                grasp_provider=self.grasp_provider,
                render_callback=self.render_callback,
                image_callback=self.image_callback,
                grasp_3d_callback=self.grasp_3d_callback,
                event_callback=event_callback,
            ),
        )


def run_agent_command(
    options: AgentDemoOptions, project_root: str | Path
) -> dict[str, object]:
    root = Path(project_root).resolve()
    validate_options(options, root)
    if options.mode == "real":
        check_real_prerequisites(options, root)
    if options.mode == "mock":
        runner = lambda run_dir, artifacts, events: _run_mock(  # noqa: E731
            options, artifacts, events
        )
    else:
        runner = lambda run_dir, artifacts, events: _run_real(  # noqa: E731
            options, root, run_dir, artifacts, events
        )
    return _persist_agent_run(options, root, runner)
