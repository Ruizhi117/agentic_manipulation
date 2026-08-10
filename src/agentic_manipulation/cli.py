"""Safe command-line construction for explicit mock and real modes."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
import json
from pathlib import Path
from typing import Any
from urllib import error, request

from agentic_manipulation.config import RuntimeConfig
from agentic_manipulation.conversation.text import run_text_loop
from agentic_manipulation.conversation.voice import transcribe_audio
from agentic_manipulation.errors import ConfigurationError
from agentic_manipulation.runtime.agent import AgentRuntime
from agentic_manipulation.runtime.mock import build_mock_runtime
from agentic_manipulation.types import RuntimeEvent, RuntimeResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL + GraspNet ManiSkill closed-loop manipulation demo"
    )
    parser.add_argument("--mode", choices=("mock", "real"), default=None)
    parser.add_argument("--input", choices=("text", "audio"), default="text")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--command", help="run one command without reading stdin")
    parser.add_argument("--voice-model", default="small")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera", default="scene_camera")
    parser.add_argument("--render-backend", choices=("cpu", "gpu"), default="cpu")
    return parser


def _validate_ollama(config: RuntimeConfig) -> None:
    endpoint = f"{config.ollama_url}/api/tags"
    try:
        with request.urlopen(endpoint, timeout=3.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        raise ConfigurationError(
            f"real mode requires reachable Ollama at {config.ollama_url}: {exc}"
        ) from exc
    models = {
        str(item.get("name", ""))
        for item in payload.get("models", [])
        if isinstance(item, dict)
    }
    if config.qwen_model not in models:
        raise ConfigurationError(
            f"Ollama model {config.qwen_model!r} is not installed; "
            f"available models: {sorted(models)}"
        )


def _validate_real_requirements(config: RuntimeConfig) -> Path:
    checkpoint = config.graspnet_checkpoint
    if checkpoint is None:
        raise ConfigurationError(
            "real mode requires GraspNet checkpoint via "
            "AGENTIC_GRASPNET_CHECKPOINT"
        )
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ConfigurationError(
            f"GraspNet checkpoint does not exist: {checkpoint}"
        )
    try:
        from mani_skill import ASSET_DIR
    except ImportError as exc:
        raise ConfigurationError(
            "real mode requires ManiSkill 3.0.1 and SAPIEN 3.0.3"
        ) from exc
    asset_path = Path(ASSET_DIR) / "robots" / "xlerobot" / "xlerobot.urdf"
    if not asset_path.is_file():
        raise ConfigurationError(
            f"xlerobot asset is missing: {asset_path}; "
            "run scripts/download_maniskill_assets.ps1"
        )
    _validate_ollama(config)
    return checkpoint


def build_runtime(
    config: RuntimeConfig,
    *,
    seed: int = 7,
    camera_uid: str = "scene_camera",
    render_backend: str = "cpu",
    event_callback: Callable[[RuntimeEvent], None] | None = None,
) -> AgentRuntime:
    if config.mode == "mock":
        return build_mock_runtime(config, event_callback=event_callback)

    checkpoint = _validate_real_requirements(config)
    try:
        import gymnasium as gym

        import agentic_manipulation.envs  # noqa: F401 - registers the environment
        from agentic_manipulation.control.executor import (
            ManiSkillXlerobotBackend,
            MotionExecutor,
        )
        from agentic_manipulation.envs.runtime_scene import ManiSkillRuntimeScene
        from agentic_manipulation.models.graspnet import GraspNetProvider
        from agentic_manipulation.models.qwen_vl import OllamaQwenVLClient
        from agentic_manipulation.runtime.artifacts import ArtifactWriter
        from agentic_manipulation.runtime.checker import CompositeChecker
    except ImportError as exc:
        raise ConfigurationError(f"real runtime dependency is missing: {exc}") from exc

    env: Any | None = None
    try:
        env = gym.make(
            "AgenticPickPlace-v1",
            robot_uids="xlerobot",
            obs_mode="rgb+depth+segmentation",
            control_mode=ManiSkillXlerobotBackend.control_mode,
            render_backend=render_backend,
        )
        env.reset(seed=seed)
        scene = ManiSkillRuntimeScene(env, camera_uid=camera_uid)
        backend = ManiSkillXlerobotBackend(env)
        return AgentRuntime(
            config=config,
            scene=scene,
            vlm=OllamaQwenVLClient(config),
            grasp_provider=GraspNetProvider(checkpoint),
            executor=MotionExecutor(backend),
            checker=CompositeChecker(),
            artifacts=ArtifactWriter(None),
            event_callback=event_callback,
        )
    except Exception as exc:
        if env is not None:
            env.close()
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"failed to construct real runtime: {exc}") from exc


def _event_line(event: RuntimeEvent) -> str:
    prefix = f"[步骤 {event.task_step}]" if event.task_step is not None else "[runtime]"
    return f"{prefix} {event.type.value}: {event.message}"


def _print_result(result: RuntimeResult) -> None:
    print("[MOCK]" if result.is_mock else "[REAL]")
    for task in result.task_results:
        status = "成功" if task.success else "失败"
        print(f"步骤 {task.step}：{status}（尝试 {task.attempts} 次）")
    print(result.message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = RuntimeConfig.from_env()
    if args.mode is not None:
        config = replace(config, mode=args.mode)

    try:
        command = args.command.strip() if args.command is not None else None
        if args.command is not None and not command:
            raise ConfigurationError("--command must not be blank")
        if args.input == "audio":
            if command is not None:
                raise ConfigurationError("--command cannot be combined with audio input")
            if args.audio is None:
                raise ConfigurationError("--input audio requires --audio PATH")
            command = transcribe_audio(args.audio, model_name=args.voice_model)

        runtime = build_runtime(
            config,
            seed=args.seed,
            camera_uid=args.camera,
            render_backend=args.render_backend,
            event_callback=lambda event: print(_event_line(event)),
        )
        try:
            if command is None:
                run_text_loop(runtime)
            else:
                print(f"识别指令：{command}")
                _print_result(runtime.run(command))
        finally:
            close = getattr(runtime.scene, "close", None)
            if callable(close):
                close()
    except ConfigurationError as exc:
        parser.exit(2, f"配置错误：{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
