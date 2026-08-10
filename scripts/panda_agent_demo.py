#!/usr/bin/env python
"""Interactive or one-shot human-command Panda agent demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_manipulation.demo.agent_session import (  # noqa: E402
    AgentSessionPaths,
    next_agent_request_id,
    publish_agent_request,
    wait_for_agent_response,
)
from agentic_manipulation.demo.panda_agent_demo import (  # noqa: E402
    AgentDemoOptions,
    ollama_launch_instructions,
    run_agent_command,
)
from agentic_manipulation.demo.protocol import read_json  # noqa: E402
from agentic_manipulation.errors import AgenticManipulationError  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the Panda Qwen3-VL sorting agent.")
    value.add_argument("--mode", choices=("mock", "real"), default="mock")
    value.add_argument("--command")
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--max-retries", type=int, default=4)
    value.add_argument("--output-root", default="temp/panda_agent")
    value.add_argument("--render-backend", choices=("cpu", "gpu"), default="cpu")
    value.add_argument("--record", action="store_true")
    value.add_argument("--checkpoint", default="graspnet-baseline/checkpoint-rs.tar")
    value.add_argument("--device", default="cuda")
    value.add_argument(
        "--connect",
        action="store_true",
        help="Send commands to the long-running visible ee_camera_demo scene.",
    )
    value.add_argument(
        "--session-dir",
        default="temp/panda_agent_session",
        help="Project-relative shared Agent session directory.",
    )
    value.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait for one connected Agent command.",
    )
    value.epilog = "Ollama launch:\n" + ollama_launch_instructions()
    return value


def _options(args: argparse.Namespace, command: str) -> AgentDemoOptions:
    return AgentDemoOptions(
        mode=args.mode,
        command=command,
        seed=args.seed,
        max_retries=args.max_retries,
        output_root=args.output_root,
        render_backend=args.render_backend,
        record=args.record,
        checkpoint=args.checkpoint,
        device=args.device,
    )


def _run(args: argparse.Namespace, command: str) -> int:
    try:
        status = run_agent_command(_options(args, command), PROJECT_ROOT)
    except AgenticManipulationError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] == "ok" else 1


def _run_connected(args: argparse.Namespace, command: str) -> int:
    try:
        paths = AgentSessionPaths.from_value(PROJECT_ROOT, args.session_dir)
        if not paths.ready.is_file():
            raise AgenticManipulationError(f"可视仿真服务尚未启动：{paths.ready}")
        ready = read_json(paths.ready)
        if ready.get("status") != "ready":
            raise AgenticManipulationError(
                f"可视仿真服务当前状态不是 ready：{ready.get('status')}"
            )
        request_id = next_agent_request_id(paths)
        publish_agent_request(paths, command, request_id=request_id)
        print("[agent-client] 开始全新无历史对话，等待可视场景执行……")
        response = wait_for_agent_response(
            paths,
            request_id,
            timeout=args.timeout,
            progress_callback=_print_agent_progress,
        )
        if response is None:
            raise AgenticManipulationError(
                f"等待全新对话超时（{args.timeout:.1f} 秒）"
            )
    except AgenticManipulationError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1
    visible_response = dict(response)
    visible_response.pop("request_id", None)
    visible_response.pop("progress_events", None)
    print(json.dumps(visible_response, ensure_ascii=False, indent=2))
    return 0 if response.get("status") == "ok" else 1


def _print_agent_progress(response: dict[str, object]) -> None:
    phase = response.get("phase")
    message = response.get("message")
    if not isinstance(phase, str) or not isinstance(message, str) or not message.strip():
        return
    labels = {
        "observing": "观察",
        "doable": "场景",
        "planning": "规划",
        "task_started": "开始",
        "action": "识别",
        "grasping": "抓取位姿",
        "executing": "执行",
        "checking": "检查",
        "retrying": "重试",
        "task_completed": "完成",
        "succeeded": "完成",
        "failed": "失败",
    }
    label = labels.get(phase, phase)
    task_step = response.get("task_step")
    prefix = (
        f"[步骤 {task_step}][{label}]"
        if isinstance(task_step, int) and not isinstance(task_step, bool)
        else f"[{label}]"
    )
    print(f"{prefix} {message.strip()}")


def _dispatch(args: argparse.Namespace, command: str) -> int:
    return _run_connected(args, command) if args.connect else _run(args, command)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.timeout <= 0:
        print("--timeout 必须为正数。")
        return 2
    if args.connect and args.mode != "real":
        print("--connect 需要同时指定 --mode real。")
        return 2
    if args.command is not None:
        return _dispatch(args, args.command)
    print("输入操作命令；输入‘退出’、exit 或 quit 结束。任务完成后可继续提问。")
    while True:
        try:
            command = input("> ").strip()
        except EOFError:
            return 0
        if command.lower() in {"退出", "exit", "quit"}:
            return 0
        if not command:
            print("命令不能为空。")
            continue
        _dispatch(args, command)


if __name__ == "__main__":
    raise SystemExit(main())
