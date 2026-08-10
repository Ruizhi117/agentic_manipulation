"""Plain terminal conversation loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from agentic_manipulation.types import RuntimeResult


class CommandRuntime(Protocol):
    def run(self, command: str) -> RuntimeResult: ...


EXIT_COMMANDS = {"退出", "exit", "quit"}


def _write_result(result: RuntimeResult, output_fn: Callable[[str], None]) -> None:
    output_fn("[MOCK]" if result.is_mock else "[REAL]")
    for task in result.task_results:
        status = "成功" if task.success else "失败"
        output_fn(f"步骤 {task.step}：{status}（尝试 {task.attempts} 次）")
        for failure in task.failures:
            output_fn(f"  原因：{failure}")
    output_fn(result.message)


def run_text_loop(
    runtime: CommandRuntime,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    while True:
        try:
            command = input_fn("用户> ")
        except (EOFError, StopIteration):
            return
        command = command.strip()
        if not command:
            continue
        if command.lower() in EXIT_COMMANDS:
            return
        _write_result(runtime.run(command), output_fn)

