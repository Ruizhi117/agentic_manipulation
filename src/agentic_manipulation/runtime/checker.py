"""Combine visual-language and simulator-state task checks."""

from __future__ import annotations

from typing import Protocol

from agentic_manipulation.runtime.semantics import ResolvedAction
from agentic_manipulation.types import CheckerEval, TaskResult


class SceneState(Protocol):
    def is_in_bin(self, instance_id: str, bin_id: str) -> bool: ...

    def is_released(self, instance_id: str) -> bool: ...

    def is_stable(self, instance_id: str) -> bool: ...


class CompositeChecker:
    def check(
        self,
        visual: CheckerEval,
        scene_state: SceneState,
        resolved: ResolvedAction,
    ) -> TaskResult:
        failures: list[str] = []
        if not visual.status:
            failures.append(f"visual checker returned false: {visual.thought}")
        if not scene_state.is_in_bin(
            resolved.target_instance_id, resolved.destination_instance_id
        ):
            failures.append("target is not inside destination bin")
        if not scene_state.is_released(resolved.target_instance_id):
            failures.append("target is still held by the gripper")
        if not scene_state.is_stable(resolved.target_instance_id):
            failures.append("target is not stable")
        return TaskResult(
            step=resolved.task_step,
            success=not failures,
            failures=tuple(failures),
        )
