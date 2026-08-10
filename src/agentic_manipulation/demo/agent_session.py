"""Validated file IPC for a long-running visible Panda agent session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

from agentic_manipulation.demo.protocol import (
    atomic_write_json,
    read_json,
    resolve_project_path,
)
from agentic_manipulation.errors import ConfigurationError


@dataclass(frozen=True)
class AgentRequest:
    request_id: int
    command: str


@dataclass(frozen=True)
class AgentSessionPaths:
    directory: Path
    request: Path
    response: Path
    ready: Path

    @classmethod
    def from_value(
        cls, project_root: str | Path, value: str | Path
    ) -> "AgentSessionPaths":
        directory = resolve_project_path(Path(project_root), value)
        return cls(
            directory=directory,
            request=directory / "request.json",
            response=directory / "response.json",
            ready=directory / "ready.json",
        )


def parse_agent_request(payload: object) -> AgentRequest:
    if not isinstance(payload, dict):
        raise ConfigurationError("agent request must be a JSON object")
    request_id = payload.get("request_id")
    if isinstance(request_id, bool) or not isinstance(request_id, int):
        raise ConfigurationError("request_id must be a positive integer")
    if request_id <= 0:
        raise ConfigurationError("request_id must be positive")
    command = payload.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ConfigurationError("command must not be blank")
    return AgentRequest(request_id=request_id, command=command.strip())


def mark_session_ready(paths: AgentSessionPaths, *, pid: int) -> None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ConfigurationError("pid must be a positive integer")
    atomic_write_json(paths.ready, {"status": "ready", "pid": pid})


def mark_session_stopped(paths: AgentSessionPaths, *, pid: int) -> None:
    atomic_write_json(paths.ready, {"status": "stopped", "pid": int(pid)})


def read_agent_request(paths: AgentSessionPaths) -> AgentRequest | None:
    if not paths.request.is_file():
        return None
    return parse_agent_request(read_json(paths.request))


def publish_agent_request(
    paths: AgentSessionPaths, command: str, *, request_id: int
) -> AgentRequest:
    value = parse_agent_request({"request_id": request_id, "command": command})
    atomic_write_json(
        paths.request,
        {"request_id": value.request_id, "command": value.command},
    )
    return value


def publish_agent_response(
    paths: AgentSessionPaths,
    request_id: int,
    payload: dict[str, object],
) -> None:
    if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id <= 0:
        raise ConfigurationError("request_id must be positive")
    response = dict(payload)
    response["request_id"] = request_id
    atomic_write_json(paths.response, response)


def wait_for_agent_response(
    paths: AgentSessionPaths,
    request_id: int,
    *,
    timeout: float,
    poll_interval: float = 0.1,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object] | None:
    if timeout < 0 or poll_interval <= 0:
        raise ConfigurationError("timeout must be non-negative and poll_interval positive")
    deadline = time.monotonic() + timeout
    last_progress: dict[str, object] | None = None
    seen_event_ids: set[int] = set()

    def forward_progress(response: dict[str, object]) -> None:
        nonlocal last_progress
        if progress_callback is None:
            return
        progress_events = response.get("progress_events")
        if isinstance(progress_events, list):
            for raw_event in progress_events:
                if not isinstance(raw_event, dict):
                    continue
                event_id = raw_event.get("event_id")
                if (
                    isinstance(event_id, int)
                    and not isinstance(event_id, bool)
                    and event_id > 0
                    and event_id not in seen_event_ids
                ):
                    seen_event_ids.add(event_id)
                    progress_callback(dict(raw_event))
            return
        if response.get("status") == "running" and response != last_progress:
            progress_callback(response)
            last_progress = dict(response)

    while True:
        if paths.response.is_file():
            try:
                response = read_json(paths.response)
            except ConfigurationError:
                response = None
            if response is not None and response.get("request_id") == request_id:
                forward_progress(response)
                if response.get("status") != "running":
                    return response
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_interval)


def next_agent_request_id(paths: AgentSessionPaths) -> int:
    """Return one more than every request/response ID currently on disk."""

    ids = [0]
    for path in (paths.request, paths.response):
        if not path.is_file():
            continue
        try:
            value = read_json(path).get("request_id")
        except ConfigurationError:
            continue
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            ids.append(value)
    return max(ids) + 1
