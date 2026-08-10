#!/usr/bin/env python
"""Run one prepared-prompt VLM phase through ``temp`` JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_manipulation.demo.protocol import (  # noqa: E402
    atomic_write_json,
    read_json,
    resolve_project_path,
)
from agentic_manipulation.demo.vlm_component import run_vlm_request  # noqa: E402
from agentic_manipulation.errors import AgenticManipulationError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot Qwen3-VL component")
    parser.add_argument("--mode", choices=("real", "mock"), required=True)
    parser.add_argument("--request", default="temp/vlm/request.json")
    parser.add_argument("--response", default="temp/vlm/response.json")
    args = parser.parse_args(argv)

    response_path = resolve_project_path(PROJECT_ROOT, args.response)
    request_id: object = None
    try:
        request = read_json(resolve_project_path(PROJECT_ROOT, args.request))
        request_id = request.get("request_id")
        response = run_vlm_request(request, PROJECT_ROOT, args.mode)
    except AgenticManipulationError as exc:
        response = {
            "request_id": request_id,
            "status": "error",
            "message": str(exc),
            "is_mock": args.mode == "mock",
        }
        atomic_write_json(response_path, response)
        print(f"[{args.mode.upper()}] VLM error: {exc}")
        return 2
    atomic_write_json(response_path, response)
    print(f"[{args.mode.upper()}] VLM {response['phase']} -> {response_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
