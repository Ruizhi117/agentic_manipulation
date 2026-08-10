#!/usr/bin/env python
"""Process one EE command without ManiSkill and emit protocol-compatible RGB-D."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_manipulation.demo.ee_mock import run_ee_mock_once  # noqa: E402
from agentic_manipulation.demo.protocol import resolve_project_path  # noqa: E402
from agentic_manipulation.errors import AgenticManipulationError  # noqa: E402


run_once = run_ee_mock_once


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot EE camera mock")
    parser.add_argument("--ipc-dir", default="temp/ee")
    args = parser.parse_args(argv)
    try:
        directory = resolve_project_path(PROJECT_ROOT, args.ipc_dir)
        response = run_once(PROJECT_ROOT, directory)
    except AgenticManipulationError as exc:
        print(f"[MOCK] EE error: {exc}")
        return 2
    print(
        f"[MOCK] EE command #{response['command_id']} -> "
        f"{directory / 'response.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
