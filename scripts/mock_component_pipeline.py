#!/usr/bin/env python
"""Generate and run all three component mock requests through ``temp``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_manipulation.demo.mock_pipeline import run_mock_pipeline  # noqa: E402
from agentic_manipulation.demo.protocol import resolve_project_path  # noqa: E402
from agentic_manipulation.errors import AgenticManipulationError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Three-component mock pipeline")
    parser.add_argument("--temp-root", default="temp")
    args = parser.parse_args(argv)
    try:
        temp_root = resolve_project_path(PROJECT_ROOT, args.temp_root)
        summary = run_mock_pipeline(PROJECT_ROOT, temp_root)
    except AgenticManipulationError as exc:
        print(f"[MOCK] Pipeline error: {exc}")
        return 2
    print(f"[MOCK] VLM phases: {', '.join(summary['vlm'])}")
    print(f"[MOCK] Grasp response: {temp_root / 'grasp' / 'response.json'}")
    print(f"[MOCK] EE response: {temp_root / 'ee' / 'response.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
