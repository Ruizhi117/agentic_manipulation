#!/usr/bin/env python
"""Run one inspectable Panda atomic-demo stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_manipulation.demo.panda_atomic_demo import STAGES, run_stage  # noqa: E402
from agentic_manipulation.envs.ee_camera_scene import GRASPABLE_INSTANCE_IDS  # noqa: E402
from agentic_manipulation.errors import AgenticManipulationError  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run scene, point-cloud, grasp, or Panda pick as a separate demo stage."
    )
    value.add_argument("--stage", required=True, choices=STAGES)
    value.add_argument("--mode", required=True, choices=("mock", "real"))
    value.add_argument("--target", default="blue_cube", choices=GRASPABLE_INSTANCE_IDS)
    value.add_argument("--destination", default="white_bin", choices=("white_bin", "pink_bin"))
    value.add_argument("--output", default="temp/feedback_v1")
    value.add_argument(
        "--checkpoint", default="graspnet-baseline/checkpoint-rs.tar"
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--render-backend", default="cpu")
    value.add_argument("--motion-steps", type=int, default=40)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        status = run_stage(
            stage=args.stage,
            mode=args.mode,
            target=args.target,
            output=args.output,
            project_root=PROJECT_ROOT,
            checkpoint_path=args.checkpoint,
            device=args.device,
            render_backend=args.render_backend,
            motion_steps=args.motion_steps,
            destination=args.destination,
        )
    except AgenticManipulationError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
