"""Deterministic EE camera protocol implementation without ManiSkill."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from agentic_manipulation.demo.ee_protocol import camera_response, parse_ee_command
from agentic_manipulation.demo.protocol import (
    atomic_write_json,
    read_json,
    resolve_project_path,
)


def run_ee_mock_once(project_root: Path, ipc_dir: Path) -> dict[str, object]:
    """Consume exactly one command and write deterministic vision artifacts."""
    root = Path(project_root).resolve()
    directory = resolve_project_path(root, ipc_dir)
    command_path = directory / "command.json"
    response_path = directory / "response.json"
    try:
        command = parse_ee_command(read_json(command_path))
    finally:
        try:
            command_path.unlink()
        except OSError:
            pass

    directory.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[:, :, :] = [90, 105, 120]
    rgb[80:160, 120:200] = [220, 40, 30]
    depth = np.full((240, 320), 0.5, dtype=np.float32)
    rgb_path = directory / f"rgb_{command.command_id:06d}.png"
    depth_path = directory / f"depth_{command.command_id:06d}.npy"
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    intrinsic = np.array(
        [[240.0, 0.0, 159.5], [0.0, 240.0, 119.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    response = camera_response(
        command.command_id,
        command.target_pose,
        str(rgb_path.relative_to(root)),
        str(depth_path.relative_to(root)),
        intrinsic,
        np.eye(4, dtype=np.float64),
        is_mock=True,
    )
    atomic_write_json(response_path, response)
    return response
