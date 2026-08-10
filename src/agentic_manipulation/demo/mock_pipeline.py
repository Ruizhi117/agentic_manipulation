"""File-producing mock pipeline for the three independent components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from agentic_manipulation.demo.ee_mock import run_ee_mock_once
from agentic_manipulation.demo.grasp_component import run_grasp_request
from agentic_manipulation.demo.protocol import (
    atomic_write_json,
    resolve_project_path,
)
from agentic_manipulation.demo.vlm_component import run_vlm_request


_PHASE_PROMPTS = {
    "doable": "根据腕部相机图片，判断是否能抓取红色方块，只输出可行性 JSON。",
    "planning": "理解任务并拆分步骤：抓取红色方块，然后检查是否抓住。",
    "action": "当前步骤是抓取红色方块，请定位单个目标并输出 action JSON。",
    "checker": "检查红色方块是否位于夹爪两指之间，只输出 checker JSON。",
}


def run_mock_pipeline(
    project_root: Path, temp_root: Path
) -> dict[str, object]:
    """Create fixtures and pass them through VLM, grasp, and EE mock files."""
    root = Path(project_root).resolve()
    exchange = resolve_project_path(root, temp_root)
    input_dir = exchange / "input"
    vlm_dir = exchange / "vlm"
    grasp_dir = exchange / "grasp"
    ee_dir = exchange / "ee"
    for directory in (input_dir, vlm_dir, grasp_dir, ee_dir):
        directory.mkdir(parents=True, exist_ok=True)

    command_path = input_dir / "command.txt"
    command_path.write_text("请抓取桌面上的红色方块", encoding="utf-8")
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[:, :] = [105, 100, 90]
    rgb[85:155, 125:195] = [235, 35, 25]
    image_path = input_dir / "rgb.png"
    depth_path = input_dir / "depth.npy"
    Image.fromarray(rgb).save(image_path)
    np.save(depth_path, np.full((240, 320), 0.5, dtype=np.float32))

    vlm_results: dict[str, dict[str, object]] = {}
    for request_id, phase in enumerate(_PHASE_PROMPTS, start=1):
        prompt_path = input_dir / f"{phase}_prompt.txt"
        prompt_path.write_text(_PHASE_PROMPTS[phase], encoding="utf-8")
        request = {
            "request_id": request_id,
            "phase": phase,
            "prompt_path": str(prompt_path.relative_to(root)),
            "image_path": str(image_path.relative_to(root)),
        }
        response = run_vlm_request(request, root, "mock")
        atomic_write_json(vlm_dir / "request.json", request)
        atomic_write_json(vlm_dir / "response.json", response)
        atomic_write_json(vlm_dir / f"{phase}_request.json", request)
        atomic_write_json(vlm_dir / f"{phase}_response.json", response)
        vlm_results[phase] = response

    target_points = np.array(
        [[0.0, 0.0, 0.48], [0.02, 0.0, 0.49], [0.0, 0.02, 0.50]],
        dtype=np.float32,
    )
    workspace_points = np.vstack(
        [target_points, [[-0.1, -0.1, 0.55], [0.1, 0.1, 0.56]]]
    ).astype(np.float32)
    target_path = grasp_dir / "target_points.npy"
    workspace_path = grasp_dir / "workspace_points.npy"
    np.save(target_path, target_points)
    np.save(workspace_path, workspace_points)
    grasp_request = {
        "request_id": 1,
        "target_points_path": str(target_path.relative_to(root)),
        "workspace_points_path": str(workspace_path.relative_to(root)),
        "world_from_camera": np.eye(4).tolist(),
        "grasp_from_ee": np.eye(4).tolist(),
    }
    grasp_response = run_grasp_request(grasp_request, root, "mock")
    atomic_write_json(grasp_dir / "request.json", grasp_request)
    atomic_write_json(grasp_dir / "response.json", grasp_response)

    ee_command = {
        "command_id": 1,
        "target_pose": grasp_response["world_from_ee"],
        "gripper": 0.0,
        "steps": 20,
    }
    atomic_write_json(ee_dir / "command.json", ee_command)
    ee_response = run_ee_mock_once(root, ee_dir)
    return {
        "vlm": vlm_results,
        "grasp": grasp_response,
        "ee": ee_response,
        "is_mock": True,
    }
