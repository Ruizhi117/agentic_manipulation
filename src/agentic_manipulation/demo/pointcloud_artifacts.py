"""Inspectable camera-frame point-cloud artifacts for staged demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from agentic_manipulation.demo.protocol import atomic_write_json
from agentic_manipulation.errors import PerceptionError
from agentic_manipulation.perception.pointcloud import (
    backproject_camera,
    point_colors,
)
from agentic_manipulation.types import BBox, CameraFrame


@dataclass(frozen=True)
class PointCloudArtifacts:
    target_npy: Path
    target_ply: Path
    workspace_npy: Path
    overlay_png: Path
    metadata_json: Path
    target_point_count: int
    workspace_point_count: int


def _write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise PerceptionError("PLY points and colors must both have shape (N, 3)")
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    rows = "".join(
        f"{float(x):.8g} {float(y):.8g} {float(z):.8g} "
        f"{int(r)} {int(g)} {int(b)}\n"
        for (x, y, z), (r, g, b) in zip(points, colors, strict=True)
    )
    path.write_text(header + rows, encoding="ascii")


def _write_overlay(
    path: Path,
    frame: CameraFrame,
    bbox: BBox,
    segmentation_id: int,
) -> None:
    overlay = frame.rgb.copy()
    if frame.segmentation is None:
        raise PerceptionError("segmentation is required for point-cloud overlay")
    mask = frame.segmentation == segmentation_id
    overlay[mask] = (
        0.35 * overlay[mask].astype(np.float32)
        + 0.65 * np.array([40, 255, 80], dtype=np.float32)
    ).astype(np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox.as_pixels(frame.rgb.shape[1], frame.rgb.shape[0])
    draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), outline=(255, 220, 0), width=2)
    image.save(path)


def write_point_cloud_bundle(
    run_dir: Path,
    frame: CameraFrame,
    target_bbox: BBox,
    segmentation_id: int,
) -> PointCloudArtifacts:
    """Write lossless target/workspace clouds plus PLY and RGB overlay."""

    target = backproject_camera(frame, target_bbox, segmentation_id)
    colors = point_colors(frame, target_bbox, segmentation_id)
    if len(target) == 0:
        raise PerceptionError("target point cloud is empty")
    workspace = backproject_camera(frame)
    if len(workspace) == 0:
        raise PerceptionError("workspace point cloud is empty")

    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target_npy = directory / "target_points_camera.npy"
    target_ply = directory / "target_points_camera.ply"
    workspace_npy = directory / "workspace_points_camera.npy"
    overlay_png = directory / "segmentation_overlay.png"
    metadata_json = directory / "pointcloud.json"
    np.save(target_npy, target.astype(np.float32, copy=False))
    np.save(workspace_npy, workspace.astype(np.float32, copy=False))
    _write_ascii_ply(target_ply, target, colors)
    _write_overlay(overlay_png, frame, target_bbox, segmentation_id)
    atomic_write_json(
        metadata_json,
        {
            "target_points_path": target_npy.name,
            "target_ply_path": target_ply.name,
            "workspace_points_path": workspace_npy.name,
            "overlay_path": overlay_png.name,
            "target_point_count": int(len(target)),
            "workspace_point_count": int(len(workspace)),
            "segmentation_id": int(segmentation_id),
        },
    )
    return PointCloudArtifacts(
        target_npy=target_npy,
        target_ply=target_ply,
        workspace_npy=workspace_npy,
        overlay_png=overlay_png,
        metadata_json=metadata_json,
        target_point_count=int(len(target)),
        workspace_point_count=int(len(workspace)),
    )
