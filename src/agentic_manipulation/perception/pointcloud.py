"""Calibrated RGB-D geometry and semantic instance matching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from agentic_manipulation.errors import PerceptionError, SemanticValidationError
from agentic_manipulation.types import BBox, CameraFrame


def _selected_camera_data(
    frame: CameraFrame,
    bbox: BBox | None,
    segmentation_id: int | None,
    pixel_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select calibrated camera points and their RGB values in pixel order."""

    height, width = frame.depth_m.shape
    if bbox is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = bbox.as_pixels(width, height)
    if segmentation_id is not None and frame.segmentation is None:
        raise PerceptionError("segmentation is required when segmentation_id is set")
    if pixel_mask is not None:
        pixel_mask = np.asarray(pixel_mask, dtype=bool)
        if pixel_mask.shape != (height, width):
            raise PerceptionError(
                "pixel_mask shape must match the RGB-D frame height and width"
            )

    depth = frame.depth_m[y1:y2, x1:x2]
    vv, uu = np.mgrid[y1:y2, x1:x2]
    valid = np.isfinite(depth) & (depth > 0)
    if segmentation_id is not None:
        valid &= frame.segmentation[y1:y2, x1:x2] == segmentation_id
    if pixel_mask is not None:
        valid &= pixel_mask[y1:y2, x1:x2]
    if not np.any(valid):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )

    z = depth[valid].astype(np.float64)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    fx, fy = float(frame.intrinsic[0, 0]), float(frame.intrinsic[1, 1])
    cx, cy = float(frame.intrinsic[0, 2]), float(frame.intrinsic[1, 2])
    if fx == 0 or fy == 0:
        raise PerceptionError("camera focal lengths must be nonzero")
    camera_points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    colors = frame.rgb[y1:y2, x1:x2][valid]
    return camera_points.astype(np.float32), colors.astype(np.uint8, copy=False)


def backproject_camera(
    frame: CameraFrame,
    bbox: BBox | None = None,
    segmentation_id: int | None = None,
    *,
    pixel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project valid pixels into the OpenCV camera coordinate frame."""

    points, _ = _selected_camera_data(frame, bbox, segmentation_id, pixel_mask)
    return points


def point_colors(
    frame: CameraFrame,
    bbox: BBox | None = None,
    segmentation_id: int | None = None,
    *,
    pixel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return RGB colors aligned with :func:`backproject_camera` points."""

    _, colors = _selected_camera_data(frame, bbox, segmentation_id, pixel_mask)
    return colors


def backproject(
    frame: CameraFrame,
    bbox: BBox | None = None,
    segmentation_id: int | None = None,
    *,
    pixel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project valid pixels into world coordinates."""

    camera_points = backproject_camera(
        frame, bbox, segmentation_id, pixel_mask=pixel_mask
    )
    if len(camera_points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    homogeneous = np.column_stack((camera_points, np.ones(len(camera_points))))
    world = (frame.world_from_camera.astype(np.float64) @ homogeneous.T).T[:, :3]
    return world.astype(np.float32)


def crop_local_workspace_camera(
    workspace_points_camera: object,
    target_points_camera: object,
    world_from_camera: object,
    *,
    radius_xy_m: float = 0.15,
    margin_below_m: float = 0.06,
    margin_above_m: float = 0.12,
) -> np.ndarray:
    """Keep the local tabletop neighborhood used for GraspNet inference.

    The returned points remain in the camera frame. World coordinates are used
    only to reject robot/background geometry and to preserve nearby collision
    geometry around the grounded target.
    """

    workspace = np.asarray(workspace_points_camera, dtype=np.float32)
    target = np.asarray(target_points_camera, dtype=np.float32)
    transform = np.asarray(world_from_camera, dtype=np.float64)
    for name, points in (("workspace", workspace), ("target", target)):
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) == 0
            or not np.isfinite(points).all()
        ):
            raise PerceptionError(f"{name} points must be a finite nonempty (N, 3) array")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise PerceptionError("world_from_camera must be a finite 4x4 matrix")
    for name, value in (
        ("radius_xy_m", radius_xy_m),
        ("margin_below_m", margin_below_m),
        ("margin_above_m", margin_above_m),
    ):
        if not np.isfinite(value) or value <= 0:
            raise PerceptionError(f"{name} must be positive and finite")

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    workspace_world = (rotation @ workspace.astype(np.float64).T).T + translation
    target_world = (rotation @ target.astype(np.float64).T).T + translation
    center_xy = np.mean(target_world[:, :2], axis=0)
    low_z = float(np.min(target_world[:, 2]) - margin_below_m)
    high_z = float(np.max(target_world[:, 2]) + margin_above_m)
    mask = (
        (np.abs(workspace_world[:, 0] - center_xy[0]) <= radius_xy_m)
        & (np.abs(workspace_world[:, 1] - center_xy[1]) <= radius_xy_m)
        & (workspace_world[:, 2] >= low_z)
        & (workspace_world[:, 2] <= high_z)
    )
    cropped = workspace[mask]
    if len(cropped) == 0:
        raise PerceptionError("local GraspNet workspace is empty")
    return cropped


def match_instance(
    bbox: BBox,
    segmentation: np.ndarray,
    id_to_instance: Mapping[int, str],
) -> str:
    """Resolve a VLM bbox to the known simulator instance with most overlap."""

    if segmentation.ndim != 2:
        raise PerceptionError("segmentation must be a 2D array")
    height, width = segmentation.shape
    x1, y1, x2, y2 = bbox.as_pixels(width, height)
    values, counts = np.unique(segmentation[y1:y2, x1:x2], return_counts=True)
    known = [
        (int(count), id_to_instance[int(value)])
        for value, count in zip(values, counts, strict=True)
        if int(value) != 0 and int(value) in id_to_instance
    ]
    if not known:
        raise SemanticValidationError("bbox does not overlap a known instance")
    known.sort(key=lambda pair: pair[0], reverse=True)
    if len(known) > 1 and known[0][0] == known[1][0]:
        raise SemanticValidationError("bbox instance overlap is ambiguous")
    return known[0][1]


def nearest_instance(
    candidate_ids: Sequence[str],
    centers: Mapping[str, np.ndarray],
    destination_id: str,
    *,
    ambiguity_margin: float = 0.05,
) -> str:
    """Return the unique nearest candidate using XY world distance."""

    if not candidate_ids:
        raise SemanticValidationError("nearest relation has no candidates")
    if destination_id not in centers:
        raise SemanticValidationError(f"unknown destination center: {destination_id}")
    missing = [candidate for candidate in candidate_ids if candidate not in centers]
    if missing:
        raise SemanticValidationError(f"unknown candidate centers: {missing}")
    destination = np.asarray(centers[destination_id], dtype=np.float64)
    ranked = sorted(
        (
            float(
                np.linalg.norm(
                    np.asarray(centers[candidate], dtype=np.float64)[:2]
                    - destination[:2]
                )
            ),
            candidate,
        )
        for candidate in candidate_ids
    )
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < ambiguity_margin:
        raise SemanticValidationError("nearest relation is ambiguous")
    return ranked[0][1]
