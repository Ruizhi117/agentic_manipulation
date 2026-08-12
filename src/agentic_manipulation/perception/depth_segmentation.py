"""Depth-map object segmentation — find distinct tabletop object regions.

Uses simple connected-component analysis on a depth threshold mask to
isolate individual objects without relying on simulator ground-truth
segmentation.  This mirrors what a real depth sensor would provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from agentic_manipulation.types import BBox, CameraFrame


# Broad sensor-valid range. Table removal is performed in calibrated world
# height rather than with camera-distance thresholds, which vary as the wrist
# camera moves.
_DEFAULT_DEPTH_LO_M = 0.01
_DEFAULT_DEPTH_HI_M = 2.0
_DEFAULT_MIN_AREA_PX = 80  # ignore speckle below this pixel count
# Dense tabletop layouts can contain one-pixel gaps between distinct objects.
# Closing those gaps merges instances, so the safe default preserves raw
# connectivity. Callers may opt into closing for noisier real sensors.
_DEFAULT_CLOSING_RADIUS_PX = 0
_DEFAULT_MIN_HEIGHT_ABOVE_TABLE_M = 0.004
_DEFAULT_MAX_HEIGHT_ABOVE_TABLE_M = 0.12
_TABLE_HEIGHT_BIN_M = 0.002


@dataclass(frozen=True)
class DepthRegion:
    """One connected depth blob with its normalized bounding box."""

    bbox: BBox
    area_px: int
    centroid_norm: tuple[float, float]
    mask: np.ndarray = field(repr=False, compare=False)


def segment_depth(
    frame: CameraFrame,
    *,
    depth_lo_m: float = _DEFAULT_DEPTH_LO_M,
    depth_hi_m: float = _DEFAULT_DEPTH_HI_M,
    min_area_px: int = _DEFAULT_MIN_AREA_PX,
    closing_radius_px: int = _DEFAULT_CLOSING_RADIUS_PX,
    min_height_above_table_m: float = _DEFAULT_MIN_HEIGHT_ABOVE_TABLE_M,
    max_height_above_table_m: float = _DEFAULT_MAX_HEIGHT_ABOVE_TABLE_M,
) -> tuple[DepthRegion, ...]:
    """Return depth-connected regions sorted left→right, top→bottom.

    Parameters
    ----------
    frame:
        Camera frame whose ``depth_m`` and ``rgb`` shape are used.
    depth_lo_m / depth_hi_m:
        Valid depth window in metres.  Pixels outside this range are
        treated as background.
    min_area_px:
        Regions with fewer pixels are discarded as noise.
    closing_radius_px:
        Radius of the binary closing disk used to merge nearby fragments
        of the same object.
    """
    depth = np.asarray(frame.depth_m, dtype=np.float32)
    height, width = depth.shape

    if not 0 <= min_height_above_table_m < max_height_above_table_m:
        raise ValueError(
            "table height bounds must satisfy 0 <= min < max"
        )

    valid = (
        np.isfinite(depth)
        & (depth >= depth_lo_m)
        & (depth <= depth_hi_m)
    )
    if not np.any(valid):
        return ()

    world_z = _world_height_map(frame)
    table_z = _estimate_table_height(world_z[valid])
    height_above_table = world_z - table_z
    mask = (
        valid
        & (height_above_table >= min_height_above_table_m)
        & (height_above_table <= max_height_above_table_m)
    )

    # Morphological closing to fill small holes inside objects.
    if closing_radius_px > 0:
        selem = _disk_kernel(closing_radius_px)
        mask = ndimage.binary_closing(mask, structure=selem)

    # Connected-component labelling.
    labelled, num_features = ndimage.label(mask)
    if num_features == 0:
        return ()

    regions: list[DepthRegion] = []
    for label_id in range(1, num_features + 1):
        component_mask = labelled == label_id
        rows, cols = np.nonzero(component_mask)
        area = len(rows)
        if area < min_area_px:
            continue
        x1_norm = float(cols.min()) / width
        x2_norm = float(cols.max() + 1) / width
        y1_norm = float(rows.min()) / height
        y2_norm = float(rows.max() + 1) / height
        # Clamp and ensure strictly positive area.
        x1_norm = max(0.0, min(1.0, x1_norm))
        x2_norm = max(0.0, min(1.0, x2_norm))
        y1_norm = max(0.0, min(1.0, y1_norm))
        y2_norm = max(0.0, min(1.0, y2_norm))
        if x2_norm - x1_norm < 0.005 or y2_norm - y1_norm < 0.005:
            continue
        centroid_u = float(cols.mean()) / width
        centroid_v = float(rows.mean()) / height
        regions.append(
            DepthRegion(
                bbox=BBox(x1_norm, y1_norm, x2_norm, y2_norm),
                area_px=area,
                centroid_norm=(centroid_u, centroid_v),
                mask=component_mask,
            )
        )

    # Sort by vertical position first, then horizontal.
    regions.sort(key=lambda r: (round(r.centroid_norm[1], 3), round(r.centroid_norm[0], 3)))
    return tuple(regions)


def _disk_kernel(radius: int) -> np.ndarray:
    """Boolean disk structuring element of *radius* pixels."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _world_height_map(frame: CameraFrame) -> np.ndarray:
    """Return one calibrated world-frame z value per depth pixel."""
    depth = np.asarray(frame.depth_m, dtype=np.float64)
    height, width = depth.shape
    vv, uu = np.mgrid[:height, :width]
    fx, fy = float(frame.intrinsic[0, 0]), float(frame.intrinsic[1, 1])
    cx, cy = float(frame.intrinsic[0, 2]), float(frame.intrinsic[1, 2])
    if fx == 0 or fy == 0:
        raise ValueError("camera focal lengths must be nonzero")
    camera_x = (uu - cx) * depth / fx
    camera_y = (vv - cy) * depth / fy
    rotation_z = np.asarray(frame.world_from_camera[2, :3], dtype=np.float64)
    return (
        rotation_z[0] * camera_x
        + rotation_z[1] * camera_y
        + rotation_z[2] * depth
        + float(frame.world_from_camera[2, 3])
    )


def _estimate_table_height(world_heights: np.ndarray) -> float:
    """Estimate the dominant horizontal plane with a 2 mm height histogram."""
    values = np.asarray(world_heights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("world heights must be a finite nonempty vector")
    bins = np.rint(values / _TABLE_HEIGHT_BIN_M).astype(np.int64)
    unique, counts = np.unique(bins, return_counts=True)
    dominant = unique[int(np.argmax(counts))]
    members = values[bins == dominant]
    return float(np.median(members))


def crop_rgb(frame: CameraFrame, bbox: BBox) -> np.ndarray:
    """Return the uint8 RGB sub-image defined by a normalized *bbox*."""
    height, width = frame.rgb.shape[:2]
    x1, y1, x2, y2 = bbox.as_pixels(width, height)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    return frame.rgb[y1:y2, x1:x2].copy()
