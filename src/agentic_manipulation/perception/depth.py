"""Depth-image visualization shared by VLM input and debug presentation."""

from __future__ import annotations

import numpy as np


def depth_grayscale_rgb(depth_m: object) -> np.ndarray:
    """Return uint8 RGB grayscale where nearer valid depth is brighter."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    valid = np.isfinite(depth) & (depth > 0)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        values = depth[valid]
        near, far = np.percentile(values, (2.0, 98.0))
        if float(far - near) <= np.finfo(np.float32).eps:
            gray[valid] = 255
        else:
            intensity = (far - depth[valid]) / (far - near)
            gray[valid] = np.rint(np.clip(intensity, 0.0, 1.0) * 255).astype(
                np.uint8
            )
    return np.repeat(gray[..., None], 3, axis=2)
