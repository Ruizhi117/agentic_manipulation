"""Deterministic semantic tabletop layouts independent of the simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_ALLOWED_SHAPES = {"box", "cylinder", "banana", "bin"}


def bin_wall_components(
    half_size_xyz: np.ndarray, wall_thickness: float
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return local center/half-size pairs for an open-top bin."""

    half_size = np.asarray(half_size_xyz, dtype=np.float64)
    if half_size.shape != (3,) or np.any(half_size <= 0):
        raise ValueError("half_size_xyz must be a positive 3-vector")
    if not math.isfinite(wall_thickness) or not 0 < wall_thickness < min(
        half_size[:2]
    ):
        raise ValueError("wall_thickness must fit inside the bin")
    hx, hy, height = half_size
    wall = wall_thickness
    return (
        (np.array([0.0, 0.0, wall / 2]), np.array([hx, hy, wall / 2])),
        (
            np.array([-hx + wall / 2, 0.0, height / 2]),
            np.array([wall / 2, hy, height / 2]),
        ),
        (
            np.array([hx - wall / 2, 0.0, height / 2]),
            np.array([wall / 2, hy, height / 2]),
        ),
        (
            np.array([0.0, -hy + wall / 2, height / 2]),
            np.array([hx, wall / 2, height / 2]),
        ),
        (
            np.array([0.0, hy - wall / 2, height / 2]),
            np.array([hx, wall / 2, height / 2]),
        ),
    )


@dataclass(frozen=True)
class SceneObjectSpec:
    instance_id: str
    semantic_label: str
    center_xyz: np.ndarray
    half_size_xyz: np.ndarray
    rgba: tuple[float, float, float, float]
    shape: str
    is_static: bool

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xyz, dtype=np.float64)
        half_size = np.asarray(self.half_size_xyz, dtype=np.float64)
        object.__setattr__(self, "center_xyz", center)
        object.__setattr__(self, "half_size_xyz", half_size)
        if not self.instance_id.strip() or not self.semantic_label.strip():
            raise ValueError("instance_id and semantic_label must not be empty")
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center_xyz must be a finite 3-vector")
        if (
            half_size.shape != (3,)
            or not np.isfinite(half_size).all()
            or np.any(half_size <= 0)
        ):
            raise ValueError("half_size_xyz must be a positive finite 3-vector")
        if self.shape not in _ALLOWED_SHAPES:
            raise ValueError(f"shape must be one of {sorted(_ALLOWED_SHAPES)}")
        if len(self.rgba) != 4 or not all(
            math.isfinite(v) and 0.0 <= v <= 1.0 for v in self.rgba
        ):
            raise ValueError("rgba must contain four finite values in [0, 1]")


def _jittered(
    rng: np.random.Generator, xyz: tuple[float, float, float]
) -> np.ndarray:
    result = np.asarray(xyz, dtype=np.float64).copy()
    result[:2] += rng.uniform(-0.008, 0.008, size=2)
    return result


def sample_layout(seed: int) -> tuple[SceneObjectSpec, ...]:
    """Return a non-overlapping scene with a unique nearest banana."""

    rng = np.random.default_rng(seed)
    return (
        SceneObjectSpec(
            "tomato_can_1",
            "tomato_can",
            _jittered(rng, (-0.42, -0.18, 0.045)),
            np.array([0.03, 0.03, 0.045]),
            (0.86, 0.08, 0.06, 1.0),
            "cylinder",
            False,
        ),
        SceneObjectSpec(
            "tomato_can_2",
            "tomato_can",
            _jittered(rng, (-0.42, -0.07, 0.045)),
            np.array([0.03, 0.03, 0.045]),
            (0.86, 0.08, 0.06, 1.0),
            "cylinder",
            False,
        ),
        SceneObjectSpec(
            "banana_1",
            "banana",
            _jittered(rng, (-0.42, 0.05, 0.025)),
            np.array([0.060, 0.022, 0.022]),
            (0.96, 0.78, 0.06, 1.0),
            "banana",
            False,
        ),
        SceneObjectSpec(
            "banana_2",
            "banana",
            _jittered(rng, (-0.44, -0.30, 0.025)),
            np.array([0.060, 0.022, 0.022]),
            (0.96, 0.78, 0.06, 1.0),
            "banana",
            False,
        ),
        SceneObjectSpec(
            "gray_bin",
            "gray_bin",
            np.array([-0.60, -0.30, 0.015]),
            np.array([0.08, 0.08, 0.075]),
            (0.45, 0.45, 0.48, 1.0),
            "bin",
            True,
        ),
        SceneObjectSpec(
            "purple_bin",
            "purple_bin",
            np.array([-0.60, 0.08, 0.015]),
            np.array([0.08, 0.08, 0.075]),
            (0.52, 0.20, 0.70, 1.0),
            "bin",
            True,
        ),
    )
