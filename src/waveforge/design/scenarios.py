"""Deterministic heat-source construction for Gate 2A."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from waveforge.physics.grid import Grid2D


def area_overlap_rectangular_source(
    grid: Grid2D,
    bounds: Sequence[float],
    power: float,
) -> NDArray[np.float64]:
    """Rasterize a half-open rectangle using exact cell-area overlap.

    Bounds are ordered as ``(x_min, x_max, y_min, y_max)``. The returned
    cell-centered source density integrates to ``power`` over the grid.
    """
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("rectangle bounds must contain four finite values")
    if not np.isfinite(power):
        raise ValueError("power must be finite")

    x_min, x_max, y_min, y_max = values
    if (
        x_min < 0.0
        or x_max > grid.lx
        or y_min < 0.0
        or y_max > grid.ly
        or x_max <= x_min
        or y_max <= y_min
    ):
        raise ValueError("rectangle must have positive area inside the domain")

    x_lo = np.arange(grid.nx, dtype=np.float64) * grid.dx
    y_lo = np.arange(grid.ny, dtype=np.float64) * grid.dy
    x_overlap = np.maximum(
        0.0,
        np.minimum(x_lo + grid.dx, x_max) - np.maximum(x_lo, x_min),
    )
    y_overlap = np.maximum(
        0.0,
        np.minimum(y_lo + grid.dy, y_max) - np.maximum(y_lo, y_min),
    )
    overlap_area = np.outer(y_overlap, x_overlap)
    rectangle_area = (x_max - x_min) * (y_max - y_min)
    cell_area = grid.dx * grid.dy
    return overlap_area * (power / (rectangle_area * cell_area))
