"""Analytical fixtures for independent solver validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D


@dataclass(frozen=True)
class SteadyFixture:
    """Полностью заданная steady problem с exact cell-centered field."""

    grid: Grid2D
    conductivity: NDArray[np.float64]
    source: NDArray[np.float64]
    bcs: BoundaryConditions
    exact: NDArray[np.float64]

    def solver_arguments(
        self,
    ) -> tuple[
        Grid2D,
        NDArray[np.float64],
        NDArray[np.float64],
        BoundaryConditions,
    ]:
        """Вернуть public steady-solver arguments."""
        return self.grid, self.conductivity, self.source, self.bcs


def sine_manufactured_fixture(grid: Grid2D) -> SteadyFixture:
    """Создать zero-Dirichlet fixture `sin(pi*x) sin(pi*y)`."""
    x, y = grid.mesh
    exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    conductivity = np.ones(grid.shape, dtype=np.float64)
    source = 2.0 * np.pi**2 * exact
    return SteadyFixture(
        grid=grid,
        conductivity=conductivity,
        source=source,
        bcs=BoundaryConditions.all_dirichlet(0.0),
        exact=exact,
    )


def two_layer_fixture(grid: Grid2D) -> SteadyFixture:
    """Создать aligned interface problem с `k_left=1`, `k_right=20`."""
    if grid.nx % 2 != 0:
        raise ValueError("two-layer interface requires an even nx")
    x, _ = grid.mesh
    conductivity = np.where(x < 0.5 * grid.lx, 1.0, 20.0)
    normalized_x = x / grid.lx
    exact = np.where(
        normalized_x <= 0.5,
        (40.0 / 21.0) * normalized_x,
        20.0 / 21.0 + (2.0 / 21.0) * (normalized_x - 0.5),
    )
    return SteadyFixture(
        grid=grid,
        conductivity=conductivity,
        source=np.zeros(grid.shape, dtype=np.float64),
        bcs=BoundaryConditions.left_right(0.0, 1.0),
        exact=exact,
    )


def normalized_rectangular_source(
    grid: Grid2D,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> NDArray[np.float64]:
    """Создать cell mask с единичной integrated source power."""
    bounds = np.asarray((x_min, x_max, y_min, y_max), dtype=np.float64)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("source bounds must be finite")
    if not (0.0 <= x_min < x_max <= grid.lx):
        raise ValueError("source x bounds must lie inside the domain")
    if not (0.0 <= y_min < y_max <= grid.ly):
        raise ValueError("source y bounds must lie inside the domain")

    x, y = grid.mesh
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    if not np.any(mask):
        raise ValueError("source rectangle contains no cell centers")
    source = mask.astype(np.float64)
    return source / (source.sum() * grid.dx * grid.dy)
