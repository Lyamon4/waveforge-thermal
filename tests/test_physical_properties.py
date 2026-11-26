"""Symmetry and conductivity-monotonicity tests."""

import numpy as np

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.manufactured_solutions import normalized_rectangular_source
from waveforge.physics.steady_solver import solve_steady
from waveforge.physics.validation import symmetry_defect


def test_symmetric_source_produces_symmetric_temperature() -> None:
    grid = Grid2D(nx=32, ny=32)
    source = normalized_rectangular_source(grid, 0.4, 0.6, 0.65, 0.85)
    result = solve_steady(
        grid,
        np.ones(grid.shape),
        source,
        BoundaryConditions.production(),
    )
    assert symmetry_defect(result.temperature) <= 1e-10


def test_uniform_conductivity_increase_does_not_raise_peak_temperature() -> None:
    grid = Grid2D(nx=32, ny=32)
    source = normalized_rectangular_source(grid, 0.4, 0.6, 0.65, 0.85)
    bcs = BoundaryConditions.production()
    low_peak = solve_steady(
        grid,
        np.ones(grid.shape),
        source,
        bcs,
    ).temperature.max()
    high_peak = solve_steady(
        grid,
        np.full(grid.shape, 20.0),
        source,
        bcs,
    ).temperature.max()

    assert high_peak <= low_peak + 1e-12
