"""Independent global conservation test."""

import numpy as np

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.manufactured_solutions import normalized_rectangular_source
from waveforge.physics.steady_solver import solve_steady
from waveforge.physics.validation import dirichlet_outward_flux


def test_generated_heat_equals_dirichlet_outward_flux() -> None:
    """Small matrix residual не может скрыть нарушение conservation law."""
    grid = Grid2D(nx=48, ny=40)
    x, _ = grid.mesh
    conductivity = np.where(x < 0.5, 1.0, 20.0)
    source = normalized_rectangular_source(grid, 0.35, 0.65, 0.6, 0.8)
    bcs = BoundaryConditions.production()
    result = solve_steady(grid, conductivity, source, bcs)

    generated = float(source.sum() * grid.dx * grid.dy)
    outward = dirichlet_outward_flux(
        grid,
        conductivity,
        result.temperature,
        bcs,
    )
    imbalance = abs(generated - outward) / max(abs(generated), abs(outward), 1e-12)

    assert generated == 1.0
    assert imbalance <= 1e-10
