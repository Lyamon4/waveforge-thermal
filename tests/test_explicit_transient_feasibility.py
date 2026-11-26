"""Gate 2 explicit-scheme feasibility calculations."""

import numpy as np
import pytest
import torch

from waveforge.experiments.assess_explicit_transient import (
    explicit_flux_step,
    explicit_flux_step_prepared,
    explicit_stability_limit,
    prepare_explicit_flux_coefficients,
    steps_for_horizon,
)
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import assemble_steady_system


def test_stability_limit_includes_bottom_half_cell_dirichlet_flux() -> None:
    """Удаление factor 2 на cooled face должно нарушить hand-derived CFL."""
    grid = Grid2D(nx=4, ny=4)

    limit = explicit_stability_limit(grid, k_max=2.0, rho_c=1.0)

    assert limit.max_diagonal == pytest.approx(160.0)
    assert limit.dt_monotone == pytest.approx(0.00625)


def test_steps_for_horizon_rounds_up_without_undershooting() -> None:
    assert steps_for_horizon(t_final=0.2, dt=0.03) == 7


def test_torch_flux_step_matches_locked_scipy_operator() -> None:
    """Изменение face signs или boundary term должно расходиться с reference."""
    grid = Grid2D(nx=4, ny=3)
    conductivity = np.linspace(1.0, 2.0, grid.size).reshape(grid.shape)
    source = np.linspace(0.0, 0.5, grid.size).reshape(grid.shape)
    temperature = np.linspace(0.0, 1.0, grid.size).reshape(grid.shape)
    dt = 1.0e-3
    system = assemble_steady_system(
        grid,
        conductivity,
        source,
        BoundaryConditions.production(),
    )
    expected = temperature.ravel() + dt * (
        system.rhs - system.matrix @ temperature.ravel()
    )

    actual = explicit_flux_step(
        torch.from_numpy(temperature).unsqueeze(0),
        torch.from_numpy(conductivity),
        torch.from_numpy(source).unsqueeze(0),
        dt=dt,
        rho_c=1.0,
        dx=grid.dx,
        dy=grid.dy,
    )

    np.testing.assert_allclose(actual.squeeze(0).numpy().ravel(), expected)


def test_prepared_coefficients_preserve_flux_step() -> None:
    temperature = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 24.0
    conductivity = torch.linspace(1.0, 2.0, 12, dtype=torch.float64).reshape(3, 4)
    source = torch.full_like(temperature, 0.25)
    coefficients = prepare_explicit_flux_coefficients(
        conductivity,
        dx=0.25,
        dy=1.0 / 3.0,
    )

    prepared = explicit_flux_step_prepared(
        temperature,
        coefficients,
        source,
        dt=1.0e-3,
        rho_c=1.0,
    )
    wrapped = explicit_flux_step(
        temperature,
        conductivity,
        source,
        dt=1.0e-3,
        rho_c=1.0,
        dx=0.25,
        dy=1.0 / 3.0,
    )

    torch.testing.assert_close(prepared, wrapped)
