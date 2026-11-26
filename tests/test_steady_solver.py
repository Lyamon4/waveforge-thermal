"""Analytical и boundary tests steady SciPy reference solver."""

import numpy as np
import pytest

from waveforge.physics.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditions,
)
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import assemble_steady_system, solve_steady


def test_constant_field_with_equal_dirichlet_values() -> None:
    """Потеря Dirichlet RHS contribution разрушает constant field."""
    grid = Grid2D(nx=8, ny=6)
    result = solve_steady(
        grid,
        np.full(grid.shape, 3.0),
        np.zeros(grid.shape),
        BoundaryConditions.all_dirichlet(2.5),
    )
    np.testing.assert_allclose(result.temperature, 2.5, atol=1e-11, rtol=0.0)
    assert result.normalized_residual <= 1e-11


def test_linear_solution_matches_cell_centers() -> None:
    """Неверная half-cell distance не воспроизводит T=x."""
    grid = Grid2D(nx=16, ny=10)
    result = solve_steady(
        grid,
        np.ones(grid.shape),
        np.zeros(grid.shape),
        BoundaryConditions.left_right(0.0, 1.0),
    )
    exact = np.broadcast_to(grid.x_centers, grid.shape)
    np.testing.assert_allclose(result.temperature, exact, atol=1e-11, rtol=0.0)
    assert result.normalized_residual <= 1e-11


def test_bottom_dirichlet_half_cell_contribution_is_exact() -> None:
    grid = Grid2D(nx=3, ny=2)
    conductivity = np.full(grid.shape, 4.0)
    bcs = BoundaryConditions(
        left=BoundaryCondition("neumann"),
        right=BoundaryCondition("neumann"),
        bottom=BoundaryCondition("dirichlet", 2.0),
        top=BoundaryCondition("neumann"),
    )
    system = assemble_steady_system(grid, conductivity, np.zeros(grid.shape), bcs)

    expected_bottom_rhs = 2.0 * 4.0 / grid.dy**2 * 2.0
    boundary_rhs = system.dirichlet_rhs.reshape(grid.shape)
    np.testing.assert_allclose(boundary_rhs[0], expected_bottom_rhs)
    np.testing.assert_array_equal(boundary_rhs[1], 0.0)


def test_neumann_top_adds_no_diagonal_contribution() -> None:
    grid = Grid2D(nx=3, ny=2)
    conductivity = np.ones(grid.shape)
    production = assemble_steady_system(
        grid,
        conductivity,
        np.zeros(grid.shape),
        BoundaryConditions.production(),
    )
    top_dirichlet = BoundaryConditions(
        left=BoundaryCondition("neumann"),
        right=BoundaryCondition("neumann"),
        bottom=BoundaryCondition("dirichlet", 0.0),
        top=BoundaryCondition("dirichlet", 0.0),
    )
    with_top = assemble_steady_system(
        grid, conductivity, np.zeros(grid.shape), top_dirichlet
    )

    diagonal_delta = with_top.matrix.diagonal() - production.matrix.diagonal()
    expected = np.zeros(grid.shape)
    expected[-1, :] = 2.0 / grid.dy**2
    np.testing.assert_allclose(diagonal_delta.reshape(grid.shape), expected)


def test_residual_uses_assembled_rhs_not_raw_source() -> None:
    grid = Grid2D(nx=4, ny=4)
    result = solve_steady(
        grid,
        np.ones(grid.shape),
        np.zeros(grid.shape),
        BoundaryConditions.all_dirichlet(3.0),
    )
    assert np.linalg.norm(result.system.rhs) > 0.0
    assert np.linalg.norm(result.system.source_rhs) == 0.0
    assert result.normalized_residual <= 1e-11


def test_pure_neumann_problem_is_rejected_before_solve() -> None:
    grid = Grid2D(nx=4, ny=4)
    with pytest.raises(ValueError, match="Dirichlet"):
        solve_steady(
            grid,
            np.ones(grid.shape),
            np.zeros(grid.shape),
            BoundaryConditions.all_neumann(),
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_source_rejects_non_finite_values(bad: float) -> None:
    grid = Grid2D(nx=3, ny=2)
    source = np.zeros(grid.shape)
    source[0, 0] = bad
    with pytest.raises(ValueError, match="source"):
        assemble_steady_system(
            grid,
            np.ones(grid.shape),
            source,
            BoundaryConditions.production(),
        )


def test_source_shape_must_match_grid() -> None:
    grid = Grid2D(nx=3, ny=2)
    with pytest.raises(ValueError, match="source shape"):
        assemble_steady_system(
            grid,
            np.ones(grid.shape),
            np.zeros((3, 3)),
            BoundaryConditions.production(),
        )
