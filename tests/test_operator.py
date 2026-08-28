"""Проверки algebraic admissibility finite-volume operator."""

import numpy as np

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import assemble_steady_system


def test_operator_is_symmetric_m_matrix_and_positive_definite() -> None:
    """Асимметрия face assembly или потеря Dirichlet anchor должны проявиться."""
    grid = Grid2D(nx=5, ny=4)
    conductivity = np.linspace(1.0, 20.0, 20).reshape(grid.shape)
    system = assemble_steady_system(
        grid,
        conductivity,
        np.zeros(grid.shape),
        BoundaryConditions.production(),
    )
    dense = system.matrix.toarray()

    np.testing.assert_allclose(dense, dense.T, atol=1e-13, rtol=0.0)
    assert np.all(np.diag(dense) > 0.0)
    off_diagonal = dense - np.diag(np.diag(dense))
    assert np.max(off_diagonal) <= 1e-14
    assert np.linalg.eigvalsh(dense).min() > 1e-12


def test_operator_connects_only_axis_adjacent_cells() -> None:
    """Flattening не должно создавать wrap-around между строками grid."""
    grid = Grid2D(nx=4, ny=3)
    system = assemble_steady_system(
        grid,
        np.ones(grid.shape),
        np.zeros(grid.shape),
        BoundaryConditions.production(),
    )
    dense = system.matrix.toarray()
    assert dense[3, 4] == 0.0
    assert dense[4, 3] == 0.0
