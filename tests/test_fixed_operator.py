from __future__ import annotations

import numpy as np
import pytest

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.fixed_operator import UniformPlateFactorization
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady


def _sources(grid: Grid2D) -> np.ndarray:
    bounds = (
        (0.10, 0.30, 0.60, 0.80),
        (0.40, 0.60, 0.55, 0.75),
        (0.70, 0.90, 0.65, 0.85),
    )
    return np.stack(
        [area_overlap_rectangular_source(grid, item, 1.0) for item in bounds]
    )


def test_reusable_factorization_matches_independent_scipy_fields() -> None:
    grid = Grid2D(nx=64, ny=64)
    sources = _sources(grid)
    reusable = UniformPlateFactorization(grid_size=64, conductivity=1.0)

    result = reusable.solve_many(sources)
    reference = np.stack(
        [
            solve_steady(
                grid,
                np.ones(grid.shape, dtype=np.float64),
                source,
                BoundaryConditions.production(),
            ).temperature
            for source in sources
        ]
    )

    np.testing.assert_allclose(result.temperature, reference, rtol=1e-8, atol=1e-9)
    assert result.maximum_normalized_residual < 1e-10


def test_reusable_factorization_solves_all_rhs_in_one_factorization() -> None:
    reusable = UniformPlateFactorization(grid_size=64, conductivity=1.0)
    factorization_id = id(reusable.factorization)
    sources = _sources(Grid2D(nx=64, ny=64))

    first = reusable.solve_many(sources)
    second = reusable.solve_many(sources[::-1].copy())

    assert id(reusable.factorization) == factorization_id
    assert reusable.factorization_count == 1
    np.testing.assert_allclose(first.temperature[::-1], second.temperature)


def test_reusable_factorization_reproduces_locked_canonical_scale() -> None:
    grid = Grid2D(nx=64, ny=64)
    source = area_overlap_rectangular_source(
        grid,
        (0.40, 0.60, 0.65, 0.85),
        1.0,
    )

    result = UniformPlateFactorization(64, 1.0).solve_many(source[None, ...])

    assert float(np.max(result.temperature)) == pytest.approx(
        0.900613256638055, rel=0.0, abs=1e-14
    )


@pytest.mark.parametrize(
    "sources",
    [
        np.zeros((64, 64)),
        np.zeros((2, 32, 32)),
        np.full((1, 64, 64), np.nan),
    ],
)
def test_reusable_factorization_rejects_invalid_rhs_batches(
    sources: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        UniformPlateFactorization(64, 1.0).solve_many(sources)


def test_reusable_factorization_rejects_nonpositive_conductivity() -> None:
    with pytest.raises(ValueError, match="conductivity"):
        UniformPlateFactorization(64, 0.0)
