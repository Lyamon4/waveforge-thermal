"""Locked implicit-Euler validation tests."""

import numpy as np
import pytest

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.manufactured_solutions import normalized_rectangular_source
from waveforge.physics.steady_solver import assemble_steady_system, solve_steady
from waveforge.physics.transient_solver import (
    TransientConfig,
    solve_transient,
)
from waveforge.physics.validation import relative_l2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0.0, "n_steps": 10, "rho_c": 1.0}, "dt"),
        ({"dt": -0.1, "n_steps": 10, "rho_c": 1.0}, "dt"),
        ({"dt": 0.1, "n_steps": 0, "rho_c": 1.0}, "n_steps"),
        ({"dt": 0.1, "n_steps": 10, "rho_c": 0.0}, "rho_c"),
    ],
)
def test_transient_config_rejects_non_positive_values(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TransientConfig(**kwargs)


def test_zero_initial_zero_source_stays_zero() -> None:
    """Dirichlet/source indexing не должен создавать artificial heat."""
    grid = Grid2D(nx=8, ny=8)
    result = solve_transient(
        grid=grid,
        conductivity=np.ones(grid.shape),
        source=np.zeros(grid.shape),
        bcs=BoundaryConditions.production(),
        initial_temperature=np.zeros(grid.shape),
        config=TransientConfig(dt=0.1, n_steps=4, rho_c=1.0),
    )
    np.testing.assert_array_equal(result.temperatures, 0.0)
    np.testing.assert_allclose(result.times, [0.0, 0.1, 0.2, 0.3, 0.4])
    assert result.temperatures.shape == (5, 8, 8)


def test_time_dependent_source_is_evaluated_at_implicit_step_time() -> None:
    grid = Grid2D(nx=4, ny=4)
    evaluated_times: list[float] = []

    def source(time: float) -> np.ndarray:
        evaluated_times.append(time)
        return np.zeros(grid.shape)

    solve_transient(
        grid=grid,
        conductivity=np.ones(grid.shape),
        source=source,
        bcs=BoundaryConditions.production(),
        initial_temperature=np.zeros(grid.shape),
        config=TransientConfig(dt=0.05, n_steps=3, rho_c=1.0),
    )
    np.testing.assert_allclose(evaluated_times, [0.05, 0.1, 0.15])


def test_transient_rejects_non_finite_initial_temperature() -> None:
    grid = Grid2D(nx=4, ny=4)
    initial = np.zeros(grid.shape)
    initial[0, 0] = np.nan
    with pytest.raises(ValueError, match="initial"):
        solve_transient(
            grid=grid,
            conductivity=np.ones(grid.shape),
            source=np.zeros(grid.shape),
            bcs=BoundaryConditions.production(),
            initial_temperature=initial,
            config=TransientConfig(dt=0.1, n_steps=2),
        )


def test_transient_converges_to_steady_locked_fixture() -> None:
    """Transient и steady solver должны разделять одну reference semantics."""
    grid = Grid2D(nx=32, ny=32)
    conductivity = np.ones(grid.shape)
    source = normalized_rectangular_source(grid, 0.4, 0.6, 0.65, 0.85)
    bcs = BoundaryConditions.production()
    steady = solve_steady(grid, conductivity, source, bcs)
    transient = solve_transient(
        grid=grid,
        conductivity=conductivity,
        source=source,
        bcs=bcs,
        initial_temperature=np.zeros(grid.shape),
        config=TransientConfig(dt=0.02, n_steps=200, rho_c=1.0),
    )

    final = transient.temperatures[-1]
    system = assemble_steady_system(grid, conductivity, source, bcs)
    residual = np.linalg.norm(system.matrix @ final.ravel() - system.rhs) / max(
        np.linalg.norm(system.rhs), 1.0
    )

    assert transient.times[-1] == pytest.approx(4.0)
    assert relative_l2(final, steady.temperature) <= 5e-4
    assert residual <= 5e-4
    assert np.all(np.isfinite(transient.temperatures))


def test_implicit_euler_timestep_error_decreases() -> None:
    """Все trajectories сравниваются только в t=0.2."""
    grid = Grid2D(nx=32, ny=32)
    _, y = grid.mesh
    initial = np.sin(np.pi * y / 2.0)
    common = {
        "grid": grid,
        "conductivity": np.ones(grid.shape),
        "source": np.zeros(grid.shape),
        "bcs": BoundaryConditions.production(),
        "initial_temperature": initial,
    }
    coarse = solve_transient(
        **common,
        config=TransientConfig(dt=0.02, n_steps=10, rho_c=1.0),
    )
    half = solve_transient(
        **common,
        config=TransientConfig(dt=0.01, n_steps=20, rho_c=1.0),
    )
    reference = solve_transient(
        **common,
        config=TransientConfig(dt=0.00125, n_steps=160, rho_c=1.0),
    )

    np.testing.assert_allclose(
        [coarse.times[-1], half.times[-1], reference.times[-1]],
        0.2,
        rtol=0.0,
        atol=1e-14,
    )
    coarse_error = relative_l2(coarse.temperatures[-1], reference.temperatures[-1])
    half_error = relative_l2(half.temperatures[-1], reference.temperatures[-1])
    assert half_error < coarse_error
    assert half_error <= 0.75 * coarse_error
