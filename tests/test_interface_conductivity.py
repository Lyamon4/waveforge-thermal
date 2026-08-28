"""Analytical two-layer heterogeneous-conductivity tests."""

import numpy as np

from waveforge.physics.grid import Grid2D
from waveforge.physics.manufactured_solutions import two_layer_fixture
from waveforge.physics.steady_solver import solve_steady
from waveforge.physics.validation import relative_l2, two_layer_interface_flux


def test_two_layer_solution_and_interface_flux_converge() -> None:
    """Shared arithmetic face k или разрыв flux должны ломать этот test."""
    previous_error: float | None = None
    exact_flux = 40.0 / 21.0

    for resolution in (32, 64, 128):
        fixture = two_layer_fixture(Grid2D(resolution, resolution))
        result = solve_steady(*fixture.solver_arguments())
        error = relative_l2(result.temperature, fixture.exact)
        flux = two_layer_interface_flux(
            fixture.grid,
            fixture.conductivity,
            result.temperature,
        )

        assert error <= 1e-11
        np.testing.assert_allclose(flux, exact_flux, rtol=1e-11, atol=0.0)
        assert np.ptp(flux) / exact_flux <= 1e-11
        if previous_error is not None:
            assert error <= max(1e-11, 2.0 * previous_error)
        previous_error = error
