"""Manufactured-solution и grid-convergence tests."""

import numpy as np
import pytest

from waveforge.physics.grid import Grid2D
from waveforge.physics.manufactured_solutions import sine_manufactured_fixture
from waveforge.physics.steady_solver import solve_steady
from waveforge.physics.validation import relative_l2, symmetry_defect


def test_relative_l2_uses_exact_field_norm() -> None:
    exact = np.array([3.0, 4.0])
    predicted = np.array([0.0, 0.0])
    assert relative_l2(predicted, exact) == pytest.approx(1.0)


def test_symmetry_defect_normalizes_by_peak_magnitude() -> None:
    field = np.array([[1.0, 2.0]])
    assert symmetry_defect(field) == pytest.approx(0.5)


def test_manufactured_error_decreases_with_refinement() -> None:
    """Wrong sign, BC distance или stencil order не должны пройти refinement."""
    errors: list[float] = []
    for resolution in (32, 64, 128):
        fixture = sine_manufactured_fixture(Grid2D(resolution, resolution))
        result = solve_steady(*fixture.solver_arguments())
        errors.append(relative_l2(result.temperature, fixture.exact))

    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] >= 1.5
    assert errors[1] / errors[2] >= 1.5
    order = np.log(errors[0] / errors[2]) / np.log(4.0)
    assert order >= 1.5
