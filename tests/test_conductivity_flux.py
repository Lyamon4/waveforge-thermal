"""Проверки conductivity interpolation и face coefficients."""

import numpy as np
import pytest

from waveforge.physics.conductivity import (
    harmonic_mean,
    interpolate_conductivity,
    validate_conductivity,
)


def test_harmonic_face_matches_independent_two_material_value() -> None:
    """Arithmetic mean 10.5 не должен пройти этот test."""
    face = harmonic_mean(np.array([1.0]), np.array([20.0]))
    np.testing.assert_allclose(face, [40.0 / 21.0], rtol=1e-12, atol=0.0)


def test_conductivity_interpolation_uses_cubic_penalization() -> None:
    """Потеря exponent p=3 должна менять middle-design conductivity."""
    design = np.array([0.0, 0.5, 1.0])
    conductivity = interpolate_conductivity(design)
    np.testing.assert_allclose(conductivity, [1.0, 3.375, 20.0])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_conductivity_rejects_non_positive_or_non_finite(bad: float) -> None:
    """Invalid k не должен доходить до sparse assembly."""
    conductivity = np.ones((2, 2), dtype=np.float64)
    conductivity[0, 0] = bad
    with pytest.raises(ValueError, match="strictly positive"):
        validate_conductivity(conductivity, (2, 2))


def test_conductivity_shape_must_match_grid() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_conductivity(np.ones((2, 3)), (2, 2))


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan])
def test_design_interpolation_rejects_values_outside_unit_interval(
    bad: float,
) -> None:
    design = np.full((2, 2), 0.5)
    design[0, 0] = bad
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        interpolate_conductivity(design)
