"""Tests for literal Gate 2A objective definitions."""

import math

import pytest
import torch

from waveforge.design.objectives import (
    normalized_smooth_max,
    objective_components,
    total_variation,
)


def test_normalized_smooth_max_matches_hand_derived_formula() -> None:
    """Dropping the normalization by element count must fail."""
    values = torch.tensor([0.0, 1.0], dtype=torch.float64)
    alpha = 2.0
    expected = 1.0 + math.log((math.exp(-2.0) + 1.0) / 2.0) / alpha

    actual = normalized_smooth_max(values, alpha=alpha)

    assert actual.item() == pytest.approx(expected, abs=1e-15)


def test_normalized_smooth_max_is_invariant_to_shape() -> None:
    """Using a dimension-specific normalization must fail."""
    values = torch.tensor([0.2, 0.7, 0.4, 1.1], dtype=torch.float64)

    flat = normalized_smooth_max(values, alpha=50.0)
    matrix = normalized_smooth_max(values.reshape(2, 2), alpha=50.0)

    torch.testing.assert_close(flat, matrix, rtol=0.0, atol=0.0)


def test_total_variation_matches_literal_two_direction_means() -> None:
    """A sum, isotropic norm, or boundary term must fail this definition."""
    design = torch.tensor(
        [[0.0, 1.0, 3.0], [2.0, 2.0, 4.0]],
        dtype=torch.float64,
    )
    expected = 1.25 + 4.0 / 3.0

    actual = total_variation(design)

    assert actual.item() == pytest.approx(expected, abs=1e-15)


def test_objective_keeps_thermal_sum_float64_and_design_gradient_float32() -> None:
    """Summing direct regularizers in float32 must fail mixed precision."""
    design = torch.full((4, 4), 0.25, dtype=torch.float32, requires_grad=True)
    temperatures = torch.linspace(
        0.0,
        1.0,
        2 * 4 * 4,
        dtype=torch.float64,
    ).reshape(2, 4, 4)

    components = objective_components(
        temperatures,
        design,
        alpha=50.0,
        tv_weight=1e-3,
        binarization_weight=0.005,
    )
    components.total.backward()

    assert components.total.dtype is torch.float64
    assert components.thermal_smooth.dtype is torch.float64
    assert components.total_variation.dtype is torch.float32
    assert components.binarization_penalty.dtype is torch.float32
    assert design.grad is not None
    assert design.grad.dtype is torch.float32
    assert torch.isfinite(design.grad).all()
