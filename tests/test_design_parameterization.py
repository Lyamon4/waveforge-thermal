"""Tests for the locked Gate 2A design parameterization."""

import pytest
import torch

from waveforge.design.constraints import binary_budget_satisfied, material_fraction
from waveforge.design.parameterization import (
    VolumeProjectionError,
    binary_design,
    filter_logits,
    gaussian_kernel,
    parameterize_design,
    project_volume,
)


def test_gaussian_kernel_is_normalized_symmetric_seven_by_seven() -> None:
    """A wrong radius, normalization, or asymmetric kernel must fail."""
    kernel = gaussian_kernel(sigma=1.0, radius=3, dtype=torch.float64)

    assert kernel.shape == (7, 7)
    assert kernel.sum().item() == pytest.approx(1.0, abs=1e-15)
    torch.testing.assert_close(kernel, torch.flip(kernel, dims=(0, 1)))
    singular_values = torch.linalg.svdvals(kernel)
    assert singular_values[1].item() <= 1e-15


def test_filter_impulse_response_matches_locked_kernel() -> None:
    """Changing the convolution kernel or its centering must fail."""
    impulse = torch.zeros((13, 13), dtype=torch.float64)
    impulse[6, 6] = 1.0
    kernel = gaussian_kernel(sigma=1.0, radius=3, dtype=torch.float64)

    filtered = filter_logits(impulse, sigma=1.0, radius=3, padding="reflect")

    torch.testing.assert_close(filtered[3:10, 3:10], kernel, rtol=0.0, atol=1e-15)


def test_reflect_filter_preserves_constant_field() -> None:
    """Zero padding or a non-unit kernel must fail at the boundaries."""
    constant = torch.full((16, 16), 2.75, dtype=torch.float64)

    filtered = filter_logits(constant, sigma=1.0, radius=3, padding="reflect")

    torch.testing.assert_close(
        filtered,
        constant,
        rtol=0.0,
        atol=2e-15,
    )


@pytest.mark.parametrize("beta", [1.0, 2.0, 4.0, 8.0])
def test_volume_projection_enforces_locked_fraction(beta: float) -> None:
    """A detached or approximate volume correction must fail this contract."""
    logits = torch.linspace(-1.5, 1.5, 64 * 64, dtype=torch.float64).reshape(64, 64)

    design, diagnostics = project_volume(logits, beta=beta)

    assert abs(design.mean().item() - 0.25) <= 1e-6
    assert -40.0 <= diagnostics.offset <= 40.0
    assert diagnostics.iterations <= 80
    assert diagnostics.converged


def test_volume_projection_rejects_unbracketable_logits() -> None:
    """Silently clipping an offset outside the locked bracket must fail."""
    logits = torch.full((8, 8), 100.0, dtype=torch.float64)

    with pytest.raises(VolumeProjectionError, match="bracket"):
        project_volume(logits, beta=8.0)


def test_volume_projection_implicit_gradient_matches_finite_difference() -> None:
    """Treating the bisection offset as constant must fail this test."""
    generator = torch.Generator().manual_seed(7301)
    logits = torch.randn((8, 8), generator=generator, dtype=torch.float64)
    logits.requires_grad_(True)
    upstream = torch.randn((8, 8), generator=generator, dtype=torch.float64)
    direction = torch.randn((8, 8), generator=generator, dtype=torch.float64) + 0.3
    direction = direction / torch.linalg.vector_norm(direction)
    beta = 3.0

    design, _ = project_volume(logits, beta=beta)
    objective = torch.sum(design * upstream)
    (gradient,) = torch.autograd.grad(objective, logits)
    automatic = torch.sum(gradient * direction).item()

    step = 1e-4
    plus, _ = project_volume((logits.detach() + step * direction), beta=beta)
    minus, _ = project_volume((logits.detach() - step * direction), beta=beta)
    finite_difference = (
        torch.sum(plus * upstream) - torch.sum(minus * upstream)
    ).item() / (2.0 * step)
    relative_error = abs(automatic - finite_difference) / max(
        abs(automatic),
        abs(finite_difference),
        1e-12,
    )

    assert relative_error <= 1e-6


def test_parameterization_keeps_design_float32_and_binary_threshold_is_strict() -> None:
    """A dtype promotion or movable threshold must fail this mixed-precision gate."""
    logits = torch.zeros((16, 16), dtype=torch.float32, device="cuda")

    result = parameterize_design(logits, beta=1.0)
    threshold_fixture = torch.tensor([0.4999, 0.5, 0.5001], device="cuda")

    assert result.design.shape == (64, 64)
    assert result.design.dtype is torch.float32
    assert result.design.device.type == "cuda"
    torch.testing.assert_close(
        binary_design(threshold_fixture),
        torch.tensor([0.0, 1.0, 1.0], device="cuda"),
    )


def test_binary_budget_uses_locked_target_and_inclusive_tolerance() -> None:
    """Moving the target or silently repairing a binary map must fail."""
    design = torch.cat(
        (
            torch.ones(26, dtype=torch.float64),
            torch.zeros(74, dtype=torch.float64),
        )
    )

    assert material_fraction(design) == pytest.approx(0.26)
    assert binary_budget_satisfied(design, target=0.25, tolerance=0.01)

    outside = design.clone()
    outside[26] = 1.0
    assert not binary_budget_satisfied(outside, target=0.25, tolerance=0.01)
