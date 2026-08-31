from __future__ import annotations

import pytest
import torch

from waveforge.physics.batched_cg import (
    BatchedCGConvergenceError,
    solve_batched_cg,
)
from waveforge.physics.cg import CGConfig


def test_batched_cg_solves_independent_systems_with_per_system_diagnostics() -> None:
    coefficients = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)[:, None, None]
    diagonal = coefficients.expand(3, 4, 4).clone()
    rhs = torch.arange(1, 49, dtype=torch.float64).reshape(3, 4, 4)

    result = solve_batched_cg(
        lambda value: coefficients * value,
        diagonal,
        rhs,
        CGConfig(relative_residual_tolerance=1e-12, maximum_iterations=20),
    )

    torch.testing.assert_close(
        result.solution, rhs / coefficients, rtol=1e-12, atol=1e-12
    )
    assert result.diagnostics.converged.shape == (3,)
    assert torch.all(result.diagnostics.converged)
    assert torch.all(result.diagnostics.iterations == 1)
    assert torch.all(result.diagnostics.relative_residuals <= 1e-12)


def test_batched_cg_preserves_multiple_leading_system_dimensions() -> None:
    coefficient = torch.arange(1, 7, dtype=torch.float64).reshape(2, 3, 1, 1)
    diagonal = coefficient.expand(2, 3, 3, 3).clone()
    rhs = torch.ones((2, 3, 3, 3), dtype=torch.float64)

    result = solve_batched_cg(
        lambda value: coefficient * value,
        diagonal,
        rhs,
        CGConfig(relative_residual_tolerance=1e-12, maximum_iterations=20),
    )

    assert result.solution.shape == (2, 3, 3, 3)
    assert result.diagnostics.iterations.shape == (2, 3)
    torch.testing.assert_close(result.solution, rhs / coefficient)


def test_batched_cg_fails_closed_if_any_system_does_not_converge() -> None:
    generator = torch.Generator().manual_seed(42)
    rhs = torch.rand((2, 8, 8), generator=generator, dtype=torch.float64)
    diagonal = torch.full_like(rhs, 3.0)

    with pytest.raises(BatchedCGConvergenceError) as caught:
        solve_batched_cg(
            lambda value: (
                3.0 * value
                - 0.4 * torch.roll(value, shifts=1, dims=-1)
                - 0.4 * torch.roll(value, shifts=-1, dims=-1)
            ),
            diagonal,
            rhs,
            CGConfig(relative_residual_tolerance=1e-20, maximum_iterations=1),
        )

    assert not torch.all(caught.value.diagnostics.converged)


@pytest.mark.parametrize(
    ("diagonal", "rhs"),
    [
        (torch.ones((2, 4, 4)), torch.ones((3, 4, 4))),
        (torch.zeros((2, 4, 4)), torch.ones((2, 4, 4))),
        (torch.ones((2, 4, 4)), torch.full((2, 4, 4), float("nan"))),
    ],
)
def test_batched_cg_rejects_invalid_inputs(
    diagonal: torch.Tensor, rhs: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        solve_batched_cg(lambda value: value, diagonal, rhs, CGConfig())
