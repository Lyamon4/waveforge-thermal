"""Tests for fail-closed Jacobi-preconditioned conjugate gradients."""

import numpy as np
import pytest
import torch
from torch import Tensor

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.cg import (
    CGConfig,
    CGConvergenceError,
    solve_cg,
)
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady
from waveforge.physics.torch_operator import (
    apply_steady_operator,
    operator_diagonal,
)


def test_jacobi_cg_solves_diagonal_spd_system_in_one_iteration() -> None:
    """A non-Jacobi update or nonzero initial guess must break this test."""
    diagonal = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    rhs = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)

    result = solve_cg(
        lambda value: diagonal * value,
        diagonal,
        rhs,
        CGConfig(relative_residual_tolerance=1e-12, maximum_iterations=10),
    )

    torch.testing.assert_close(
        result.solution,
        torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.diagnostics.iterations == 1
    assert result.diagnostics.converged
    assert result.diagnostics.relative_residual <= 1e-12
    assert result.diagnostics.reason == "CONVERGED"


def test_cg_raises_instead_of_returning_unconverged_iterate() -> None:
    """Returning a last iterate after the iteration cap must break this test."""
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    diagonal = torch.diagonal(matrix)
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)

    with pytest.raises(CGConvergenceError) as captured:
        solve_cg(
            lambda value: matrix @ value,
            diagonal,
            rhs,
            CGConfig(relative_residual_tolerance=1e-14, maximum_iterations=1),
        )

    assert captured.value.diagnostics.iterations == 1
    assert not captured.value.diagnostics.converged
    assert captured.value.diagnostics.relative_residual > 1e-14
    assert captured.value.diagnostics.reason == "MAXIMUM_ITERATIONS"


def test_locked_cg_defaults_define_forward_and_adjoint_policy() -> None:
    """Drifting from the pre-registered solver policy must break this test."""
    config = CGConfig()

    assert config.relative_residual_tolerance == 1e-6
    assert config.maximum_iterations == 2000
    assert config.initial_guess == "zeros"
    assert config.preconditioner == "Jacobi"


def test_cg_reports_and_enforces_explicit_residual_not_recursive_estimate() -> None:
    """False convergence from float32 recursive-residual drift must fail."""
    size = 20
    generator = torch.Generator().manual_seed(12)
    orthogonal, _ = torch.linalg.qr(
        torch.randn(size, size, generator=generator, dtype=torch.float32)
    )
    eigenvalues = torch.logspace(0.0, 4.0, size, dtype=torch.float32)
    matrix = orthogonal @ torch.diag(eigenvalues) @ orthogonal.T
    rhs = torch.randn(
        size,
        generator=torch.Generator().manual_seed(13),
        dtype=torch.float32,
    )
    tolerance = 1.0e-4

    result = solve_cg(
        lambda value: matrix @ value,
        torch.diagonal(matrix),
        rhs,
        CGConfig(
            relative_residual_tolerance=tolerance,
            maximum_iterations=2000,
        ),
    )
    explicit_residual = float(
        torch.linalg.vector_norm(rhs - matrix @ result.solution)
        / torch.linalg.vector_norm(rhs)
    )

    assert explicit_residual <= tolerance
    assert result.diagnostics.relative_residual == pytest.approx(
        explicit_residual,
        abs=1e-10,
    )


def _solve_torch_reference_case(
    device: torch.device,
) -> tuple[Tensor, float, float, float]:
    grid = Grid2D(nx=32, ny=32)
    rng = np.random.default_rng(8102)
    conductivity_np = rng.uniform(1.0, 20.0, size=grid.shape)
    source_np = area_overlap_rectangular_source(
        grid,
        bounds=(0.40, 0.60, 0.62, 0.82),
        power=1.0,
    )
    expected = solve_steady(
        grid,
        conductivity_np,
        source_np,
        BoundaryConditions.production(),
    ).temperature
    dtype = torch.float64
    conductivity = torch.as_tensor(conductivity_np, dtype=dtype, device=device)
    rhs = torch.as_tensor(source_np, dtype=dtype, device=device)

    def apply(value: Tensor) -> Tensor:
        return apply_steady_operator(value, conductivity, grid)

    result = solve_cg(apply, operator_diagonal(conductivity, grid), rhs, CGConfig())
    actual = result.solution.detach().cpu().to(torch.float64).numpy()
    relative_l2 = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    explicit_residual = float(
        torch.linalg.vector_norm(rhs - apply(result.solution))
        / torch.linalg.vector_norm(rhs)
    )
    return (
        result.solution,
        relative_l2,
        explicit_residual,
        result.diagnostics.relative_residual,
    )


def test_cpu_cg_temperature_agrees_with_scipy() -> None:
    """A converged residual with the wrong physical solution must break this test."""
    solution, relative_l2, explicit_residual, reported_residual = (
        _solve_torch_reference_case(torch.device("cpu"))
    )

    assert solution.device.type == "cpu"
    assert relative_l2 <= 5e-5
    assert explicit_residual <= 1e-6
    assert reported_residual == pytest.approx(explicit_residual, abs=1e-14)


def test_cuda_float64_cg_temperature_and_residual_agree_with_scipy() -> None:
    """CUDA arithmetic or device fallback errors must break this test."""
    assert torch.cuda.is_available(), "Gate 2A locked environment requires CUDA"

    solution, relative_l2, explicit_residual, reported_residual = (
        _solve_torch_reference_case(torch.device("cuda"))
    )

    assert solution.device.type == "cuda"
    assert solution.dtype is torch.float64
    assert relative_l2 <= 5e-5
    assert explicit_residual <= 1e-6
    assert reported_residual == pytest.approx(explicit_residual, abs=1e-14)
