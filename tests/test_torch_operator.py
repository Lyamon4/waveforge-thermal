"""Tests for the independent PyTorch finite-volume operator."""

import numpy as np
import pytest
import torch

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import assemble_steady_system
from waveforge.physics.torch_operator import (
    apply_steady_operator,
    operator_diagonal,
)


def test_uniform_square_grid_has_hand_derived_diagonal() -> None:
    """Dropping the half-cell bottom conductance must break this test."""
    grid = Grid2D(nx=3, ny=3)
    conductivity = torch.full(grid.shape, 2.0, dtype=torch.float64)

    diagonal = operator_diagonal(conductivity, grid)

    scale = 2.0 / grid.dx**2
    assert diagonal[0, 1].item() == pytest.approx(5.0 * scale)
    assert diagonal[1, 1].item() == pytest.approx(4.0 * scale)


def test_operator_has_negative_neighbor_coupling_and_interior_nullspace() -> None:
    """A wrong flux sign or spurious interior source must break this test."""
    grid = Grid2D(nx=3, ny=3)
    conductivity = torch.ones(grid.shape, dtype=torch.float64)
    impulse = torch.zeros(grid.shape, dtype=torch.float64)
    impulse[1, 1] = 1.0

    applied_impulse = apply_steady_operator(impulse, conductivity, grid)
    applied_constant = apply_steady_operator(
        torch.ones_like(impulse), conductivity, grid
    )

    assert applied_impulse[1, 0].item() == pytest.approx(-1.0 / grid.dx**2)
    assert applied_impulse[0, 1].item() == pytest.approx(-1.0 / grid.dy**2)
    assert applied_constant[1, 1].item() == pytest.approx(0.0, abs=1e-14)
    assert applied_constant[2, 1].item() == pytest.approx(0.0, abs=1e-14)


def test_operator_and_diagonal_match_independent_scipy_assembly() -> None:
    """Changing either public numerical boundary must reveal disagreement."""
    grid = Grid2D(nx=8, ny=7)
    rng = np.random.default_rng(4117)
    conductivity_np = rng.uniform(1.0, 20.0, size=grid.shape)
    temperature_np = rng.normal(size=grid.shape)
    system = assemble_steady_system(
        grid,
        conductivity_np,
        np.zeros(grid.shape),
        BoundaryConditions.production(),
    )
    conductivity = torch.from_numpy(conductivity_np)
    temperature = torch.from_numpy(temperature_np)

    actual = apply_steady_operator(temperature, conductivity, grid).numpy()
    expected = (system.matrix @ temperature_np.ravel()).reshape(grid.shape)
    actual_diagonal = operator_diagonal(conductivity, grid).numpy()

    relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_l2 <= 1e-12
    assert np.max(np.abs(actual_diagonal.ravel() - system.matrix.diagonal())) <= 1e-11


def test_batched_operator_matches_individual_cpu_results() -> None:
    """Accidental broadcasting across scenarios must break this test."""
    grid = Grid2D(nx=9, ny=8)
    generator = torch.Generator().manual_seed(6123)
    conductivity = 1.0 + 19.0 * torch.rand(
        grid.shape, generator=generator, dtype=torch.float64
    )
    temperatures = torch.randn(
        (3, *grid.shape), generator=generator, dtype=torch.float64
    )

    batched = apply_steady_operator(temperatures, conductivity, grid)
    individual = torch.stack(
        [apply_steady_operator(field, conductivity, grid) for field in temperatures]
    )

    torch.testing.assert_close(batched, individual, rtol=0.0, atol=0.0)
    assert torch.isfinite(batched).all()


def test_batched_operator_is_finite_on_locked_cuda_environment() -> None:
    """A hidden CPU fallback must break this environment-specific gate."""
    assert torch.cuda.is_available(), "Gate 2A locked environment requires CUDA"
    grid = Grid2D(nx=9, ny=8)
    generator = torch.Generator(device="cuda").manual_seed(6124)
    conductivity = 1.0 + 19.0 * torch.rand(
        grid.shape, generator=generator, device="cuda"
    )
    temperatures = torch.randn((3, *grid.shape), generator=generator, device="cuda")

    result = apply_steady_operator(temperatures, conductivity, grid)

    assert result.device.type == "cuda"
    assert torch.isfinite(result).all()
