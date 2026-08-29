"""Tests for the mixed-precision implicit steady solver and adjoint."""

import numpy as np
import pytest
import torch

from waveforge.design.differentiable_solver import SolveTrace, solve_steady_implicit
from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady


def _three_sources(grid: Grid2D) -> np.ndarray:
    bounds = (
        (0.40, 0.60, 0.62, 0.82),
        (0.18, 0.38, 0.62, 0.82),
        (0.62, 0.82, 0.62, 0.82),
    )
    return np.stack(
        [area_overlap_rectangular_source(grid, item, 1.0) for item in bounds]
    )


def test_implicit_forward_fields_match_independent_scipy_solver() -> None:
    """A self-consistent but wrong matrix-free solve must fail at this boundary."""
    grid = Grid2D(nx=16, ny=16)
    rng = np.random.default_rng(8211)
    conductivity_np = rng.uniform(1.0, 20.0, size=grid.shape)
    sources_np = _three_sources(grid)
    expected = np.stack(
        [
            solve_steady(
                grid,
                conductivity_np,
                source,
                BoundaryConditions.production(),
            ).temperature
            for source in sources_np
        ]
    )
    conductivity = torch.from_numpy(conductivity_np)
    sources = torch.from_numpy(sources_np)
    trace = SolveTrace()

    actual = solve_steady_implicit(conductivity, sources, grid, trace=trace)

    relative_l2 = np.linalg.norm(actual.detach().numpy() - expected) / np.linalg.norm(
        expected
    )
    assert relative_l2 <= 5e-5
    assert actual.dtype is torch.float64
    assert torch.isfinite(actual).all()
    assert len(trace.records) == 3
    assert all(record.role == "forward" for record in trace.records)
    assert all(record.relative_residual <= 1e-6 for record in trace.records)


def test_implicit_conductivity_gradient_matches_three_finite_differences() -> None:
    """A sign error or missing operator derivative in the adjoint must fail."""
    grid = Grid2D(nx=8, ny=8)
    generator = torch.Generator().manual_seed(8212)
    conductivity = 1.0 + 3.0 * torch.rand(
        grid.shape,
        generator=generator,
        dtype=torch.float64,
    )
    conductivity.requires_grad_(True)
    source_np = area_overlap_rectangular_source(
        grid,
        bounds=(0.30, 0.60, 0.55, 0.85),
        power=1.0,
    )
    source = torch.from_numpy(source_np)
    weight = torch.randn(grid.shape, generator=generator, dtype=torch.float64)
    trace = SolveTrace()

    temperature = solve_steady_implicit(conductivity, source, grid, trace=trace)
    objective = torch.sum(temperature * weight)
    (gradient,) = torch.autograd.grad(objective, conductivity)

    step = 1e-4
    for seed in (8301, 8302, 8303):
        direction = torch.randn(
            grid.shape,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float64,
        )
        direction = direction / torch.linalg.vector_norm(direction)
        with torch.no_grad():
            plus = solve_steady_implicit(
                conductivity + step * direction,
                source,
                grid,
            )
            minus = solve_steady_implicit(
                conductivity - step * direction,
                source,
                grid,
            )
        finite_difference = torch.sum((plus - minus) * weight).item() / (2.0 * step)
        automatic = torch.sum(gradient * direction).item()
        relative_error = abs(automatic - finite_difference) / max(
            abs(automatic),
            abs(finite_difference),
            1e-12,
        )
        assert relative_error <= 1e-5

    assert len(trace.records) == 2
    assert trace.records[0].role == "forward"
    assert trace.records[1].role == "adjoint"
    assert trace.records[1].relative_residual <= 1e-6


def test_cuda_mixed_precision_returns_gradient_to_float32_design() -> None:
    """Casting after k(D), solving in float32, or returning float64 must fail."""
    assert torch.cuda.is_available(), "Gate 2A locked environment requires CUDA"
    grid = Grid2D(nx=8, ny=8)
    design = torch.full(
        grid.shape,
        0.25,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    conductivity = 1.0 + 19.0 * design.to(torch.float64) ** 3
    source = torch.from_numpy(
        area_overlap_rectangular_source(
            grid,
            bounds=(0.30, 0.60, 0.55, 0.85),
            power=1.0,
        )
    ).to(device="cuda", dtype=torch.float64)
    trace = SolveTrace()

    temperature = solve_steady_implicit(conductivity, source, grid, trace=trace)
    temperature.sum().backward()

    assert conductivity.dtype is torch.float64
    assert temperature.dtype is torch.float64
    assert temperature.device.type == "cuda"
    assert design.grad is not None
    assert design.grad.dtype is torch.float32
    assert torch.isfinite(design.grad).all()
    assert {record.dtype for record in trace.records} == {"float64"}
    assert all(record.relative_residual <= 1e-6 for record in trace.records)


def test_implicit_solver_rejects_float32_physics_input() -> None:
    """A silent float32 physical solve must fail the amended protocol."""
    grid = Grid2D(nx=8, ny=8)
    conductivity = torch.ones(grid.shape, dtype=torch.float32)
    source = torch.ones(grid.shape, dtype=torch.float32)

    with pytest.raises(ValueError, match="float64"):
        solve_steady_implicit(conductivity, source, grid)
