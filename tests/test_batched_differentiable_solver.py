from __future__ import annotations

import pytest
import torch

from waveforge.design.batched_differentiable_solver import (
    BatchedSolveTrace,
    solve_steady_implicit_batched,
)
from waveforge.design.differentiable_solver import SolveTrace, solve_steady_implicit
from waveforge.physics.cg import CGConfig
from waveforge.physics.grid import Grid2D


def _problem(batch: int, grid_size: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260922 + batch)
    conductivity = 1.0 + 19.0 * torch.rand(
        (batch, grid_size, grid_size), generator=generator, dtype=torch.float64
    )
    sources = torch.rand(
        (batch, 3, grid_size, grid_size), generator=generator, dtype=torch.float64
    )
    return conductivity, sources


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_batched_solver_matches_sequential_temperatures(batch: int) -> None:
    conductivity, sources = _problem(batch)
    grid = Grid2D(nx=12, ny=12)
    config = CGConfig(relative_residual_tolerance=1e-10, maximum_iterations=2000)
    batched_trace = BatchedSolveTrace()

    batched = solve_steady_implicit_batched(
        conductivity,
        sources,
        grid,
        config=config,
        trace=batched_trace,
    )
    sequential = torch.stack(
        [
            solve_steady_implicit(
                conductivity[index],
                sources[index],
                grid,
                config=config,
                trace=SolveTrace(),
            )
            for index in range(batch)
        ]
    )

    torch.testing.assert_close(batched, sequential, rtol=1e-8, atol=1e-9)
    assert len(batched_trace.records) == batch * 3
    assert all(record.role == "forward" for record in batched_trace.records)
    assert all(record.relative_residual <= 1e-10 for record in batched_trace.records)


def test_batched_solver_matches_sequential_implicit_gradients() -> None:
    conductivity, sources = _problem(2, grid_size=10)
    grid = Grid2D(nx=10, ny=10)
    config = CGConfig(relative_residual_tolerance=1e-10, maximum_iterations=2000)
    batched_k = conductivity.clone().requires_grad_(True)
    sequential_k = conductivity.clone().requires_grad_(True)
    weights = torch.linspace(0.1, 1.0, sources.numel(), dtype=torch.float64).reshape(
        sources.shape
    )
    trace = BatchedSolveTrace()

    batched_temperature = solve_steady_implicit_batched(
        batched_k, sources, grid, config=config, trace=trace
    )
    batched_loss = torch.sum(batched_temperature * weights)
    (batched_gradient,) = torch.autograd.grad(batched_loss, batched_k)

    sequential_temperature = torch.stack(
        [
            solve_steady_implicit(
                sequential_k[index],
                sources[index],
                grid,
                config=config,
                trace=SolveTrace(),
            )
            for index in range(2)
        ]
    )
    sequential_loss = torch.sum(sequential_temperature * weights)
    (sequential_gradient,) = torch.autograd.grad(sequential_loss, sequential_k)

    torch.testing.assert_close(
        batched_temperature, sequential_temperature, rtol=1e-8, atol=1e-9
    )
    relative_loss_error = (
        torch.abs(batched_loss - sequential_loss) / torch.abs(sequential_loss)
    ).detach()
    assert float(relative_loss_error) <= 1e-8
    relative_l2 = torch.linalg.vector_norm(
        batched_gradient - sequential_gradient
    ) / torch.linalg.vector_norm(sequential_gradient)
    cosine = torch.nn.functional.cosine_similarity(
        batched_gradient.reshape(1, -1), sequential_gradient.reshape(1, -1)
    )
    assert float(relative_l2) <= 1e-6
    assert float(cosine) >= 0.9999999
    assert len(trace.records) == 12
    assert sum(record.role == "forward" for record in trace.records) == 6
    assert sum(record.role == "adjoint" for record in trace.records) == 6


@pytest.mark.parametrize(
    ("conductivity", "sources"),
    [
        (
            torch.ones((64, 64), dtype=torch.float64),
            torch.ones((1, 3, 64, 64), dtype=torch.float64),
        ),
        (
            torch.ones((2, 64, 64), dtype=torch.float32),
            torch.ones((2, 3, 64, 64), dtype=torch.float64),
        ),
        (
            torch.ones((2, 64, 64), dtype=torch.float64),
            torch.ones((3, 3, 64, 64), dtype=torch.float64),
        ),
    ],
)
def test_batched_solver_rejects_invalid_shapes_or_precision(
    conductivity: torch.Tensor, sources: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        solve_steady_implicit_batched(conductivity, sources, Grid2D(nx=64, ny=64))
