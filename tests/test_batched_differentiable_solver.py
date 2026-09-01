from __future__ import annotations

import pytest
import torch

import waveforge.design.batched_differentiable_solver as batched_solver_module
from waveforge.design.batched_differentiable_solver import (
    BatchedSolveTrace,
    _apply_assembled_batched_operator,
    _apply_batched_operator,
    _assemble_batched_operator,
    _diagnostics_to_host,
    solve_steady_implicit_batched,
)
from waveforge.design.differentiable_solver import SolveTrace, solve_steady_implicit
from waveforge.physics.batched_cg import BatchedCGDiagnostics
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


def test_assembled_batched_operator_matches_recomputed_operator_and_gradient() -> None:
    grid = Grid2D(nx=12, ny=12)
    generator = torch.Generator().manual_seed(2026092401)
    temperature = torch.rand((3, 2, 12, 12), generator=generator, dtype=torch.float64)
    conductivity = 1.0 + 19.0 * torch.rand(
        (3, 12, 12), generator=generator, dtype=torch.float64
    )
    original_temperature = temperature.clone().requires_grad_(True)
    original_conductivity = conductivity.clone().requires_grad_(True)
    assembled_temperature = temperature.clone().requires_grad_(True)
    assembled_conductivity = conductivity.clone().requires_grad_(True)

    original = _apply_batched_operator(
        original_temperature,
        original_conductivity,
        grid,
    )
    coefficients = _assemble_batched_operator(assembled_conductivity, grid)
    assembled = _apply_assembled_batched_operator(
        assembled_temperature,
        coefficients,
    )
    original_gradients = torch.autograd.grad(
        original.square().sum(),
        (original_temperature, original_conductivity),
    )
    assembled_gradients = torch.autograd.grad(
        assembled.square().sum(),
        (assembled_temperature, assembled_conductivity),
    )

    torch.testing.assert_close(assembled, original, rtol=0.0, atol=0.0)
    for actual, expected in zip(assembled_gradients, original_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_batched_forward_assembles_operator_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conductivity, sources = _problem(2, grid_size=8)
    grid = Grid2D(nx=8, ny=8)
    original = batched_solver_module._assemble_batched_operator
    calls = 0

    def counted_assembly(
        value: torch.Tensor,
        target_grid: Grid2D,
    ) -> object:
        nonlocal calls
        calls += 1
        return original(value, target_grid)

    monkeypatch.setattr(
        batched_solver_module,
        "_assemble_batched_operator",
        counted_assembly,
    )

    solve_steady_implicit_batched(
        conductivity,
        sources,
        grid,
        config=CGConfig(relative_residual_tolerance=1.0e-8),
    )

    assert calls == 1


def test_batched_diagnostics_transfer_preserves_complete_row_major_values() -> None:
    diagnostics = BatchedCGDiagnostics(
        iterations=torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int64),
        relative_residuals=torch.tensor(
            [[1.0e-7, 2.0e-7, 3.0e-7], [4.0e-7, 5.0e-7, 6.0e-7]],
            dtype=torch.float64,
        ),
        converged=torch.tensor([[True, False, True], [False, True, False]]),
    )

    iterations, residuals, converged = _diagnostics_to_host(diagnostics)

    assert iterations == [[11, 12, 13], [21, 22, 23]]
    assert residuals == [
        [1.0e-7, 2.0e-7, 3.0e-7],
        [4.0e-7, 5.0e-7, 6.0e-7],
    ]
    assert converged == [[True, False, True], [False, True, False]]


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
