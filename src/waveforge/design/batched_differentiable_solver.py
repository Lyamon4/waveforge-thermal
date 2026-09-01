"""Vectorized mixed-precision steady physics with an implicit batched adjoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from waveforge.physics.batched_cg import BatchedCGDiagnostics, solve_batched_cg
from waveforge.physics.cg import CGConfig
from waveforge.physics.grid import Grid2D

SolveRole = Literal["forward", "adjoint"]
_HARMONIC_EPSILON = 1.0e-12


@dataclass(frozen=True)
class BatchedSolveRecord:
    role: SolveRole
    batch_index: int
    scenario_index: int
    iterations: int
    relative_residual: float
    converged: bool
    dtype: str
    device: str


def _diagnostics_to_host(
    diagnostics: BatchedCGDiagnostics,
) -> tuple[list[list[int]], list[list[float]], list[list[bool]]]:
    iterations = diagnostics.iterations.detach().cpu().tolist()
    residuals = diagnostics.relative_residuals.detach().cpu().tolist()
    converged = diagnostics.converged.detach().cpu().tolist()
    return iterations, residuals, converged


@dataclass
class BatchedSolveTrace:
    records: list[BatchedSolveRecord] = field(default_factory=list)

    def append(
        self,
        role: SolveRole,
        diagnostics: BatchedCGDiagnostics,
        tensor: Tensor,
    ) -> None:
        iterations, residuals, converged = _diagnostics_to_host(diagnostics)
        for batch_index in range(len(iterations)):
            for scenario_index in range(len(iterations[batch_index])):
                self.records.append(
                    BatchedSolveRecord(
                        role=role,
                        batch_index=batch_index,
                        scenario_index=scenario_index,
                        iterations=int(iterations[batch_index][scenario_index]),
                        relative_residual=float(residuals[batch_index][scenario_index]),
                        converged=bool(converged[batch_index][scenario_index]),
                        dtype=str(tensor.dtype).removeprefix("torch."),
                        device=str(tensor.device),
                    )
                )


@dataclass(frozen=True)
class _BatchedOperatorCoefficients:
    x_conductance: Tensor
    y_conductance: Tensor
    bottom_conductance: Tensor


def _harmonic(first: Tensor, second: Tensor) -> Tensor:
    return 2.0 * first * second / (first + second + _HARMONIC_EPSILON)


def _assemble_batched_operator(
    conductivity: Tensor,
    grid: Grid2D,
) -> _BatchedOperatorCoefficients:
    return _BatchedOperatorCoefficients(
        x_conductance=(
            _harmonic(conductivity[:, :, :-1], conductivity[:, :, 1:]) / grid.dx**2
        ),
        y_conductance=(
            _harmonic(conductivity[:, :-1, :], conductivity[:, 1:, :]) / grid.dy**2
        ),
        bottom_conductance=2.0 * conductivity[:, 0, :] / grid.dy**2,
    )


def _apply_assembled_batched_operator(
    temperature: Tensor,
    coefficients: _BatchedOperatorCoefficients,
) -> Tensor:
    result = torch.zeros_like(temperature)
    x_conductance = coefficients.x_conductance[:, None, :, :]
    x_jump = temperature[:, :, :, :-1] - temperature[:, :, :, 1:]
    result[:, :, :, :-1] += x_conductance * x_jump
    result[:, :, :, 1:] -= x_conductance * x_jump

    y_conductance = coefficients.y_conductance[:, None, :, :]
    y_jump = temperature[:, :, :-1, :] - temperature[:, :, 1:, :]
    result[:, :, :-1, :] += y_conductance * y_jump
    result[:, :, 1:, :] -= y_conductance * y_jump

    bottom = coefficients.bottom_conductance[:, None, :]
    result[:, :, 0, :] += bottom * temperature[:, :, 0, :]
    return result


def _apply_batched_operator(
    temperature: Tensor,
    conductivity: Tensor,
    grid: Grid2D,
) -> Tensor:
    return _apply_assembled_batched_operator(
        temperature,
        _assemble_batched_operator(conductivity, grid),
    )


def _diagonal_from_coefficients(
    coefficients: _BatchedOperatorCoefficients,
) -> Tensor:
    x_conductance = coefficients.x_conductance
    y_conductance = coefficients.y_conductance
    diagonal = torch.zeros(
        (
            x_conductance.shape[0],
            y_conductance.shape[1] + 1,
            x_conductance.shape[2] + 1,
        ),
        dtype=x_conductance.dtype,
        device=x_conductance.device,
    )
    diagonal[:, :, :-1] += x_conductance
    diagonal[:, :, 1:] += x_conductance
    diagonal[:, :-1, :] += y_conductance
    diagonal[:, 1:, :] += y_conductance
    diagonal[:, 0, :] += coefficients.bottom_conductance
    return diagonal


def _batched_diagonal(conductivity: Tensor, grid: Grid2D) -> Tensor:
    return _diagonal_from_coefficients(_assemble_batched_operator(conductivity, grid))


def _validate_inputs(conductivity: Tensor, sources: Tensor, grid: Grid2D) -> None:
    if conductivity.dtype is not torch.float64 or sources.dtype is not torch.float64:
        raise ValueError("batched physics conductivity and sources must be float64")
    if conductivity.device != sources.device:
        raise ValueError("conductivity and sources must share a device")
    if conductivity.ndim != 3 or tuple(conductivity.shape[-2:]) != grid.shape:
        raise ValueError("conductivity must have shape [batch,ny,nx]")
    if (
        sources.ndim != 4
        or sources.shape[0] != conductivity.shape[0]
        or tuple(sources.shape[-2:]) != grid.shape
    ):
        raise ValueError("sources must have shape [batch,scenario,ny,nx]")
    if not torch.isfinite(conductivity).all() or not torch.all(conductivity > 0.0):
        raise ValueError("conductivity must be finite and strictly positive")
    if not torch.isfinite(sources).all():
        raise ValueError("sources must be finite")


class _ImplicitBatchedSteadySolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        conductivity: Tensor,
        sources: Tensor,
        grid: Grid2D,
        config: CGConfig,
        trace: BatchedSolveTrace,
    ) -> Tensor:
        coefficients = _assemble_batched_operator(conductivity, grid)
        diagonal = _diagonal_from_coefficients(coefficients)[:, None].expand_as(sources)

        def apply(temperature: Tensor) -> Tensor:
            return _apply_assembled_batched_operator(temperature, coefficients)

        result = solve_batched_cg(apply, diagonal, sources, config)
        trace.append("forward", result.diagnostics, result.solution)
        if not torch.isfinite(result.solution).all():
            raise FloatingPointError("batched forward temperature contains NaN or Inf")
        ctx.save_for_backward(conductivity, result.solution)
        ctx.grid = grid
        ctx.config = config
        ctx.trace = trace
        return result.solution

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        upstream: Tensor,
    ) -> tuple[Tensor, Tensor, None, None, None]:
        conductivity, temperature = ctx.saved_tensors
        if not torch.isfinite(upstream).all():
            raise FloatingPointError("batched adjoint RHS contains NaN or Inf")
        coefficients = _assemble_batched_operator(conductivity, ctx.grid)
        diagonal = _diagonal_from_coefficients(coefficients)[:, None].expand_as(
            upstream
        )

        def apply(value: Tensor) -> Tensor:
            return _apply_assembled_batched_operator(value, coefficients)

        adjoint = solve_batched_cg(apply, diagonal, upstream, ctx.config)
        ctx.trace.append("adjoint", adjoint.diagnostics, adjoint.solution)

        with torch.enable_grad():
            differentiable_conductivity = conductivity.detach().requires_grad_(True)
            differentiable_coefficients = _assemble_batched_operator(
                differentiable_conductivity,
                ctx.grid,
            )
            action = _apply_assembled_batched_operator(
                temperature.detach(),
                differentiable_coefficients,
            )
            functional = -torch.sum(adjoint.solution.detach() * action)
            (conductivity_gradient,) = torch.autograd.grad(
                functional, differentiable_conductivity
            )
        if not torch.isfinite(conductivity_gradient).all():
            raise FloatingPointError(
                "batched conductivity gradient contains NaN or Inf"
            )
        return conductivity_gradient, adjoint.solution, None, None, None


def solve_steady_implicit_batched(
    conductivity: Tensor,
    sources: Tensor,
    grid: Grid2D,
    *,
    config: CGConfig | None = None,
    trace: BatchedSolveTrace | None = None,
) -> Tensor:
    """Solve `[batch,scenario,ny,nx]` systems with one vectorized physics path."""
    _validate_inputs(conductivity, sources, grid)
    solve_config = config or CGConfig()
    solve_trace = trace if trace is not None else BatchedSolveTrace()
    return _ImplicitBatchedSteadySolve.apply(
        conductivity, sources, grid, solve_config, solve_trace
    )
