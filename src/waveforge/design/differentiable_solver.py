"""Mixed-precision steady solve with an implicit custom adjoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from waveforge.physics.cg import CGConfig, CGDiagnostics, solve_cg
from waveforge.physics.grid import Grid2D
from waveforge.physics.torch_operator import (
    apply_steady_operator,
    operator_diagonal,
)

SolveRole = Literal["forward", "adjoint"]


@dataclass(frozen=True)
class SolveRecord:
    """One forward or adjoint solve retained for fail-closed reporting."""

    role: SolveRole
    scenario_index: int
    iterations: int
    relative_residual: float
    converged: bool
    reason: str
    dtype: str
    device: str


@dataclass
class SolveTrace:
    """Mutable trace shared across a custom-autograd forward/backward pair."""

    records: list[SolveRecord] = field(default_factory=list)

    def append(
        self,
        role: SolveRole,
        scenario_index: int,
        diagnostics: CGDiagnostics,
        tensor: Tensor,
    ) -> None:
        self.records.append(
            SolveRecord(
                role=role,
                scenario_index=scenario_index,
                iterations=diagnostics.iterations,
                relative_residual=diagnostics.relative_residual,
                converged=diagnostics.converged,
                reason=diagnostics.reason,
                dtype=str(tensor.dtype).removeprefix("torch."),
                device=str(tensor.device),
            )
        )


def _validate_physics_inputs(
    conductivity: Tensor,
    sources: Tensor,
    grid: Grid2D,
) -> None:
    if conductivity.dtype is not torch.float64 or sources.dtype is not torch.float64:
        raise ValueError("Gate 2A physical conductivity and sources must be float64")
    if conductivity.device != sources.device:
        raise ValueError("conductivity and sources must share a device")
    if conductivity.ndim != 2 or tuple(conductivity.shape) != grid.shape:
        raise ValueError("conductivity must have shape [ny,nx]")
    if sources.ndim not in (2, 3) or tuple(sources.shape[-2:]) != grid.shape:
        raise ValueError("sources must have shape [ny,nx] or [scenario,ny,nx]")
    if not torch.isfinite(conductivity).all() or not torch.all(conductivity > 0.0):
        raise ValueError("conductivity must be finite and strictly positive")
    if not torch.isfinite(sources).all():
        raise ValueError("sources must be finite")


class _ImplicitSteadySolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        conductivity: Tensor,
        sources: Tensor,
        grid: Grid2D,
        config: CGConfig,
        trace: SolveTrace,
    ) -> Tensor:
        source_batch = sources.unsqueeze(0) if sources.ndim == 2 else sources
        diagonal = operator_diagonal(conductivity, grid)

        def apply(temperature: Tensor) -> Tensor:
            return apply_steady_operator(temperature, conductivity, grid)

        temperatures: list[Tensor] = []
        for scenario_index, source in enumerate(source_batch):
            result = solve_cg(apply, diagonal, source, config)
            temperatures.append(result.solution)
            trace.append("forward", scenario_index, result.diagnostics, result.solution)
        temperature_batch = torch.stack(temperatures)
        if not torch.isfinite(temperature_batch).all():
            raise FloatingPointError("forward temperature contains NaN or Inf")

        ctx.save_for_backward(conductivity, temperature_batch)
        ctx.grid = grid
        ctx.config = config
        ctx.trace = trace
        ctx.sources_were_unbatched = sources.ndim == 2
        return temperature_batch[0] if sources.ndim == 2 else temperature_batch

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        upstream: Tensor,
    ) -> tuple[Tensor, Tensor, None, None, None]:
        conductivity, temperature_batch = ctx.saved_tensors
        upstream_batch = (
            upstream.unsqueeze(0) if ctx.sources_were_unbatched else upstream
        )
        if not torch.isfinite(upstream_batch).all():
            raise FloatingPointError("adjoint RHS contains NaN or Inf")
        diagonal = operator_diagonal(conductivity, ctx.grid)

        def apply(temperature: Tensor) -> Tensor:
            return apply_steady_operator(temperature, conductivity, ctx.grid)

        adjoints: list[Tensor] = []
        for scenario_index, adjoint_rhs in enumerate(upstream_batch):
            result = solve_cg(apply, diagonal, adjoint_rhs, ctx.config)
            adjoints.append(result.solution)
            ctx.trace.append(
                "adjoint",
                scenario_index,
                result.diagnostics,
                result.solution,
            )
        adjoint_batch = torch.stack(adjoints)

        with torch.enable_grad():
            differentiable_conductivity = conductivity.detach().requires_grad_(True)
            action = apply_steady_operator(
                temperature_batch.detach(),
                differentiable_conductivity,
                ctx.grid,
            )
            adjoint_functional = -torch.sum(adjoint_batch.detach() * action)
            (conductivity_gradient,) = torch.autograd.grad(
                adjoint_functional,
                differentiable_conductivity,
            )
        if not torch.isfinite(conductivity_gradient).all():
            raise FloatingPointError(
                "conductivity adjoint gradient contains NaN or Inf"
            )
        source_gradient = (
            adjoint_batch[0] if ctx.sources_were_unbatched else adjoint_batch
        )
        return conductivity_gradient, source_gradient, None, None, None


def solve_steady_implicit(
    conductivity: Tensor,
    sources: Tensor,
    grid: Grid2D,
    *,
    config: CGConfig | None = None,
    trace: SolveTrace | None = None,
) -> Tensor:
    """Solve Gate 2A physics and attach the exact implicit-adjoint backward."""
    _validate_physics_inputs(conductivity, sources, grid)
    solve_config = config or CGConfig()
    solve_trace = trace if trace is not None else SolveTrace()
    return _ImplicitSteadySolve.apply(
        conductivity,
        sources,
        grid,
        solve_config,
        solve_trace,
    )
