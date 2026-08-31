"""Vectorized fail-closed CG for independent leading-dimension systems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from waveforge.physics.cg import CGConfig


@dataclass(frozen=True)
class BatchedCGDiagnostics:
    iterations: Tensor
    relative_residuals: Tensor
    converged: Tensor


@dataclass(frozen=True)
class BatchedCGResult:
    solution: Tensor
    diagnostics: BatchedCGDiagnostics


class BatchedCGConvergenceError(RuntimeError):
    def __init__(self, diagnostics: BatchedCGDiagnostics) -> None:
        self.diagnostics = diagnostics
        failed = int(torch.count_nonzero(~diagnostics.converged).item())
        super().__init__(f"batched CG failed for {failed} independent systems")


def _dot(first: Tensor, second: Tensor) -> Tensor:
    return torch.sum(first * second, dim=(-2, -1))


def _expand(values: Tensor) -> Tensor:
    return values[..., None, None]


def solve_batched_cg(
    apply: Callable[[Tensor], Tensor],
    diagonal: Tensor,
    rhs: Tensor,
    config: CGConfig,
) -> BatchedCGResult:
    """Solve all independent RHS systems in one vectorized CG iteration loop."""
    if diagonal.shape != rhs.shape or rhs.ndim < 3:
        raise ValueError("diagonal and rhs must share [...,ny,nx] shape")
    if diagonal.device != rhs.device or diagonal.dtype != rhs.dtype:
        raise ValueError("diagonal and rhs must share device and dtype")
    if not torch.isfinite(diagonal).all() or not torch.all(diagonal > 0.0):
        raise ValueError("Jacobi diagonal must be finite and strictly positive")
    if not torch.isfinite(rhs).all():
        raise ValueError("rhs must be finite")

    leading_shape = rhs.shape[:-2]
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    denominator = torch.clamp(torch.linalg.vector_norm(rhs, dim=(-2, -1)), min=1.0e-12)
    relative = torch.linalg.vector_norm(residual, dim=(-2, -1)) / denominator
    converged = relative <= config.relative_residual_tolerance
    iterations = torch.zeros(leading_shape, dtype=torch.int64, device=rhs.device)
    if bool(torch.all(converged).item()):
        return BatchedCGResult(
            solution=solution,
            diagnostics=BatchedCGDiagnostics(iterations, relative, converged),
        )

    preconditioned = residual / diagonal
    direction = torch.where(_expand(converged), torch.zeros_like(rhs), preconditioned)
    residual_preconditioned = _dot(residual, preconditioned)

    for iteration in range(1, config.maximum_iterations + 1):
        active = ~converged
        applied_direction = apply(direction)
        if applied_direction.shape != rhs.shape:
            raise ValueError("batched operator returned the wrong shape")
        curvature = _dot(direction, applied_direction)
        invalid_curvature = active & (~torch.isfinite(curvature) | (curvature <= 0.0))
        if bool(torch.any(invalid_curvature).item()):
            diagnostics = BatchedCGDiagnostics(iterations, relative, converged)
            raise BatchedCGConvergenceError(diagnostics)

        safe_curvature = torch.where(active, curvature, torch.ones_like(curvature))
        step = torch.where(
            active,
            residual_preconditioned / safe_curvature,
            torch.zeros_like(curvature),
        )
        solution = solution + _expand(step) * direction
        residual = residual - _expand(step) * applied_direction
        if not torch.isfinite(solution).all() or not torch.isfinite(residual).all():
            diagnostics = BatchedCGDiagnostics(iterations, relative, converged)
            raise BatchedCGConvergenceError(diagnostics)

        relative = torch.linalg.vector_norm(residual, dim=(-2, -1)) / denominator
        candidate = active & (relative <= config.relative_residual_tolerance)
        restart = torch.zeros_like(candidate)
        if bool(torch.any(candidate).item()):
            explicit_residual = rhs - apply(solution)
            explicit_relative = (
                torch.linalg.vector_norm(explicit_residual, dim=(-2, -1)) / denominator
            )
            success = candidate & (
                explicit_relative <= config.relative_residual_tolerance
            )
            restart = candidate & ~success
            residual = torch.where(_expand(candidate), explicit_residual, residual)
            relative = torch.where(candidate, explicit_relative, relative)
            iterations = torch.where(
                success, torch.full_like(iterations, iteration), iterations
            )
            converged = converged | success

        remaining = ~converged
        if bool(torch.all(converged).item()):
            break
        preconditioned = residual / diagonal
        updated = _dot(residual, preconditioned)
        invalid_norm = remaining & (~torch.isfinite(updated) | (updated <= 0.0))
        if bool(torch.any(invalid_norm).item()):
            diagnostics = BatchedCGDiagnostics(iterations, relative, converged)
            raise BatchedCGConvergenceError(diagnostics)
        safe_previous = torch.where(
            remaining,
            residual_preconditioned,
            torch.ones_like(residual_preconditioned),
        )
        beta = torch.where(
            remaining & ~restart,
            updated / safe_previous,
            torch.zeros_like(updated),
        )
        next_direction = preconditioned + _expand(beta) * direction
        direction = torch.where(
            _expand(remaining), next_direction, torch.zeros_like(direction)
        )
        residual_preconditioned = torch.where(
            remaining, updated, torch.ones_like(updated)
        )

    explicit_residual = rhs - apply(solution)
    relative = torch.linalg.vector_norm(explicit_residual, dim=(-2, -1)) / denominator
    final_converged = relative <= config.relative_residual_tolerance
    iterations = torch.where(
        final_converged & (iterations == 0),
        torch.full_like(iterations, config.maximum_iterations),
        iterations,
    )
    diagnostics = BatchedCGDiagnostics(iterations, relative, final_converged)
    if not bool(torch.all(final_converged).item()):
        raise BatchedCGConvergenceError(diagnostics)
    return BatchedCGResult(solution=solution, diagnostics=diagnostics)
