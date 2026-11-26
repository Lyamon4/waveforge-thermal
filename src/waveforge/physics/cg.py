"""Fail-closed Jacobi-preconditioned conjugate-gradient solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class CGConfig:
    """Locked Gate 2A conjugate-gradient policy."""

    relative_residual_tolerance: float = 1.0e-6
    maximum_iterations: int = 2000
    initial_guess: Literal["zeros"] = "zeros"
    preconditioner: Literal["Jacobi"] = "Jacobi"

    def __post_init__(self) -> None:
        if self.relative_residual_tolerance <= 0.0:
            raise ValueError("relative residual tolerance must be positive")
        if self.maximum_iterations < 1:
            raise ValueError("maximum iterations must be positive")
        if self.initial_guess != "zeros":
            raise ValueError("Gate 2A CG requires a zero initial guess")
        if self.preconditioner != "Jacobi":
            raise ValueError("Gate 2A CG requires Jacobi preconditioning")


@dataclass(frozen=True)
class CGDiagnostics:
    """Numerical evidence retained for every linear solve."""

    iterations: int
    relative_residual: float
    converged: bool
    reason: str


@dataclass(frozen=True)
class CGResult:
    """A converged solution and its diagnostics."""

    solution: Tensor
    diagnostics: CGDiagnostics


class CGConvergenceError(RuntimeError):
    """Raised whenever CG cannot return a solver-valid field."""

    def __init__(self, diagnostics: CGDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "CG failed: "
            f"{diagnostics.reason} after {diagnostics.iterations} iterations "
            f"(relative residual {diagnostics.relative_residual:.6e})"
        )


def _dot(first: Tensor, second: Tensor) -> Tensor:
    return torch.dot(first.reshape(-1), second.reshape(-1))


def _relative_residual(residual: Tensor, denominator: float) -> float:
    return float(torch.linalg.vector_norm(residual).item() / denominator)


def _explicit_residual(
    apply: Callable[[Tensor], Tensor],
    rhs: Tensor,
    solution: Tensor,
    denominator: float,
) -> tuple[Tensor, float]:
    residual = rhs - apply(solution)
    return residual, _relative_residual(residual, denominator)


def _raise_failure(iterations: int, relative_residual: float, reason: str) -> None:
    raise CGConvergenceError(
        CGDiagnostics(
            iterations=iterations,
            relative_residual=relative_residual,
            converged=False,
            reason=reason,
        )
    )


def solve_cg(
    apply: Callable[[Tensor], Tensor],
    diagonal: Tensor,
    rhs: Tensor,
    config: CGConfig,
) -> CGResult:
    """Solve an SPD system using the locked zero-start Jacobi-CG policy."""
    if diagonal.shape != rhs.shape:
        raise ValueError("diagonal and rhs shapes must match")
    if diagonal.device != rhs.device or diagonal.dtype != rhs.dtype:
        raise ValueError("diagonal and rhs must share device and dtype")
    if not torch.isfinite(diagonal).all() or not torch.all(diagonal > 0.0):
        raise ValueError("Jacobi diagonal must be finite and strictly positive")
    if not torch.isfinite(rhs).all():
        raise ValueError("rhs must be finite")

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    denominator = max(float(torch.linalg.vector_norm(rhs).item()), 1.0e-12)
    relative_residual = _relative_residual(residual, denominator)
    if relative_residual <= config.relative_residual_tolerance:
        return CGResult(
            solution=solution,
            diagnostics=CGDiagnostics(0, relative_residual, True, "CONVERGED"),
        )

    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    residual_preconditioned = _dot(residual, preconditioned)
    if not torch.isfinite(residual_preconditioned) or residual_preconditioned <= 0:
        _raise_failure(0, relative_residual, "NON_POSITIVE_PRECONDITIONED_NORM")

    for iteration in range(1, config.maximum_iterations + 1):
        applied_direction = apply(direction)
        direction_curvature = _dot(direction, applied_direction)
        if not torch.isfinite(direction_curvature) or direction_curvature <= 0:
            _raise_failure(iteration - 1, relative_residual, "NON_POSITIVE_CURVATURE")

        step = residual_preconditioned / direction_curvature
        solution = solution + step * direction
        residual = residual - step * applied_direction
        relative_residual = _relative_residual(residual, denominator)
        if not torch.isfinite(residual).all() or not torch.isfinite(solution).all():
            _raise_failure(iteration, float("inf"), "NON_FINITE_ITERATE")
        if relative_residual <= config.relative_residual_tolerance:
            residual, relative_residual = _explicit_residual(
                apply,
                rhs,
                solution,
                denominator,
            )
            if relative_residual <= config.relative_residual_tolerance:
                return CGResult(
                    solution=solution,
                    diagnostics=CGDiagnostics(
                        iteration,
                        relative_residual,
                        True,
                        "CONVERGED",
                    ),
                )
            preconditioned = residual / diagonal
            direction = preconditioned.clone()
            residual_preconditioned = _dot(residual, preconditioned)
            if (
                not torch.isfinite(residual_preconditioned)
                or residual_preconditioned <= 0
            ):
                _raise_failure(
                    iteration,
                    relative_residual,
                    "NON_POSITIVE_PRECONDITIONED_NORM",
                )
            continue

        preconditioned = residual / diagonal
        updated_residual_preconditioned = _dot(residual, preconditioned)
        if (
            not torch.isfinite(updated_residual_preconditioned)
            or updated_residual_preconditioned <= 0
        ):
            _raise_failure(
                iteration,
                relative_residual,
                "NON_POSITIVE_PRECONDITIONED_NORM",
            )
        beta = updated_residual_preconditioned / residual_preconditioned
        direction = preconditioned + beta * direction
        residual_preconditioned = updated_residual_preconditioned

    _, relative_residual = _explicit_residual(
        apply,
        rhs,
        solution,
        denominator,
    )
    if relative_residual <= config.relative_residual_tolerance:
        return CGResult(
            solution=solution,
            diagnostics=CGDiagnostics(
                config.maximum_iterations,
                relative_residual,
                True,
                "CONVERGED",
            ),
        )
    _raise_failure(config.maximum_iterations, relative_residual, "MAXIMUM_ITERATIONS")
