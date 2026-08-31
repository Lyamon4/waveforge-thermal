"""Agreement-qualified scenario-vectorized direct-gradient MT2B reference."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from waveforge.design.batched_differentiable_solver import (
    BatchedSolveTrace,
    solve_steady_implicit_batched,
)
from waveforge.design.binary_readout import exact_cardinality_binary
from waveforge.design.objectives import objective_components
from waveforge.design.optimize import (
    alpha_for_iteration,
    array_sha256,
    beta_for_iteration,
    binarization_weight_for_iteration,
    initialize_logits,
)
from waveforge.design.parameterization import parameterize_design
from waveforge.physics.grid import Grid2D


@dataclass(frozen=True)
class ReferenceIterationRecord:
    iteration: int
    total_objective: float
    exact_peak: float
    gradient_norm_before_clipping: float
    maximum_relative_residual: float
    wall_seconds: float


@dataclass(frozen=True)
class ReferenceOptimizationResult:
    seed: int
    completed_iterations: int
    records: tuple[ReferenceIterationRecord, ...]
    final_logits: Tensor
    final_logits_sha256: str
    continuous_design: Tensor
    binary_design: Tensor
    binary_material_fraction: float


def _validate_trace(trace: BatchedSolveTrace, *, after_backward: bool) -> float:
    expected_roles = ["forward"] * 3 + (["adjoint"] * 3 if after_backward else [])
    if [record.role for record in trace.records] != expected_roles:
        raise RuntimeError("scenario-vectorized reference trace is incomplete")
    residual = max(record.relative_residual for record in trace.records)
    if any(not record.converged for record in trace.records) or residual > 1.0e-6:
        raise RuntimeError("scenario-vectorized reference CG is invalid")
    return residual


def optimize_reference_scenario_batched(
    sources: Tensor,
    *,
    seed: int,
    iterations: int = 600,
    allow_cpu_unit_test: bool = False,
) -> ReferenceOptimizationResult:
    """Run the locked optimizer with only the three scenarios vectorized."""
    if allow_cpu_unit_test:
        if iterations < 1 or iterations > 600:
            raise ValueError("unit-test reference iterations must lie in [1,600]")
    elif iterations != 600:
        raise ValueError("production reference optimization requires exactly 600 steps")
    if sources.shape != (3, 64, 64) or sources.dtype is not torch.float64:
        raise ValueError("reference sources must have shape [3,64,64] and float64")
    if sources.device.type != "cuda" and not allow_cpu_unit_test:
        raise ValueError("production reference optimization requires CUDA")
    if not torch.isfinite(sources).all():
        raise ValueError("reference sources must be finite")

    logits = initialize_logits(seed, device=sources.device).requires_grad_(True)
    optimizer = torch.optim.Adam(
        [logits],
        lr=0.05,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    records: list[ReferenceIterationRecord] = []
    grid = Grid2D(nx=64, ny=64)
    for iteration in range(iterations):
        if sources.device.type == "cuda":
            torch.cuda.synchronize(sources.device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        parameterized = parameterize_design(logits, beta=beta_for_iteration(iteration))
        design = parameterized.design
        conductivity = 1.0 + 19.0 * design.to(torch.float64).pow(3)
        trace = BatchedSolveTrace()
        temperatures = solve_steady_implicit_batched(
            conductivity.unsqueeze(0),
            sources.unsqueeze(0),
            grid,
            trace=trace,
        )[0]
        _validate_trace(trace, after_backward=False)
        objective = objective_components(
            temperatures,
            design,
            alpha=alpha_for_iteration(iteration),
            tv_weight=1.0e-3,
            binarization_weight=binarization_weight_for_iteration(iteration),
        )
        objective.total.backward()
        maximum_residual = _validate_trace(trace, after_backward=True)
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError("reference gradient is missing or non-finite")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [logits],
                max_norm=1.0,
                error_if_nonfinite=True,
            ).item()
        )
        optimizer.step()
        if sources.device.type == "cuda":
            torch.cuda.synchronize(sources.device)
        elapsed = time.perf_counter() - started
        values = (
            float(objective.total.detach().item()),
            float(objective.exact_peak.detach().item()),
            gradient_norm,
            maximum_residual,
            elapsed,
        )
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("reference iteration contains NaN or Inf")
        records.append(
            ReferenceIterationRecord(
                iteration=iteration,
                total_objective=values[0],
                exact_peak=values[1],
                gradient_norm_before_clipping=gradient_norm,
                maximum_relative_residual=maximum_residual,
                wall_seconds=elapsed,
            )
        )

    final_parameterized = parameterize_design(
        logits.detach(), beta=beta_for_iteration(iterations - 1)
    )
    continuous = final_parameterized.design.detach()
    binary, budget = exact_cardinality_binary(continuous, count=1024)
    return ReferenceOptimizationResult(
        seed=seed,
        completed_iterations=len(records),
        records=tuple(records),
        final_logits=logits.detach().cpu(),
        final_logits_sha256=array_sha256(logits),
        continuous_design=continuous.cpu(),
        binary_design=binary.cpu(),
        binary_material_fraction=budget.material_fraction,
    )
