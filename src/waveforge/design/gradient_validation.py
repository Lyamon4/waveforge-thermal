"""Blocking full-pipeline directional-gradient qualification."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import torch
from torch import Tensor

from waveforge.design.differentiable_solver import (
    SolveRecord,
    SolveTrace,
    solve_steady_implicit,
)
from waveforge.design.objectives import objective_components
from waveforge.design.parameterization import parameterize_design
from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.grid import Grid2D


@dataclass(frozen=True)
class GradientValidationConfig:
    """Pre-registered directional finite-difference protocol."""

    direction_seeds: tuple[int, ...] = (7201, 7202, 7203, 7204, 7205)
    cpu_steps: tuple[float, ...] = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)
    cuda_steps: tuple[float, ...] = (1.0e-2, 3.0e-3, 1.0e-3)
    cpu_tolerance: float = 1.0e-4
    cuda_tolerance: float = 5.0e-3
    latent_shape: tuple[int, int] = (16, 16)
    simulation_shape: tuple[int, int] = (64, 64)
    base_logit_seed: int = 20260828
    beta: float = 8.0
    alpha: float = 500.0
    binarization_weight: float = 0.02


@dataclass(frozen=True)
class GradientCheckRecord:
    """One registered step-size result for one direction."""

    device: str
    dtype: str
    direction_seed: int
    step_size: float
    automatic_derivative: float
    finite_difference_derivative: float
    relative_error: float
    passed: bool


@dataclass(frozen=True)
class GradientValidationReport:
    """Complete CPU or CUDA mixed-precision gradient-gate evidence."""

    records: tuple[GradientCheckRecord, ...]
    solve_records: tuple[SolveRecord, ...]
    passed: bool
    maximum_explicit_residual: float
    physics_dtypes: tuple[str, ...]
    gradient_dtype: str


def direction_for_seed(
    seed: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    shape: tuple[int, int] = (16, 16),
) -> Tensor:
    """Construct one reproducible L2-normalized Gaussian direction."""
    values = np.random.default_rng(seed).normal(size=shape)
    direction = torch.as_tensor(values, dtype=dtype, device=device)
    return direction / torch.linalg.vector_norm(direction)


def _source_batch(grid: Grid2D, device: torch.device) -> Tensor:
    bounds = (
        (0.40, 0.60, 0.62, 0.82),
        (0.18, 0.38, 0.62, 0.82),
        (0.62, 0.82, 0.62, 0.82),
    )
    sources = np.stack(
        [area_overlap_rectangular_source(grid, item, 1.0) for item in bounds]
    )
    return torch.as_tensor(sources, dtype=torch.float64, device=device)


def _evaluate_pipeline(
    logits: Tensor,
    sources: Tensor,
    grid: Grid2D,
    config: GradientValidationConfig,
) -> tuple[Tensor, SolveTrace]:
    parameterized = parameterize_design(
        logits,
        beta=config.beta,
        simulation_shape=config.simulation_shape,
    )
    design = parameterized.design
    design_physics = design.to(torch.float64)
    conductivity = 1.0 + 19.0 * design_physics**3
    trace = SolveTrace()
    temperatures = solve_steady_implicit(
        conductivity,
        sources,
        grid,
        trace=trace,
    )
    components = objective_components(
        temperatures,
        design,
        alpha=config.alpha,
        tv_weight=1.0e-3,
        binarization_weight=config.binarization_weight,
    )
    return components.total, trace


def _has_adjacent_passes(records: list[GradientCheckRecord]) -> bool:
    flags = [record.passed for record in records]
    return any(first and second for first, second in pairwise(flags))


def validate_full_pipeline_gradient(
    config: GradientValidationConfig,
    *,
    device: torch.device,
    design_dtype: torch.dtype,
) -> GradientValidationReport:
    """Run the exact five-direction CPU or CUDA mixed-precision gate."""
    if device.type == "cpu" and design_dtype is not torch.float64:
        raise ValueError("CPU gradient validation requires float64 design state")
    if device.type == "cuda" and design_dtype is not torch.float32:
        raise ValueError("CUDA mixed validation requires float32 design state")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Gate 2A locked environment requires CUDA")

    grid = Grid2D(
        nx=config.simulation_shape[1],
        ny=config.simulation_shape[0],
    )
    sources = _source_batch(grid, device)
    initial_values = np.random.default_rng(config.base_logit_seed).normal(
        loc=0.0,
        scale=0.1,
        size=config.latent_shape,
    )
    logits = torch.as_tensor(
        initial_values,
        dtype=design_dtype,
        device=device,
    ).requires_grad_(True)

    objective, automatic_trace = _evaluate_pipeline(logits, sources, grid, config)
    (gradient,) = torch.autograd.grad(objective, logits)
    if gradient.dtype is not design_dtype or not torch.isfinite(gradient).all():
        raise FloatingPointError("full-pipeline gradient has invalid dtype or values")

    steps = config.cpu_steps if device.type == "cpu" else config.cuda_steps
    tolerance = config.cpu_tolerance if device.type == "cpu" else config.cuda_tolerance
    records: list[GradientCheckRecord] = []
    solve_records = list(automatic_trace.records)
    direction_outcomes: list[bool] = []
    for seed in config.direction_seeds:
        direction = direction_for_seed(
            seed,
            dtype=design_dtype,
            device=device,
            shape=config.latent_shape,
        )
        automatic_derivative = float(torch.sum(gradient * direction).item())
        direction_records: list[GradientCheckRecord] = []
        for step_size in steps:
            with torch.no_grad():
                plus, plus_trace = _evaluate_pipeline(
                    logits + step_size * direction,
                    sources,
                    grid,
                    config,
                )
                minus, minus_trace = _evaluate_pipeline(
                    logits - step_size * direction,
                    sources,
                    grid,
                    config,
                )
            solve_records.extend(plus_trace.records)
            solve_records.extend(minus_trace.records)
            finite_difference = float((plus - minus).item() / (2.0 * step_size))
            relative_error = abs(automatic_derivative - finite_difference) / max(
                abs(automatic_derivative),
                abs(finite_difference),
                1.0e-12,
            )
            record = GradientCheckRecord(
                device=str(device),
                dtype=str(design_dtype).removeprefix("torch."),
                direction_seed=seed,
                step_size=step_size,
                automatic_derivative=automatic_derivative,
                finite_difference_derivative=finite_difference,
                relative_error=relative_error,
                passed=relative_error <= tolerance,
            )
            records.append(record)
            direction_records.append(record)
        direction_outcomes.append(_has_adjacent_passes(direction_records))

    maximum_residual = max(record.relative_residual for record in solve_records)
    physics_dtypes = tuple(sorted({record.dtype for record in solve_records}))
    valid_solves = all(
        record.converged and record.relative_residual <= 1.0e-6
        for record in solve_records
    )
    return GradientValidationReport(
        records=tuple(records),
        solve_records=tuple(solve_records),
        passed=all(direction_outcomes) and valid_solves,
        maximum_explicit_residual=maximum_residual,
        physics_dtypes=physics_dtypes,
        gradient_dtype=str(gradient.dtype).removeprefix("torch."),
    )
