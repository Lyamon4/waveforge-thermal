"""Solver-consistent MT2B validation, selection, and paired inference."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady

SciPy64Evaluator = Callable[[NDArray[np.float64], SourceLayoutTask], float]


@dataclass(frozen=True)
class SolverConsistentGapRow:
    task_id: str
    candidate_tmax: float
    reference_tmax: float
    relative_gap: float


@dataclass(frozen=True)
class SolverConsistentEvaluation:
    solver_id: str
    rows: tuple[SolverConsistentGapRow, ...]


@dataclass(frozen=True)
class MT2BCheckpointSummary:
    completed_updates: int
    split_name: str
    task_count: int
    invalid_count: int
    median_relative_gap: float
    p90_relative_gap: float
    worst_relative_gap: float
    median_absolute_tmax: float


@dataclass(frozen=True)
class BootstrapResult:
    statistic: str
    resamples: int
    seed: int
    median_paired_delta: float
    lower_bound: float
    upper_bound: float
    conditioning_ci_pass: bool


def _validate_binary(
    design: NDArray[np.float64], *, task_id: str
) -> NDArray[np.float64]:
    array = np.asarray(design, dtype=np.float64)
    if array.shape != (64, 64) or not np.isin(array, (0.0, 1.0)).all():
        raise ValueError(f"design for {task_id} must be binary with shape [64,64]")
    if int(np.count_nonzero(array)) != 1024:
        raise ValueError(f"binary design for {task_id} must contain exactly 1024 cells")
    return array


def independent_scipy64_tmax(
    design: NDArray[np.float64], task: SourceLayoutTask
) -> float:
    """Score one exact binary design with the common independent SciPy64 path."""
    binary = _validate_binary(design, task_id=task.task_id)
    grid = Grid2D(nx=64, ny=64)
    conductivity = 1.0 + 19.0 * binary
    peaks: list[float] = []
    for bounds in task.bounds:
        source = area_overlap_rectangular_source(grid, bounds, 1.0)
        result = solve_steady(
            grid,
            conductivity,
            source,
            BoundaryConditions.production(),
        )
        if result.normalized_residual > 1.0e-10:
            raise RuntimeError("independent SciPy64 residual exceeds tolerance")
        peaks.append(float(np.max(result.temperature)))
    peak = max(peaks)
    if not math.isfinite(peak) or peak <= 0.0:
        raise FloatingPointError("independent SciPy64 Tmax must be finite and positive")
    return peak


def evaluate_solver_consistent_gaps(
    candidate_designs: dict[str, NDArray[np.float64]],
    reference_designs: dict[str, NDArray[np.float64]],
    tasks: tuple[SourceLayoutTask, ...],
    *,
    scipy64_evaluator: SciPy64Evaluator,
) -> SolverConsistentEvaluation:
    """Evaluate both design families through one injected independent SciPy path."""
    if not tasks:
        raise ValueError("solver-consistent evaluation requires at least one task")
    rows: list[SolverConsistentGapRow] = []
    for task in tasks:
        if task.task_id not in candidate_designs:
            raise KeyError(f"missing candidate design for {task.task_id}")
        if task.task_id not in reference_designs:
            raise KeyError(f"missing reference design for {task.task_id}")
        candidate = _validate_binary(
            candidate_designs[task.task_id], task_id=task.task_id
        )
        reference = _validate_binary(
            reference_designs[task.task_id], task_id=task.task_id
        )
        candidate_tmax = float(scipy64_evaluator(candidate, task))
        reference_tmax = float(scipy64_evaluator(reference, task))
        if (
            not math.isfinite(candidate_tmax)
            or not math.isfinite(reference_tmax)
            or candidate_tmax <= 0.0
            or reference_tmax <= 0.0
        ):
            raise FloatingPointError("SciPy64 Tmax values must be finite and positive")
        rows.append(
            SolverConsistentGapRow(
                task_id=task.task_id,
                candidate_tmax=candidate_tmax,
                reference_tmax=reference_tmax,
                relative_gap=(candidate_tmax - reference_tmax) / reference_tmax,
            )
        )
    return SolverConsistentEvaluation(
        solver_id="independent_scipy_64",
        rows=tuple(rows),
    )


def select_mt2b_checkpoint(
    summaries: list[MT2BCheckpointSummary],
) -> MT2BCheckpointSummary:
    """Select by paired reference gap rather than legacy absolute temperature."""
    if not summaries:
        raise ValueError("at least one validation summary is required")
    if any(item.split_name != "validation" for item in summaries):
        raise ValueError("MT2B checkpoint selection may only use validation")
    eligible: list[MT2BCheckpointSummary] = []
    for item in summaries:
        metrics = (
            item.median_relative_gap,
            item.p90_relative_gap,
            item.worst_relative_gap,
            item.median_absolute_tmax,
        )
        if item.invalid_count == 0 and all(math.isfinite(value) for value in metrics):
            eligible.append(item)
    if not eligible:
        raise ValueError("no eligible MT2B validation checkpoint")
    return min(
        eligible,
        key=lambda item: (
            item.median_relative_gap,
            item.p90_relative_gap,
            item.median_absolute_tmax,
            item.completed_updates,
        ),
    )


def paired_bootstrap(
    raw_gaps: NDArray[np.float64],
    physics_gaps: NDArray[np.float64],
) -> BootstrapResult:
    """Run the exact preregistered paired-layout median bootstrap."""
    raw = np.asarray(raw_gaps, dtype=np.float64)
    physics = np.asarray(physics_gaps, dtype=np.float64)
    if raw.shape != (32,) or physics.shape != (32,):
        raise ValueError("paired bootstrap requires exactly 32 paired layouts")
    paired = np.column_stack((raw, physics))
    if not np.isfinite(paired).all():
        raise ValueError("paired bootstrap gaps must be finite")
    deltas = raw - physics
    rng = np.random.default_rng(2026092203)
    indices = rng.integers(0, 32, size=(10_000, 32))
    bootstrap_medians = np.median(deltas[indices], axis=1)
    lower, upper = np.percentile(bootstrap_medians, [2.5, 97.5])
    return BootstrapResult(
        statistic="median",
        resamples=10_000,
        seed=2026092203,
        median_paired_delta=float(np.median(deltas)),
        lower_bound=float(lower),
        upper_bound=float(upper),
        conditioning_ci_pass=bool(lower > 0.0),
    )
