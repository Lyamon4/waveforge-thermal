"""Qualified NLopt MMA baseline using WaveForge's differentiable objective."""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from waveforge.design.batched_differentiable_solver import (
    BatchedSolveTrace,
    solve_steady_implicit_batched,
)
from waveforge.design.binary_readout import exact_cardinality_binary
from waveforge.design.objectives import objective_components
from waveforge.design.optimize import (
    alpha_for_iteration,
    beta_for_iteration,
    binarization_weight_for_iteration,
)
from waveforge.design.parameterization import parameterize_design
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.physics.grid import Grid2D
from waveforge.reproducibility import content_hash

_REGISTERED_BUDGETS = frozenset({25, 50, 100, 200, 600})
_GRID = Grid2D(nx=64, ny=64)


@dataclass(frozen=True)
class MMAQualification:
    available: bool
    backend: str
    version: str | None
    algorithm: str
    reason: str


@dataclass(frozen=True)
class MMAEvaluationRecord:
    evaluation: int
    beta: float
    alpha: float
    binarization_weight: float
    objective: float
    gradient_norm: float
    maximum_relative_residual: float


@dataclass(frozen=True)
class MMASnapshot:
    evaluation: int
    logits: NDArray[np.float64]
    binary_design: NDArray[np.float64]
    binary_cell_count: int
    binary_material_fraction: float


@dataclass(frozen=True)
class MMABaselineResult:
    status: Literal["PASS", "MMA_BACKEND_UNAVAILABLE", "INVALID_RUN"]
    requested_evaluations: int
    completed_evaluations: int
    seed: int
    termination_codes: tuple[int, ...]
    records: tuple[MMAEvaluationRecord, ...]
    final_logits: NDArray[np.float64] | None
    final_logits_sha256: str | None
    continuous_design: NDArray[np.float64] | None
    binary_design: NDArray[np.float64] | None
    binary_cell_count: int | None
    binary_material_fraction: float | None
    snapshots: dict[int, MMASnapshot]


def qualify_mma_backend() -> MMAQualification:
    """Accept only the prospectively pinned NLopt 2.10 LD_MMA backend."""
    try:
        version = importlib.metadata.version("nlopt")
        import nlopt  # type: ignore[import-not-found]
    except (importlib.metadata.PackageNotFoundError, ImportError):
        return MMAQualification(
            available=False,
            backend="nlopt",
            version=None,
            algorithm="LD_MMA",
            reason="MMA_BACKEND_UNAVAILABLE",
        )
    available = version.startswith("2.10.") and hasattr(nlopt, "LD_MMA")
    return MMAQualification(
        available=available,
        backend="nlopt",
        version=version,
        algorithm="LD_MMA",
        reason="QUALIFIED" if available else "MMA_BACKEND_UNAVAILABLE",
    )


def _validate_trace(trace: BatchedSolveTrace) -> float:
    expected = ["forward"] * 3 + ["adjoint"] * 3
    if [record.role for record in trace.records] != expected:
        raise RuntimeError("MMA objective physics trace is incomplete")
    maximum = max(record.relative_residual for record in trace.records)
    if any(not record.converged for record in trace.records) or maximum > 1.0e-6:
        raise RuntimeError("MMA objective physics residual exceeds tolerance")
    return maximum


def _evaluate_mma_objective(
    flat_logits: NDArray[np.float64],
    task: SourceLayoutTask,
    *,
    beta: float,
    alpha: float,
    binarization_weight: float,
    allow_cpu_unit_test: bool = False,
    device: torch.device | None = None,
) -> tuple[float, NDArray[np.float64], float]:
    array = np.array(flat_logits, dtype=np.float64, copy=True)
    if array.shape != (256,) or not np.isfinite(array).all():
        raise ValueError("MMA logits must be a finite vector of length 256")
    target_device = device or torch.device("cpu" if allow_cpu_unit_test else "cuda")
    if target_device.type != "cuda" and not allow_cpu_unit_test:
        raise ValueError("production MMA objective requires CUDA")
    logits = (
        torch.as_tensor(
            array.reshape(16, 16),
            dtype=torch.float32,
            device=target_device,
        )
        .clone()
        .requires_grad_(True)
    )
    sources = torch.as_tensor(
        task.sources,
        dtype=torch.float64,
        device=target_device,
    )
    parameterized = parameterize_design(logits, beta=beta)
    design = parameterized.design
    conductivity = 1.0 + 19.0 * design.to(torch.float64).pow(3)
    trace = BatchedSolveTrace()
    temperatures = solve_steady_implicit_batched(
        conductivity.unsqueeze(0),
        sources.unsqueeze(0),
        _GRID,
        trace=trace,
    )[0]
    objective = objective_components(
        temperatures,
        design,
        alpha=alpha,
        tv_weight=0.001,
        binarization_weight=binarization_weight,
    )
    objective.total.backward()
    maximum_residual = _validate_trace(trace)
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise FloatingPointError("MMA objective gradient is missing or non-finite")
    value = float(objective.total.detach().item())
    gradient = (
        logits.grad.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    )
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("MMA callback returned NaN or Inf")
    return value, gradient, maximum_residual


def mma_objective_callback(
    flat_logits: NDArray[np.float64],
    task: SourceLayoutTask,
    *,
    beta: float,
    alpha: float,
    binarization_weight: float,
    allow_cpu_unit_test: bool = False,
    device: torch.device | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Return one solver-validated objective and total derivative for NLopt."""
    value, gradient, _ = _evaluate_mma_objective(
        flat_logits,
        task,
        beta=beta,
        alpha=alpha,
        binarization_weight=binarization_weight,
        allow_cpu_unit_test=allow_cpu_unit_test,
        device=device,
    )
    return value, gradient


def _continuation_segments(evaluations: int) -> tuple[tuple[int, int], ...]:
    boundaries = (0, 200, 350, 500, 600)
    segments: list[tuple[int, int]] = []
    for start, stop in pairwise(boundaries):
        if start >= evaluations:
            break
        segments.append((start, min(stop, evaluations)))
    return tuple(segments)


def optimize_mma(
    task: SourceLayoutTask,
    evaluations: int,
    seed: int,
    *,
    device: torch.device | None = None,
    allow_cpu_unit_test: bool = False,
    snapshot_evaluations: tuple[int, ...] = (),
) -> MMABaselineResult:
    """Run the registered restarted-continuation NLopt LD_MMA baseline."""
    if evaluations not in _REGISTERED_BUDGETS:
        raise ValueError("MMA evaluations must be one of 25/50/100/200/600")
    if tuple(sorted(set(snapshot_evaluations))) != snapshot_evaluations or any(
        value not in _REGISTERED_BUDGETS or value > evaluations
        for value in snapshot_evaluations
    ):
        raise ValueError(
            "MMA snapshot evaluations must be registered and within budget"
        )
    qualification = qualify_mma_backend()
    if not qualification.available:
        return MMABaselineResult(
            status="MMA_BACKEND_UNAVAILABLE",
            requested_evaluations=evaluations,
            completed_evaluations=0,
            seed=seed,
            termination_codes=(),
            records=(),
            final_logits=None,
            final_logits_sha256=None,
            continuous_design=None,
            binary_design=None,
            binary_cell_count=None,
            binary_material_fraction=None,
            snapshots={},
        )
    import nlopt  # type: ignore[import-not-found]

    target_device = device or torch.device("cpu" if allow_cpu_unit_test else "cuda")
    rng = np.random.Generator(np.random.PCG64(seed))
    current = rng.normal(0.0, 0.1, size=256).astype(np.float64)
    records: list[MMAEvaluationRecord] = []
    termination_codes: list[int] = []
    evaluation_index = 0
    raw_snapshots: dict[int, NDArray[np.float64]] = {}
    try:
        for start, stop in _continuation_segments(evaluations):
            beta = beta_for_iteration(start)
            alpha = alpha_for_iteration(start)
            binary_weight = binarization_weight_for_iteration(start)

            def callback(
                x: NDArray[np.float64],
                grad: NDArray[np.float64],
                *,
                beta: float = beta,
                alpha: float = alpha,
                binary_weight: float = binary_weight,
            ) -> float:
                nonlocal evaluation_index
                value, gradient, maximum_residual = _evaluate_mma_objective(
                    x,
                    task,
                    beta=beta,
                    alpha=alpha,
                    binarization_weight=binary_weight,
                    allow_cpu_unit_test=allow_cpu_unit_test,
                    device=target_device,
                )
                if grad.size:
                    grad[:] = gradient
                records.append(
                    MMAEvaluationRecord(
                        evaluation=evaluation_index,
                        beta=beta,
                        alpha=alpha,
                        binarization_weight=binary_weight,
                        objective=value,
                        gradient_norm=float(np.linalg.norm(gradient)),
                        maximum_relative_residual=maximum_residual,
                    )
                )
                evaluation_index += 1
                if evaluation_index in snapshot_evaluations:
                    raw_snapshots[evaluation_index] = np.array(x, copy=True)
                return value

            optimizer = nlopt.opt(nlopt.LD_MMA, 256)
            optimizer.set_min_objective(callback)
            optimizer.set_maxeval(stop - start)
            current = np.asarray(optimizer.optimize(current), dtype=np.float64)
            if stop in snapshot_evaluations:
                raw_snapshots[stop] = current.copy()
            termination_codes.append(int(optimizer.last_optimize_result()))
    except (RuntimeError, FloatingPointError, nlopt.nlopt_error):
        return MMABaselineResult(
            status="INVALID_RUN",
            requested_evaluations=evaluations,
            completed_evaluations=len(records),
            seed=seed,
            termination_codes=tuple(termination_codes),
            records=tuple(records),
            final_logits=None,
            final_logits_sha256=None,
            continuous_design=None,
            binary_design=None,
            binary_cell_count=None,
            binary_material_fraction=None,
            snapshots={},
        )

    logits_tensor = torch.as_tensor(
        current.reshape(16, 16),
        dtype=torch.float32,
        device=target_device,
    )
    final_stage = min(evaluations - 1, 599)
    parameterized = parameterize_design(
        logits_tensor,
        beta=beta_for_iteration(final_stage),
    )
    continuous = parameterized.design.detach().cpu()
    binary, budget = exact_cardinality_binary(continuous, count=1024)
    continuous_array = continuous.numpy().astype(np.float64, copy=False)
    binary_array = binary.numpy().astype(np.float64, copy=False)
    raw_snapshots[evaluations] = current.copy()
    snapshots: dict[int, MMASnapshot] = {}
    for snapshot_evaluation in snapshot_evaluations:
        snapshot_logits = raw_snapshots.get(snapshot_evaluation)
        if snapshot_logits is None:
            continue
        snapshot_tensor = torch.as_tensor(
            snapshot_logits.reshape(16, 16),
            dtype=torch.float32,
            device=target_device,
        )
        snapshot_design = (
            parameterize_design(
                snapshot_tensor,
                beta=beta_for_iteration(snapshot_evaluation - 1),
            )
            .design.detach()
            .cpu()
        )
        snapshot_binary, snapshot_budget = exact_cardinality_binary(
            snapshot_design, count=1024
        )
        snapshot_binary_array = snapshot_binary.numpy().astype(np.float64, copy=False)
        snapshots[snapshot_evaluation] = MMASnapshot(
            evaluation=snapshot_evaluation,
            logits=snapshot_logits.copy(),
            binary_design=snapshot_binary_array,
            binary_cell_count=int(np.count_nonzero(snapshot_binary_array)),
            binary_material_fraction=snapshot_budget.material_fraction,
        )
    return MMABaselineResult(
        status="PASS" if len(records) == evaluations else "INVALID_RUN",
        requested_evaluations=evaluations,
        completed_evaluations=len(records),
        seed=seed,
        termination_codes=tuple(termination_codes),
        records=tuple(records),
        final_logits=current.copy(),
        final_logits_sha256=content_hash(current),
        continuous_design=continuous_array,
        binary_design=binary_array,
        binary_cell_count=int(np.count_nonzero(binary_array)),
        binary_material_fraction=budget.material_fraction,
        snapshots=snapshots,
    )
