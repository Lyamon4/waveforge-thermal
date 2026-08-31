"""Frozen validation and condition-causality diagnostics for shared NCA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from waveforge.design.binary_readout import exact_cardinality_binary
from waveforge.design.differentiable_solver import SolveTrace, solve_steady_implicit
from waveforge.ml.multitask_protocol import PRIMARY_BINARY_CELL_COUNT
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.ml.nca import PureNCA, build_static_condition, project_nca_material
from waveforge.ml.nca_training import model_state_sha256
from waveforge.physics.grid import Grid2D


@dataclass(frozen=True)
class FrozenTaskEvaluation:
    """One frozen model's exact-budget 64-grid result."""

    task_id: str
    peak_temperature: float
    binary_material_fraction: float
    binary_design: NDArray[np.float64]


@dataclass(frozen=True)
class FrozenCheckpointEvaluation:
    """Frozen results for one checkpoint and one declared split."""

    checkpoint: Path
    completed_updates: int
    model_hash: str
    split_name: str
    tasks: tuple[FrozenTaskEvaluation, ...]


@dataclass(frozen=True)
class ValidationSummary:
    """Paired validation gaps used for prospective checkpoint selection."""

    completed_updates: int
    split_name: str
    task_count: int
    invalid_count: int
    median_peak: float
    p90_peak: float
    worst_peak: float
    median_relative_gap: float
    p90_relative_gap: float
    worst_relative_gap: float


@dataclass(frozen=True)
class ConditionCausalitySummary:
    """Matched-versus-cyclically-shuffled conditioning evidence."""

    task_count: int
    matched_win_count: int
    matched_win_fraction: float
    median_relative_matched_advantage: float
    pass_gate: bool


@dataclass(frozen=True)
class BinaryDiversitySummary:
    """Pairwise diversity of exact-cardinality generated designs."""

    pair_count: int
    mean_hamming_fraction: float
    mean_jaccard_similarity: float


def _load_frozen_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[PureNCA, int, str]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported multi-task checkpoint schema")
    model = PureNCA().to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    actual_hash = model_state_sha256(model)
    if actual_hash != payload["model_state_sha256"]:
        raise ValueError("checkpoint model hash mismatch")
    return model, int(payload["completed_updates"]), actual_hash


def _score_frozen_task(
    model: PureNCA,
    task: SourceLayoutTask,
    *,
    conditioning_task: SourceLayoutTask,
    device: torch.device,
) -> FrozenTaskEvaluation:
    physical_sources = torch.as_tensor(
        task.sources,
        dtype=torch.float64,
        device=device,
    )
    conditioning_sources = torch.as_tensor(
        conditioning_task.sources,
        dtype=torch.float64,
        device=device,
    )
    with torch.no_grad():
        condition = build_static_condition(conditioning_sources)
        rollout = model.rollout(condition)
        continuous = project_nca_material(rollout.material_logit, beta=8.0).design
        binary, budget = exact_cardinality_binary(
            continuous,
            count=PRIMARY_BINARY_CELL_COUNT,
        )
        conductivity = 1.0 + 19.0 * binary.to(torch.float64)
        trace = SolveTrace()
        temperatures = solve_steady_implicit(
            conductivity,
            physical_sources,
            Grid2D(nx=64, ny=64),
            trace=trace,
        )
    if len(trace.records) != 3 or any(
        record.role != "forward"
        or not record.converged
        or record.relative_residual > 1.0e-6
        for record in trace.records
    ):
        raise RuntimeError("frozen evaluation CG verification failed")
    peak = float(torch.max(temperatures).item())
    if not math.isfinite(peak):
        raise FloatingPointError("frozen evaluation produced a non-finite peak")
    return FrozenTaskEvaluation(
        task_id=task.task_id,
        peak_temperature=peak,
        binary_material_fraction=budget.material_fraction,
        binary_design=binary.detach().cpu().numpy().astype(np.float64, copy=False),
    )


def evaluate_frozen_checkpoint(
    checkpoint: Path,
    tasks: tuple[SourceLayoutTask, ...],
    *,
    split_name: Literal["validation", "test_id", "test_ood"],
    device: torch.device,
    conditioning_tasks: tuple[SourceLayoutTask, ...] | None = None,
) -> FrozenCheckpointEvaluation:
    """Generate and physically score tasks without optimizer or backward calls."""
    if len(tasks) < 1:
        raise ValueError("frozen evaluation requires at least one task")
    conditions = tasks if conditioning_tasks is None else conditioning_tasks
    if len(conditions) != len(tasks):
        raise ValueError("conditioning task count must match physical task count")
    model, completed_updates, model_hash = _load_frozen_model(checkpoint, device)
    evaluations = tuple(
        _score_frozen_task(
            model,
            task,
            conditioning_task=condition,
            device=device,
        )
        for task, condition in zip(tasks, conditions, strict=True)
    )
    return FrozenCheckpointEvaluation(
        checkpoint=checkpoint,
        completed_updates=completed_updates,
        model_hash=model_hash,
        split_name=split_name,
        tasks=evaluations,
    )


def summarize_against_reference(
    *,
    completed_updates: int,
    split_name: str,
    task_ids: tuple[str, ...],
    candidate_peaks: tuple[float, ...],
    reference_peaks: dict[str, float],
) -> ValidationSummary:
    """Calculate paired relative gaps from full-precision values."""
    if len(task_ids) != len(candidate_peaks) or len(task_ids) < 1:
        raise ValueError("task IDs and candidate peaks must have equal nonzero length")
    gaps: list[float] = []
    valid_peaks: list[float] = []
    invalid_count = 0
    for task_id, candidate in zip(task_ids, candidate_peaks, strict=True):
        if task_id not in reference_peaks:
            raise KeyError(f"missing reference result for task {task_id}")
        reference = reference_peaks[task_id]
        if (
            not math.isfinite(candidate)
            or not math.isfinite(reference)
            or reference <= 0.0
        ):
            invalid_count += 1
            continue
        valid_peaks.append(candidate)
        gaps.append((candidate - reference) / reference)
    if gaps:
        gap_array = np.asarray(gaps, dtype=np.float64)
        peak_array = np.asarray(valid_peaks, dtype=np.float64)
        median_peak = float(np.median(peak_array))
        p90_peak = float(np.quantile(peak_array, 0.9))
        worst_peak = float(np.max(peak_array))
        median = float(np.median(gap_array))
        p90 = float(np.quantile(gap_array, 0.9))
        worst = float(np.max(gap_array))
    else:
        median_peak = p90_peak = worst_peak = math.inf
        median = p90 = worst = math.inf
    return ValidationSummary(
        completed_updates=completed_updates,
        split_name=split_name,
        task_count=len(task_ids),
        invalid_count=invalid_count,
        median_peak=median_peak,
        p90_peak=p90_peak,
        worst_peak=worst_peak,
        median_relative_gap=median,
        p90_relative_gap=p90,
        worst_relative_gap=worst,
    )


def select_validation_checkpoint(
    summaries: list[ValidationSummary],
) -> ValidationSummary:
    """Select by median gap, p90 gap, invalid count, then earlier checkpoint."""
    if not summaries:
        raise ValueError("at least one validation summary is required")
    for item in summaries:
        if item.split_name != "validation":
            raise ValueError("checkpoint selection may only use validation summaries")
        numerical = (
            item.median_peak,
            item.p90_peak,
            item.worst_peak,
            item.median_relative_gap,
            item.p90_relative_gap,
            item.worst_relative_gap,
        )
        if not all(math.isfinite(value) for value in numerical):
            raise ValueError("validation summary metrics must be finite")
    return min(
        summaries,
        key=lambda item: (
            item.median_peak,
            item.p90_peak,
            item.invalid_count,
            item.completed_updates,
        ),
    )


def condition_causality_summary(
    *,
    matched: list[float],
    shuffled: list[float],
) -> ConditionCausalitySummary:
    """Require matched conditioning to win strictly on at least 23 of 32 tasks."""
    if len(matched) != len(shuffled) or len(matched) != 32:
        raise ValueError("condition causality requires exactly 32 paired tasks")
    paired = np.column_stack(
        [np.asarray(matched, dtype=np.float64), np.asarray(shuffled, dtype=np.float64)]
    )
    if not np.isfinite(paired).all() or np.any(paired <= 0.0):
        raise ValueError("condition causality peaks must be finite and positive")
    wins = int(np.sum(paired[:, 0] < paired[:, 1]))
    advantages = (paired[:, 1] - paired[:, 0]) / paired[:, 1]
    return ConditionCausalitySummary(
        task_count=32,
        matched_win_count=wins,
        matched_win_fraction=wins / 32,
        median_relative_matched_advantage=float(np.median(advantages)),
        pass_gate=wins >= 23,
    )


def pairwise_binary_diversity(
    designs: list[NDArray[np.float64]],
) -> BinaryDiversitySummary:
    """Report pairwise Hamming fraction and Jaccard similarity."""
    if len(designs) < 2:
        raise ValueError("diversity requires at least two designs")
    arrays = [np.asarray(design) for design in designs]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("all binary designs must share a shape")
    if any(not np.isin(array, (0.0, 1.0)).all() for array in arrays):
        raise ValueError("all designs must be binary")
    hamming: list[float] = []
    jaccard: list[float] = []
    for left in range(len(arrays)):
        for right in range(left + 1, len(arrays)):
            first = arrays[left].astype(bool, copy=False)
            second = arrays[right].astype(bool, copy=False)
            hamming.append(float(np.mean(first != second)))
            union = int(np.count_nonzero(first | second))
            intersection = int(np.count_nonzero(first & second))
            jaccard.append(intersection / union if union else 1.0)
    return BinaryDiversitySummary(
        pair_count=len(hamming),
        mean_hamming_fraction=float(np.mean(hamming)),
        mean_jaccard_similarity=float(np.mean(jaccard)),
    )
