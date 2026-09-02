"""Deterministic candidate scoring and exactly-one MT3 refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from waveforge.design.batched_differentiable_solver import (
    BatchedSolveTrace,
    solve_steady_implicit_batched,
)
from waveforge.design.binary_readout import exact_cardinality_binary
from waveforge.design.objectives import objective_components
from waveforge.design.parameterization import parameterize_design
from waveforge.ml.mt2b_evaluation import SciPy64Evaluator
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.physics.grid import Grid2D


class MT3InvalidRun(RuntimeError):
    """Raised when inference violates the registered single-chain accounting."""


@dataclass(frozen=True)
class CandidateScore:
    head_index: int
    binary_tmax: float


@dataclass(frozen=True)
class RefinementTraceRecord:
    iteration: int
    head_index: int
    total_objective: float
    exact_peak: float
    gradient_norm_before_clipping: float
    maximum_relative_residual: float


@dataclass(frozen=True)
class MT3RefinementResult:
    selected_head: int
    candidate_scores: tuple[CandidateScore, ...]
    requested_steps: int
    refined_heads: tuple[int, ...]
    records: tuple[RefinementTraceRecord, ...]
    final_logits: Tensor
    continuous_design: Tensor
    binary_design: Tensor

    @property
    def total_refinement_updates(self) -> int:
        return len(self.records)


class RefinementStepper(Protocol):
    def __call__(
        self,
        logits: Tensor,
        sources: Tensor,
        head_index: int,
        iteration: int,
    ) -> tuple[Tensor, RefinementTraceRecord]: ...


def _validate_candidates(candidate_logits: Tensor, binary_designs: Tensor) -> None:
    expected = (4, 64, 64)
    if (
        tuple(candidate_logits.shape) != expected
        or not candidate_logits.is_floating_point()
    ):
        raise ValueError("candidate logits must be floating point with shape [4,64,64]")
    if tuple(binary_designs.shape) != expected:
        raise ValueError("binary designs must have shape [4,64,64]")
    if (
        not torch.isfinite(candidate_logits).all()
        or not torch.isfinite(binary_designs).all()
    ):
        raise ValueError("candidate tensors must be finite")
    if not torch.all((binary_designs == 0.0) | (binary_designs == 1.0)):
        raise ValueError("candidate binary designs must contain only zero and one")
    counts = binary_designs.sum(dim=(-2, -1))
    if not torch.equal(counts, torch.full_like(counts, 1024)):
        raise ValueError(
            "every candidate binary design must contain exactly 1024 cells"
        )


def _score_candidates(
    candidate_logits: Tensor,
    binary_designs: Tensor,
    task: SourceLayoutTask,
    scorer: SciPy64Evaluator,
) -> tuple[CandidateScore, ...]:
    _validate_candidates(candidate_logits, binary_designs)
    scores: list[CandidateScore] = []
    for head_index in range(4):
        binary = (
            binary_designs[head_index]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        value = float(scorer(np.asarray(binary, dtype=np.float64), task))
        if not math.isfinite(value) or value <= 0.0:
            raise FloatingPointError(
                "candidate SciPy64 score must be finite and positive"
            )
        scores.append(CandidateScore(head_index=head_index, binary_tmax=value))
    return tuple(scores)


def select_best_candidate(
    candidate_logits: Tensor,
    binary_designs: Tensor,
    task: SourceLayoutTask,
    scorer: SciPy64Evaluator,
) -> CandidateScore:
    """Score all four heads once and apply the locked numeric-head tie break."""
    scores = _score_candidates(candidate_logits, binary_designs, task, scorer)
    return min(scores, key=lambda row: (row.binary_tmax, row.head_index))


def validate_refinement_accounting(
    *,
    refined_heads: tuple[int, ...],
    requested_steps: int,
    record_count: int,
) -> None:
    """Fail closed on any four-chain or incomplete-refinement artifact."""
    if requested_steps not in (25, 50):
        raise ValueError("registered refinement requires exactly 25 or 50 steps")
    if len(refined_heads) != 1 or refined_heads[0] not in range(4):
        raise MT3InvalidRun("exactly one candidate must be refined")
    if record_count != requested_steps:
        raise MT3InvalidRun(
            f"refinement trace must contain exactly {requested_steps} records"
        )


def _validate_trace(trace: BatchedSolveTrace, *, after_backward: bool) -> float:
    expected_roles = ["forward"] * 3 + (["adjoint"] * 3 if after_backward else [])
    if [record.role for record in trace.records] != expected_roles:
        raise MT3InvalidRun("refinement physics trace is incomplete")
    maximum = max(record.relative_residual for record in trace.records)
    if any(not record.converged for record in trace.records) or maximum > 1.0e-6:
        raise MT3InvalidRun("refinement physics residual exceeds tolerance")
    return maximum


def _run_physics_refinement(
    selected: CandidateScore,
    selected_logits: Tensor,
    sources: Tensor,
    *,
    steps: int,
    capture_steps: tuple[int, ...] = (),
) -> tuple[
    Tensor,
    tuple[RefinementTraceRecord, ...],
    dict[int, Tensor],
]:
    logits = selected_logits.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [logits],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    grid = Grid2D(nx=64, ny=64)
    records: list[RefinementTraceRecord] = []
    snapshots: dict[int, Tensor] = {}
    for iteration in range(steps):
        optimizer.zero_grad(set_to_none=True)
        parameterized = parameterize_design(logits, beta=8.0)
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
            alpha=500.0,
            tv_weight=0.001,
            binarization_weight=0.02,
        )
        objective.total.backward()
        maximum_residual = _validate_trace(trace, after_backward=True)
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError("refinement gradient is missing or non-finite")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [logits],
                max_norm=1.0,
                error_if_nonfinite=True,
            ).item()
        )
        optimizer.step()
        record = RefinementTraceRecord(
            iteration=iteration,
            head_index=selected.head_index,
            total_objective=float(objective.total.detach().item()),
            exact_peak=float(objective.exact_peak.detach().item()),
            gradient_norm_before_clipping=gradient_norm,
            maximum_relative_residual=maximum_residual,
        )
        if not all(
            math.isfinite(value)
            for value in (
                record.total_objective,
                record.exact_peak,
                record.gradient_norm_before_clipping,
                record.maximum_relative_residual,
            )
        ):
            raise FloatingPointError("refinement record contains NaN or Inf")
        records.append(record)
        completed = iteration + 1
        if completed in capture_steps:
            snapshots[completed] = logits.detach().clone()
    return logits.detach(), tuple(records), snapshots


def refine_selected_candidate(
    selected: CandidateScore,
    candidate_logits: Tensor,
    sources: Tensor,
    *,
    steps: int,
    allow_cpu_unit_test: bool = False,
    stepper: RefinementStepper | None = None,
) -> MT3RefinementResult:
    """Create one optimizer chain for the already selected candidate only."""
    if steps not in (25, 50):
        raise ValueError("registered refinement requires exactly 25 or 50 steps")
    if tuple(candidate_logits.shape) != (4, 64, 64):
        raise ValueError("candidate logits must have shape [4,64,64]")
    if tuple(sources.shape) != (3, 64, 64) or sources.dtype is not torch.float64:
        raise ValueError("sources must have shape [3,64,64] and float64")
    if sources.device.type != "cuda" and not allow_cpu_unit_test:
        raise ValueError("production refinement requires CUDA")
    if selected.head_index not in range(4):
        raise ValueError("selected head index must lie in [0,4)")

    selected_logits = candidate_logits[selected.head_index].to(device=sources.device)
    if stepper is None:
        final_logits, records, _ = _run_physics_refinement(
            selected,
            selected_logits,
            sources,
            steps=steps,
        )
    else:
        current = selected_logits.detach().clone()
        injected_records: list[RefinementTraceRecord] = []
        for iteration in range(steps):
            current, record = stepper(
                current,
                sources,
                selected.head_index,
                iteration,
            )
            injected_records.append(record)
        final_logits = current.detach()
        records = tuple(injected_records)

    validate_refinement_accounting(
        refined_heads=(selected.head_index,),
        requested_steps=steps,
        record_count=len(records),
    )
    if any(
        record.head_index != selected.head_index or record.iteration != iteration
        for iteration, record in enumerate(records)
    ):
        raise MT3InvalidRun("refinement trace contains another head or iteration")
    final_parameterized = parameterize_design(final_logits, beta=8.0)
    continuous = final_parameterized.design.detach()
    binary, _ = exact_cardinality_binary(continuous, count=1024)
    return MT3RefinementResult(
        selected_head=selected.head_index,
        candidate_scores=(selected,),
        requested_steps=steps,
        refined_heads=(selected.head_index,),
        records=records,
        final_logits=final_logits.cpu(),
        continuous_design=continuous.cpu(),
        binary_design=binary.cpu(),
    )


def select_and_refine(
    candidate_logits: Tensor,
    binary_designs: Tensor,
    task: SourceLayoutTask,
    sources: Tensor,
    *,
    scorer: SciPy64Evaluator,
    steps: int,
    allow_cpu_unit_test: bool = False,
    stepper: RefinementStepper | None = None,
) -> MT3RefinementResult:
    """Run four forward-only scores, then one and only one refinement chain."""
    scores = _score_candidates(candidate_logits, binary_designs, task, scorer)
    selected = min(scores, key=lambda row: (row.binary_tmax, row.head_index))
    result = refine_selected_candidate(
        selected,
        candidate_logits,
        sources,
        steps=steps,
        allow_cpu_unit_test=allow_cpu_unit_test,
        stepper=stepper,
    )
    return replace(result, candidate_scores=scores)


def _result_from_logits(
    *,
    selected: CandidateScore,
    scores: tuple[CandidateScore, ...],
    steps: int,
    records: tuple[RefinementTraceRecord, ...],
    logits: Tensor,
) -> MT3RefinementResult:
    validate_refinement_accounting(
        refined_heads=(selected.head_index,),
        requested_steps=steps,
        record_count=len(records),
    )
    parameterized = parameterize_design(logits, beta=8.0)
    continuous = parameterized.design.detach()
    binary, _ = exact_cardinality_binary(continuous, count=1024)
    return MT3RefinementResult(
        selected_head=selected.head_index,
        candidate_scores=scores,
        requested_steps=steps,
        refined_heads=(selected.head_index,),
        records=records,
        final_logits=logits.detach().cpu(),
        continuous_design=continuous.cpu(),
        binary_design=binary.cpu(),
    )


def select_and_refine_trajectory(
    candidate_logits: Tensor,
    binary_designs: Tensor,
    task: SourceLayoutTask,
    sources: Tensor,
    *,
    scorer: SciPy64Evaluator,
    steps: tuple[int, ...] = (25, 50),
    allow_cpu_unit_test: bool = False,
    stepper: RefinementStepper | None = None,
) -> dict[int, MT3RefinementResult]:
    """Capture registered R25 and R50 from one selected-candidate chain."""
    if steps not in ((25,), (50,), (25, 50)):
        raise ValueError("trajectory steps must be (25,), (50,), or (25,50)")
    if tuple(candidate_logits.shape) != (4, 64, 64):
        raise ValueError("candidate logits must have shape [4,64,64]")
    if tuple(sources.shape) != (3, 64, 64) or sources.dtype is not torch.float64:
        raise ValueError("sources must have shape [3,64,64] and float64")
    if sources.device.type != "cuda" and not allow_cpu_unit_test:
        raise ValueError("production refinement requires CUDA")

    scores = _score_candidates(candidate_logits, binary_designs, task, scorer)
    selected = min(scores, key=lambda row: (row.binary_tmax, row.head_index))
    current = candidate_logits[selected.head_index].to(device=sources.device)
    maximum = max(steps)
    snapshots: dict[int, Tensor] = {}
    if stepper is None:
        _, records, snapshots = _run_physics_refinement(
            selected,
            current,
            sources,
            steps=maximum,
            capture_steps=steps,
        )
    else:
        injected: list[RefinementTraceRecord] = []
        for iteration in range(maximum):
            current, record = stepper(
                current,
                sources,
                selected.head_index,
                iteration,
            )
            injected.append(record)
            completed = iteration + 1
            if completed in steps:
                snapshots[completed] = current.detach().clone()
        records = tuple(injected)
    if set(snapshots) != set(steps):
        raise MT3InvalidRun("refinement trajectory is missing a registered snapshot")
    return {
        step: _result_from_logits(
            selected=selected,
            scores=scores,
            steps=step,
            records=records[:step],
            logits=snapshots[step],
        )
        for step in steps
    }
