"""Frozen MT2B verdict logic and solver-independent result records."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from waveforge.design.binary_readout import exact_cardinality_binary
from waveforge.ml.mt2b_conditioning import build_mt2b_conditioning
from waveforge.ml.mt2b_evaluation import BootstrapResult, MT2BCheckpointSummary
from waveforge.ml.mt2b_nca import MT2BNCA
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.ml.nca import project_nca_material
from waveforge.ml.nca_training import model_state_sha256
from waveforge.physics.fixed_operator import UniformPlateFactorization

MT2BStatus = Literal[
    "PHYSICS_VERY_STRONG_GO",
    "PHYSICS_GO",
    "PHYSICS_CONDITIONAL_GO",
    "PHYSICS_NO_GO",
    "MT2B_INVALID_RUN",
]


@dataclass(frozen=True)
class MT2BFinalVerdict:
    status: MT2BStatus
    paired_win_count: int
    median_paired_gap_reduction: float
    paired_win_gate: bool
    median_reduction_gate: bool
    bootstrap_gate: bool
    physics_median_gap: float
    physics_p90_gap: float
    exact_reason: str


@dataclass(frozen=True)
class FrozenMT2BDesign:
    task_id: str
    continuous_design: NDArray[np.float64]
    binary_design: NDArray[np.float64]
    binary_material_fraction: float


@dataclass(frozen=True)
class FrozenMT2BBatch:
    variant: Literal["RAW", "PHYSICS"]
    completed_updates: int
    model_state_sha256: str
    designs: tuple[FrozenMT2BDesign, ...]


def _load_model(checkpoint: str | bytes | PathLike[str], device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported MT2B checkpoint schema")
    model = MT2BNCA().to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    observed = model_state_sha256(model)
    if payload.get("model_state_sha256") != observed:
        raise ValueError("MT2B checkpoint model hash mismatch")
    return model, int(payload["completed_updates"]), observed


def _conditioning_solver(
    factorization: UniformPlateFactorization,
    sources: NDArray[np.float64],
) -> NDArray[np.float64]:
    batch, scenarios, ny, nx = sources.shape
    result = factorization.solve_many(sources.reshape(batch * scenarios, ny, nx))
    if result.maximum_normalized_residual > 1.0e-10:
        raise FloatingPointError("MT2B conditioning residual exceeds tolerance")
    return result.temperature.reshape(sources.shape)


def generate_mt2b_designs(
    checkpoint: str | bytes | PathLike[str],
    tasks: tuple[SourceLayoutTask, ...],
    *,
    variant: Literal["RAW", "PHYSICS"],
    device: torch.device,
    conditioning_tasks: tuple[SourceLayoutTask, ...] | None = None,
    batch_size: int = 8,
) -> FrozenMT2BBatch:
    """Run frozen weights only and return exact-budget designs for each task."""
    if not tasks:
        raise ValueError("frozen MT2B generation requires at least one task")
    if variant not in {"RAW", "PHYSICS"}:
        raise ValueError("MT2B generation variant must be RAW or PHYSICS")
    if batch_size < 1:
        raise ValueError("MT2B generation batch_size must be positive")
    conditions = tasks if conditioning_tasks is None else conditioning_tasks
    if len(conditions) != len(tasks):
        raise ValueError("conditioning task count must equal physical task count")
    model, completed, model_hash = _load_model(checkpoint, device)
    factorization = (
        UniformPlateFactorization(grid_size=64, conductivity=1.0)
        if variant == "PHYSICS"
        else None
    )
    generated: list[FrozenMT2BDesign] = []
    with torch.no_grad():
        for start in range(0, len(tasks), batch_size):
            task_chunk = tasks[start : start + batch_size]
            condition_chunk = conditions[start : start + batch_size]
            source_array = np.stack([task.sources for task in condition_chunk])
            sources = torch.as_tensor(source_array, dtype=torch.float64, device=device)
            temperature_solver = None
            if factorization is not None:

                def solve(array: NDArray[np.float64]) -> NDArray[np.float64]:
                    return _conditioning_solver(factorization, array)

                temperature_solver = solve
            condition = build_mt2b_conditioning(
                sources,
                variant=variant,
                temperature_solver=temperature_solver,
            )
            rollout = model.rollout(condition, steps=64)
            for local_index, task in enumerate(task_chunk):
                one: Tensor = project_nca_material(
                    rollout.material_logit[local_index : local_index + 1], beta=8.0
                ).design
                binary, budget = exact_cardinality_binary(one, count=1024)
                generated.append(
                    FrozenMT2BDesign(
                        task_id=task.task_id,
                        continuous_design=one.cpu()
                        .numpy()
                        .astype(np.float64, copy=False),
                        binary_design=binary.cpu()
                        .numpy()
                        .astype(np.float64, copy=False),
                        binary_material_fraction=budget.material_fraction,
                    )
                )
    return FrozenMT2BBatch(
        variant=variant,
        completed_updates=completed,
        model_state_sha256=model_hash,
        designs=tuple(generated),
    )


def classify_mt2b_result(
    *,
    physics_summary: MT2BCheckpointSummary,
    raw_gaps: NDArray[np.float64],
    physics_gaps: NDArray[np.float64],
    bootstrap: BootstrapResult,
) -> MT2BFinalVerdict:
    """Apply the immutable paired-effect and absolute-quality MT2B gates."""
    raw = np.asarray(raw_gaps, dtype=np.float64)
    physics = np.asarray(physics_gaps, dtype=np.float64)
    if raw.shape != (32,) or physics.shape != (32,):
        raise ValueError("MT2B verdict requires exactly 32 paired validation gaps")
    if not np.isfinite(np.column_stack((raw, physics))).all():
        raise ValueError("MT2B verdict gaps must be finite")
    deltas = raw - physics
    wins = int(np.count_nonzero(deltas > 0.0))
    median_delta = float(np.median(deltas))
    win_gate = wins >= 24
    reduction_gate = median_delta >= 0.03
    bootstrap_gate = bootstrap.conditioning_ci_pass and bootstrap.lower_bound > 0.0

    if physics_summary.invalid_count != 0 or physics_summary.task_count != 32:
        status: MT2BStatus = "MT2B_INVALID_RUN"
        reason = "invalid or incomplete PHYSICS validation records"
    elif not (win_gate and reduction_gate and bootstrap_gate):
        status = "PHYSICS_NO_GO"
        reason = "paired conditioning-effect evidence did not pass every locked gate"
    elif (
        physics_summary.median_relative_gap <= 0.05
        and physics_summary.p90_relative_gap <= 0.10
    ):
        status = "PHYSICS_VERY_STRONG_GO"
        reason = "paired effect and very-strong absolute quality thresholds passed"
    elif (
        physics_summary.median_relative_gap <= 0.10
        and physics_summary.p90_relative_gap <= 0.20
    ):
        status = "PHYSICS_GO"
        reason = "paired effect and primary absolute quality thresholds passed"
    elif (
        physics_summary.median_relative_gap <= 0.15
        and physics_summary.p90_relative_gap <= 0.30
    ):
        status = "PHYSICS_CONDITIONAL_GO"
        reason = "paired effect and conditional absolute quality thresholds passed"
    else:
        status = "PHYSICS_NO_GO"
        reason = "PHYSICS absolute quality exceeded the locked conditional limits"

    return MT2BFinalVerdict(
        status=status,
        paired_win_count=wins,
        median_paired_gap_reduction=median_delta,
        paired_win_gate=win_gate,
        median_reduction_gate=reduction_gate,
        bootstrap_gate=bootstrap_gate,
        physics_median_gap=physics_summary.median_relative_gap,
        physics_p90_gap=physics_summary.p90_relative_gap,
        exact_reason=reason,
    )
