"""Independent direct-gradient Adam baselines sharing one batched physics solve."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

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
    initialize_logits,
)
from waveforge.design.parameterization import parameterize_design
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.physics.grid import Grid2D


@dataclass(frozen=True)
class AdamSnapshot:
    update: int
    binary_design: NDArray[np.float64]
    binary_cell_count: int
    binary_material_fraction: float


@dataclass(frozen=True)
class AdamTaskResult:
    task_id: str
    seed: int
    final_logits: NDArray[np.float32]
    snapshots: dict[int, AdamSnapshot]


@dataclass(frozen=True)
class BatchedAdamResult:
    completed_updates: int
    snapshot_updates: tuple[int, ...]
    tasks: tuple[AdamTaskResult, ...]


def taskwise_clip_gradients_(gradients: Tensor, *, max_norm: float) -> Tensor:
    """Clip every task slice independently and return its unclipped norm."""
    if gradients.ndim < 2 or not gradients.is_floating_point():
        raise ValueError("task gradients must be a batched floating tensor")
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("maximum gradient norm must be finite and positive")
    if not bool(torch.isfinite(gradients).all().item()):
        raise FloatingPointError("task gradients contain NaN or Inf")
    norms = torch.linalg.vector_norm(gradients.reshape(gradients.shape[0], -1), dim=1)
    scales = torch.clamp(max_norm / torch.clamp(norms, min=1.0e-30), max=1.0)
    gradients.mul_(scales.reshape((-1,) + (1,) * (gradients.ndim - 1)))
    return norms


def _parameterize_batch(logits: Tensor, *, beta: float) -> Tensor:
    return torch.stack(
        [parameterize_design(item, beta=beta).design for item in logits], dim=0
    )


def _validate_trace(trace: BatchedSolveTrace, *, batch_size: int) -> None:
    expected = ["forward"] * (3 * batch_size) + ["adjoint"] * (3 * batch_size)
    if [record.role for record in trace.records] != expected:
        raise RuntimeError("batched Adam physics trace is incomplete")
    if any(
        (not record.converged) or record.relative_residual > 1.0e-6
        for record in trace.records
    ):
        raise RuntimeError("batched Adam physics residual exceeds tolerance")


def _snapshot(logits: Tensor, *, update: int) -> AdamSnapshot:
    design = parameterize_design(logits, beta=beta_for_iteration(update - 1)).design
    binary, diagnostics = exact_cardinality_binary(design, count=1024)
    array = binary.detach().cpu().numpy().astype(np.float64, copy=False)
    return AdamSnapshot(
        update=update,
        binary_design=array,
        binary_cell_count=diagnostics.selected_cells,
        binary_material_fraction=diagnostics.material_fraction,
    )


def optimize_adam_batched(
    tasks: tuple[SourceLayoutTask, ...],
    *,
    seeds: tuple[int, ...],
    total_updates: int = 600,
    snapshot_updates: tuple[int, ...] = (25, 50, 100, 200, 600),
    device: torch.device | None = None,
    allow_cpu_unit_test: bool = False,
) -> BatchedAdamResult:
    """Optimize independent tasks while vectorizing only their thermal solves."""
    if not tasks or len(tasks) != len(seeds):
        raise ValueError("tasks and seeds must be non-empty and have equal length")
    if allow_cpu_unit_test:
        if not 1 <= total_updates <= 600:
            raise ValueError("unit-test Adam updates must lie in [1,600]")
    elif total_updates != 600:
        raise ValueError("production batched Adam requires exactly 600 updates")
    if (
        not snapshot_updates
        or tuple(sorted(set(snapshot_updates))) != snapshot_updates
        or snapshot_updates[-1] > total_updates
        or snapshot_updates[0] < 1
    ):
        raise ValueError("snapshot updates must be ordered, unique, and within budget")
    target_device = device or torch.device("cpu" if allow_cpu_unit_test else "cuda")
    if target_device.type != "cuda" and not allow_cpu_unit_test:
        raise ValueError("production batched Adam requires CUDA")

    logits = torch.stack(
        [initialize_logits(seed, device=target_device) for seed in seeds], dim=0
    ).requires_grad_(True)
    sources = torch.as_tensor(
        np.stack([task.sources for task in tasks]),
        dtype=torch.float64,
        device=target_device,
    )
    optimizer = torch.optim.Adam(
        [logits], lr=0.05, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0
    )
    stored: list[dict[int, AdamSnapshot]] = [dict() for _ in tasks]
    grid = Grid2D(nx=64, ny=64)
    wanted = frozenset(snapshot_updates)

    for iteration in range(total_updates):
        optimizer.zero_grad(set_to_none=True)
        designs = _parameterize_batch(logits, beta=beta_for_iteration(iteration))
        conductivity = 1.0 + 19.0 * designs.to(torch.float64).pow(3)
        trace = BatchedSolveTrace()
        temperatures = solve_steady_implicit_batched(
            conductivity, sources, grid, trace=trace
        )
        losses = [
            objective_components(
                temperatures[index],
                designs[index],
                alpha=alpha_for_iteration(iteration),
                tv_weight=1.0e-3,
                binarization_weight=binarization_weight_for_iteration(iteration),
            ).total
            for index in range(len(tasks))
        ]
        torch.stack(losses).sum().backward()
        _validate_trace(trace, batch_size=len(tasks))
        if logits.grad is None:
            raise FloatingPointError("batched Adam gradient is missing")
        taskwise_clip_gradients_(logits.grad, max_norm=1.0)
        optimizer.step()

        update = iteration + 1
        if update in wanted:
            for index in range(len(tasks)):
                stored[index][update] = _snapshot(logits[index].detach(), update=update)

    final = logits.detach().cpu().numpy().astype(np.float32, copy=False)
    results = tuple(
        AdamTaskResult(
            task_id=task.task_id,
            seed=seed,
            final_logits=final[index],
            snapshots=stored[index],
        )
        for index, (task, seed) in enumerate(zip(tasks, seeds, strict=True))
    )
    return BatchedAdamResult(
        completed_updates=total_updates,
        snapshot_updates=snapshot_updates,
        tasks=results,
    )
