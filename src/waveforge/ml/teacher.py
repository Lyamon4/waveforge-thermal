"""Fail-closed teacher optimization and independent cost-pilot verification."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from waveforge.design.constraints import binary_budget_satisfied, material_fraction
from waveforge.design.differentiable_solver import SolveTrace, solve_steady_implicit
from waveforge.design.objectives import objective_components
from waveforge.design.optimize import (
    OptimizationConfig,
    alpha_for_iteration,
    array_sha256,
    beta_for_iteration,
    binarization_weight_for_iteration,
    initialize_logits,
    optimize_design,
)
from waveforge.design.parameterization import (
    VolumeProjectionError,
    binary_design,
    parameterize_design,
)
from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.cg import CGConfig, CGConvergenceError
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import (
    assemble_steady_system,
    factorize_system,
    solve_factorized,
)
from waveforge.verification.compare import Gate2Status

TeacherMode = Literal["unit", "production"]


class TeacherStatus(StrEnum):
    """Machine-readable teacher outcomes with invalid-run separation."""

    PASS = "PASS"
    NO_GO_BUDGET = "NO_GO_BUDGET"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class TeacherConfig:
    """Immutable Stage C teacher settings."""

    resolution: int
    iterations: int
    mode: TeacherMode = "production"
    learning_rate: float = 0.05
    gradient_clip_norm: float = 1.0
    enforce_final_binary_budget: bool = True
    device: str = "cuda"
    cg_config: CGConfig = field(default_factory=CGConfig)

    def __post_init__(self) -> None:
        if self.mode == "production" and (self.resolution, self.iterations) not in {
            (32, 200),
            (64, 600),
        }:
            raise ValueError("production teacher must use a locked protocol")
        if self.mode == "unit" and self.iterations != 1:
            raise ValueError("unit teacher is locked to one iteration")
        if self.resolution not in (32, 64):
            raise ValueError("teacher resolution must be 32 or 64")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("teacher optimizer settings must be positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("teacher device must be cpu or cuda")


@dataclass(frozen=True)
class TeacherIterationRecord:
    """Scientific and numerical terms for one teacher update."""

    iteration: int
    beta: float
    alpha: float
    binarization_weight: float
    total_objective: float
    exact_peak: float
    continuous_material_fraction: float
    binary_material_fraction: float
    gradient_norm_before_clipping: float
    maximum_cg_iterations: int
    maximum_residual: float
    wall_seconds: float


@dataclass(frozen=True)
class TeacherResult:
    """One complete or fail-closed teacher optimization result."""

    status: TeacherStatus
    reason_codes: tuple[str, ...]
    seed: int
    centers: tuple[tuple[float, float], ...]
    resolution: int
    completed_iterations: int
    records: tuple[TeacherIterationRecord, ...]
    initial_logits: Tensor
    final_logits: Tensor
    continuous_design: Tensor | None
    binary_design: Tensor | None
    continuous_material_fraction: float | None
    binary_material_fraction: float | None
    total_wall_seconds: float


@dataclass(frozen=True)
class TeacherVerification:
    """Independent `64×64` SciPy verification of one frozen teacher design."""

    source_resolution: int
    transferred_design: NDArray[np.float64]
    material_fraction: float
    scenario_peaks: tuple[float, float, float]
    worst_peak: float
    maximum_residual: float


def teacher_schedule(
    iteration: int,
    *,
    resolution: int,
) -> tuple[float, float, float]:
    """Return beta, alpha and binary weight for one locked teacher step."""
    if resolution == 64:
        return (
            beta_for_iteration(iteration),
            alpha_for_iteration(iteration),
            binarization_weight_for_iteration(iteration),
        )
    if resolution != 32 or not 0 <= iteration < 200:
        raise ValueError("reduced teacher iteration must lie in [0,199]")
    if iteration <= 66:
        return (1.0, 50.0, 0.0)
    if iteration <= 116:
        return (2.0, 200.0, 0.005)
    if iteration <= 166:
        return (4.0, 500.0, 0.01)
    return (8.0, 500.0, 0.02)


def _validate_centers(
    centers: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    if len(centers) != 3 or centers != tuple(sorted(centers)):
        raise ValueError("teacher centers must be a canonical sorted triple")
    if not all(
        len(center) == 2
        and all(math.isfinite(value) for value in center)
        and 0.1 <= center[0] <= 0.9
        and 0.1 <= center[1] <= 0.9
        for center in centers
    ):
        raise ValueError("teacher source rectangles must lie inside the domain")
    if not all(
        math.dist(left, right) >= 0.2 - 1.0e-12
        for index, left in enumerate(centers)
        for right in centers[index + 1 :]
    ):
        raise ValueError("teacher centers violate locked minimum separation")
    return centers


def teacher_source_batch(
    centers: tuple[tuple[float, float], ...],
    *,
    resolution: int,
    device: torch.device,
) -> Tensor:
    """Rasterize three canonical equal-power source rectangles independently."""
    selected_centers = _validate_centers(centers)
    if resolution not in (32, 64):
        raise ValueError("teacher source resolution must be 32 or 64")
    grid = Grid2D(nx=resolution, ny=resolution)
    sources = np.stack(
        [
            area_overlap_rectangular_source(
                grid,
                (x - 0.1, x + 0.1, y - 0.1, y + 0.1),
                1.0,
            )
            for x, y in selected_centers
        ]
    )
    return torch.as_tensor(sources, dtype=torch.float64, device=device)


def _status_from_budget(
    binary: Tensor, enforce: bool
) -> tuple[TeacherStatus, tuple[str, ...]]:
    if enforce and not binary_budget_satisfied(binary):
        return TeacherStatus.NO_GO_BUDGET, ("BINARY_BUDGET_FAILURE",)
    return TeacherStatus.PASS, ()


def _failed_result(
    *,
    seed: int,
    centers: tuple[tuple[float, float], ...],
    resolution: int,
    records: list[TeacherIterationRecord],
    initial_logits: Tensor,
    logits: Tensor,
    reason_code: str,
    total_wall_seconds: float,
) -> TeacherResult:
    return TeacherResult(
        status=TeacherStatus.INVALID_RUN,
        reason_codes=(reason_code,),
        seed=seed,
        centers=centers,
        resolution=resolution,
        completed_iterations=len(records),
        records=tuple(records),
        initial_logits=initial_logits.detach().cpu(),
        final_logits=logits.detach().cpu(),
        continuous_design=None,
        binary_design=None,
        continuous_material_fraction=None,
        binary_material_fraction=None,
        total_wall_seconds=total_wall_seconds,
    )


def _optimize_generic(
    centers: tuple[tuple[float, float], ...],
    *,
    seed: int,
    config: TeacherConfig,
) -> TeacherResult:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage C teacher requires available CUDA")
    sources = teacher_source_batch(
        centers,
        resolution=config.resolution,
        device=device,
    )
    grid = Grid2D(nx=config.resolution, ny=config.resolution)
    logits = initialize_logits(seed, device=device).requires_grad_(True)
    initial_logits = logits.detach().clone()
    optimizer = torch.optim.Adam([logits], lr=config.learning_rate)
    records: list[TeacherIterationRecord] = []
    total_started = time.perf_counter()
    try:
        for iteration in range(config.iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            beta, alpha, binary_weight = teacher_schedule(
                iteration,
                resolution=config.resolution,
            )
            parameterized = parameterize_design(
                logits,
                beta=beta,
                simulation_shape=grid.shape,
            )
            design = parameterized.design
            conductivity = 1.0 + 19.0 * design.to(torch.float64) ** 3
            trace = SolveTrace()
            temperatures = solve_steady_implicit(
                conductivity,
                sources,
                grid,
                config=config.cg_config,
                trace=trace,
            )
            components = objective_components(
                temperatures,
                design,
                alpha=alpha,
                tv_weight=0.001,
                binarization_weight=binary_weight,
            )
            components.total.backward()
            if logits.grad is None or not torch.isfinite(logits.grad).all():
                raise FloatingPointError("teacher gradient is missing or non-finite")
            if logits.grad.dtype is not torch.float32:
                raise FloatingPointError("teacher design gradient must be float32")
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [logits],
                    config.gradient_clip_norm,
                    error_if_nonfinite=True,
                ).item()
            )
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            solve_records = tuple(trace.records)
            binary = binary_design(design.detach())
            records.append(
                TeacherIterationRecord(
                    iteration=iteration,
                    beta=beta,
                    alpha=alpha,
                    binarization_weight=binary_weight,
                    total_objective=float(components.total.detach().item()),
                    exact_peak=float(components.exact_peak.detach().item()),
                    continuous_material_fraction=material_fraction(design.detach()),
                    binary_material_fraction=material_fraction(binary),
                    gradient_norm_before_clipping=gradient_norm,
                    maximum_cg_iterations=max(
                        record.iterations for record in solve_records
                    ),
                    maximum_residual=max(
                        record.relative_residual for record in solve_records
                    ),
                    wall_seconds=elapsed,
                )
            )
    except CGConvergenceError:
        return _failed_result(
            seed=seed,
            centers=centers,
            resolution=config.resolution,
            records=records,
            initial_logits=initial_logits,
            logits=logits,
            reason_code="CG_NONCONVERGENCE",
            total_wall_seconds=time.perf_counter() - total_started,
        )
    except (FloatingPointError, VolumeProjectionError, ValueError) as error:
        return _failed_result(
            seed=seed,
            centers=centers,
            resolution=config.resolution,
            records=records,
            initial_logits=initial_logits,
            logits=logits,
            reason_code=f"NUMERICAL_FAILURE:{type(error).__name__}",
            total_wall_seconds=time.perf_counter() - total_started,
        )

    final_beta = teacher_schedule(
        config.iterations - 1,
        resolution=config.resolution,
    )[0]
    continuous = parameterize_design(
        logits.detach(),
        beta=final_beta,
        simulation_shape=grid.shape,
    ).design.detach()
    binary = binary_design(continuous)
    status, reason_codes = _status_from_budget(
        binary,
        config.enforce_final_binary_budget,
    )
    return TeacherResult(
        status=status,
        reason_codes=reason_codes,
        seed=seed,
        centers=centers,
        resolution=config.resolution,
        completed_iterations=len(records),
        records=tuple(records),
        initial_logits=initial_logits.detach().cpu(),
        final_logits=logits.detach().cpu(),
        continuous_design=continuous.cpu(),
        binary_design=binary.cpu(),
        continuous_material_fraction=material_fraction(continuous),
        binary_material_fraction=material_fraction(binary),
        total_wall_seconds=time.perf_counter() - total_started,
    )


def _wrap_gate2_result(
    result: object,
    centers: tuple[tuple[float, float], ...],
) -> TeacherResult:
    gate2_result = result
    status = (
        TeacherStatus.PASS
        if gate2_result.status is Gate2Status.PASS
        else TeacherStatus.INVALID_RUN
        if gate2_result.status is Gate2Status.INVALID_RUN
        else TeacherStatus.NO_GO_BUDGET
    )
    records = tuple(
        TeacherIterationRecord(
            iteration=record.iteration,
            beta=record.beta,
            alpha=record.alpha,
            binarization_weight=record.binarization_weight,
            total_objective=record.total_objective,
            exact_peak=record.exact_peak,
            continuous_material_fraction=record.continuous_material_fraction,
            binary_material_fraction=record.binary_material_fraction,
            gradient_norm_before_clipping=record.gradient_norm_before_clipping,
            maximum_cg_iterations=record.maximum_cg_iterations,
            maximum_residual=record.maximum_explicit_relative_residual,
            wall_seconds=record.wall_seconds,
        )
        for record in gate2_result.records
    )
    return TeacherResult(
        status=status,
        reason_codes=gate2_result.reason_codes,
        seed=gate2_result.seed,
        centers=centers,
        resolution=64,
        completed_iterations=gate2_result.completed_iterations,
        records=records,
        initial_logits=gate2_result.initial_logits,
        final_logits=gate2_result.final_logits,
        continuous_design=gate2_result.continuous_design,
        binary_design=gate2_result.binary_design,
        continuous_material_fraction=gate2_result.continuous_material_fraction,
        binary_material_fraction=gate2_result.binary_material_fraction,
        total_wall_seconds=sum(record.wall_seconds for record in records),
    )


def _write_teacher_artifacts(result: TeacherResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "optimization_metrics.csv"
    rows = [asdict(record) for record in result.records]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": result.status.value,
        "reason_codes": list(result.reason_codes),
        "seed": result.seed,
        "centers": [list(center) for center in result.centers],
        "resolution": result.resolution,
        "completed_iterations": result.completed_iterations,
        "initial_logits_sha256": array_sha256(result.initial_logits),
        "final_logits_sha256": array_sha256(result.final_logits),
        "continuous_material_fraction": result.continuous_material_fraction,
        "binary_material_fraction": result.binary_material_fraction,
        "total_wall_seconds": result.total_wall_seconds,
    }
    if result.continuous_design is not None and result.binary_design is not None:
        continuous_path = output_dir / f"design_continuous_{result.resolution}.npy"
        binary_path = output_dir / f"design_binary_{result.resolution}.npy"
        np.save(continuous_path, result.continuous_design.numpy(), allow_pickle=False)
        np.save(binary_path, result.binary_design.numpy(), allow_pickle=False)
        payload["continuous_design_sha256"] = array_sha256(result.continuous_design)
        payload["binary_design_sha256"] = array_sha256(result.binary_design)
    (output_dir / "teacher_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def optimize_teacher(
    centers: tuple[tuple[float, float], ...],
    *,
    seed: int,
    config: TeacherConfig,
    output_dir: Path | None,
) -> TeacherResult:
    """Optimize one locked teacher without modifying Gate 2A implementation."""
    selected_centers = _validate_centers(centers)
    if config.resolution == 64 and config.mode == "production":
        device = torch.device(config.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("locked 64x64 teacher requires CUDA")
        sources = teacher_source_batch(
            selected_centers,
            resolution=64,
            device=device,
        )
        gate2_result = optimize_design(
            sources,
            seed=seed,
            config=OptimizationConfig(),
            output_dir=None,
        )
        result = _wrap_gate2_result(gate2_result, selected_centers)
    else:
        result = _optimize_generic(
            selected_centers,
            seed=seed,
            config=config,
        )
    if output_dir is not None:
        _write_teacher_artifacts(result, output_dir)
    return result


def verify_teacher_at_64(
    centers: tuple[tuple[float, float], ...],
    design: NDArray[np.float64],
    *,
    source_resolution: int,
) -> TeacherVerification:
    """Verify a frozen strict-binary teacher with independent SciPy physics."""
    selected_centers = _validate_centers(centers)
    frozen = np.asarray(design, dtype=np.float64)
    expected_shape = (source_resolution, source_resolution)
    if source_resolution not in (32, 64) or frozen.shape != expected_shape:
        raise ValueError("teacher verification design shape does not match resolution")
    if not np.all((frozen == 0.0) | (frozen == 1.0)):
        raise ValueError("teacher verification requires strict binary design")
    transferred = (
        np.repeat(np.repeat(frozen, 2, axis=0), 2, axis=1)
        if source_resolution == 32
        else frozen.copy()
    )
    grid = Grid2D(nx=64, ny=64)
    source_batch = teacher_source_batch(
        selected_centers,
        resolution=64,
        device=torch.device("cpu"),
    ).numpy()
    conductivity = 1.0 + 19.0 * transferred**3
    first = assemble_steady_system(
        grid,
        conductivity,
        source_batch[0],
        BoundaryConditions.production(),
    )
    factorization = factorize_system(first)
    peaks: list[float] = []
    residuals: list[float] = []
    for source in source_batch:
        source_rhs = source.ravel().copy()
        system = replace(
            first,
            source_rhs=source_rhs,
            rhs=source_rhs + first.dirichlet_rhs,
        )
        solved = solve_factorized(system, factorization)
        peaks.append(float(np.max(solved.temperature)))
        residuals.append(solved.normalized_residual)
    return TeacherVerification(
        source_resolution=source_resolution,
        transferred_design=transferred,
        material_fraction=float(np.mean(transferred)),
        scenario_peaks=(peaks[0], peaks[1], peaks[2]),
        worst_peak=max(peaks),
        maximum_residual=max(residuals),
    )
