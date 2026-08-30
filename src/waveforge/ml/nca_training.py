"""Fail-closed differentiable training path for the locked pure NCA."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from waveforge.design.differentiable_solver import (
    SolveRecord,
    SolveTrace,
    solve_steady_implicit,
)
from waveforge.design.objectives import ObjectiveComponents, objective_components
from waveforge.design.parameterization import (
    ProjectionDiagnostics,
    VolumeProjectionError,
    binary_design,
)
from waveforge.ml.nca import (
    NCARollout,
    PureNCA,
    build_static_condition,
    project_nca_material,
)
from waveforge.physics.cg import CGConvergenceError
from waveforge.physics.grid import Grid2D
from waveforge.reproducibility import set_deterministic_seed

NCARunMode = Literal["unit", "benchmark", "smoke", "qualification", "production"]


class NCARunStatus(StrEnum):
    PASS = "PASS"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class NCAForwardResult:
    """Full differentiable forward and separately retained diagnostics."""

    rollout: NCARollout
    continuous_design: Tensor
    binary_design: Tensor | None
    temperatures: Tensor
    objective: ObjectiveComponents
    projection: ProjectionDiagnostics
    solve_trace: SolveTrace


@dataclass(frozen=True)
class NCAIterationRecord:
    """All preregistered numerical and scientific diagnostics for one update."""

    iteration: int
    total_objective: float
    thermal_smooth: float
    exact_continuous_tmax: float
    total_variation: float
    binarization_penalty: float
    continuous_material_fraction: float
    binary_material_fraction: float
    projection_absolute_error: float
    material_logit_mean: float
    material_logit_std: float
    material_logit_minimum: float
    material_logit_maximum: float
    material_std: float
    hidden_state_rms: float
    delta_state_rms: float
    maximum_absolute_delta: float
    maximum_absolute_state: float
    gradient_norm_before_clipping: float
    gradient_norm_after_clipping: float
    conv3x3_weight_gradient_norm: float
    conv1x1_weight_gradient_norm: float
    all_parameter_gradients_finite: bool
    maximum_cg_iterations: int
    maximum_explicit_relative_residual: float
    all_cg_converged: bool
    finite: bool
    wall_seconds: float


@dataclass(frozen=True)
class NCARunResult:
    """One fixed-length training outcome without scientific interpretation."""

    status: NCARunStatus
    reason_codes: tuple[str, ...]
    seed: int
    mode: NCARunMode
    learning_rate: float
    requested_iterations: int
    completed_iterations: int
    initial_objective: float | None
    records: tuple[NCAIterationRecord, ...]
    solve_records: tuple[SolveRecord, ...]
    initial_model_hash: str
    final_model_hash: str
    final_continuous_design: Tensor | None
    final_binary_design: Tensor | None


class _TrainingInvalidError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


NCAEvaluator = Callable[..., NCAForwardResult]


def _model_device_and_dtype(model: PureNCA) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def evaluate_nca(
    model: PureNCA,
    sources: Tensor,
    *,
    trace: SolveTrace | None = None,
    allow_cpu_unit_test: bool = False,
    include_binary_diagnostic: bool = False,
    snapshot_steps: tuple[int, ...] = (),
    projection_beta: float = 8.0,
    smooth_max_alpha: float = 500.0,
    tv_weight: float = 1.0e-3,
    binarization_weight: float = 0.02,
) -> NCAForwardResult:
    """Evaluate the fixed NCA, projection, steady physics and objective."""
    device, dtype = _model_device_and_dtype(model)
    if dtype is not torch.float32:
        raise ValueError("NCA model parameters must be float32")
    if device.type != "cuda" and not allow_cpu_unit_test:
        raise ValueError("pure-NCA training physics requires CUDA")
    if sources.shape != (3, 64, 64) or sources.dtype is not torch.float64:
        raise ValueError("sources must have shape [3,64,64] and dtype float64")
    if sources.device != device:
        raise ValueError("sources and NCA model must share a device")
    if not torch.isfinite(sources).all():
        raise ValueError("sources must be finite")

    condition = build_static_condition(sources)
    rollout = model.rollout(condition, snapshot_steps=snapshot_steps)
    projected = project_nca_material(rollout.material_logit, beta=projection_beta)
    conductivity = 1.0 + 19.0 * projected.design.to(torch.float64).pow(3)
    solve_trace = trace if trace is not None else SolveTrace()
    temperatures = solve_steady_implicit(
        conductivity,
        sources,
        Grid2D(nx=64, ny=64),
        trace=solve_trace,
    )
    objective = objective_components(
        temperatures,
        projected.design,
        alpha=smooth_max_alpha,
        tv_weight=tv_weight,
        binarization_weight=binarization_weight,
    )
    diagnostic_binary: Tensor | None = None
    if include_binary_diagnostic:
        with torch.no_grad():
            diagnostic_binary = binary_design(projected.design)
    return NCAForwardResult(
        rollout=rollout,
        continuous_design=projected.design,
        binary_design=diagnostic_binary,
        temperatures=temperatures,
        objective=objective,
        projection=projected.projection,
        solve_trace=solve_trace,
    )


def initialize_nca(seed: int, device: torch.device) -> PureNCA:
    """Create one reproducible float32 NCA model on the requested device."""
    set_deterministic_seed(seed)
    return PureNCA().to(device=device, dtype=torch.float32)


def model_state_sha256(model: PureNCA) -> str:
    """Hash every named model tensor by dtype, shape and row-major bytes."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _gradient_norm(parameters: list[Tensor]) -> float:
    squared = sum(
        float(torch.sum(parameter.grad.detach().double().pow(2)).item())
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def _weight_gradient_norm(parameter: Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(parameter.grad.detach()).item())


def _validate_forward(forward: NCAForwardResult) -> None:
    tensors = (
        forward.continuous_design,
        forward.temperatures,
        forward.objective.total,
        forward.rollout.final_state,
    )
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise _TrainingInvalidError("NONFINITE_TRAINING_STATE")
    if forward.projection.absolute_error > 1.0e-6:
        raise _TrainingInvalidError("INVALID_VOLUME_PROJECTION")
    if forward.rollout.maximum_absolute_delta > 0.100001:
        raise _TrainingInvalidError("UPDATE_BOUND_VIOLATION")
    if forward.rollout.maximum_absolute_state > 6.4001:
        raise _TrainingInvalidError("STATE_BOUND_VIOLATION")


def _validate_gradients(parameters: list[Tensor]) -> bool:
    if any(parameter.grad is None for parameter in parameters):
        raise _TrainingInvalidError("BROKEN_AUTOGRAD_GRAPH")
    finite = all(
        bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
        if parameter.grad is not None
    )
    if not finite:
        raise _TrainingInvalidError("NONFINITE_TRAINING_STATE")
    return True


def _validate_cg_trace(trace: SolveTrace, *, after_backward: bool) -> None:
    expected_roles = (["forward"] * 3) + (["adjoint"] * 3 if after_backward else [])
    records = trace.records
    if len(records) != len(expected_roles):
        raise _TrainingInvalidError("CG_NONCONVERGENCE")
    if [record.role for record in records] != expected_roles:
        raise _TrainingInvalidError("CG_NONCONVERGENCE")
    if any(
        not record.converged or record.relative_residual > 1.0e-6 for record in records
    ):
        raise _TrainingInvalidError("CG_NONCONVERGENCE")


def _iteration_record(
    iteration: int,
    forward: NCAForwardResult,
    *,
    gradient_norm_before: float,
    gradient_norm_after: float,
    conv3_gradient: float,
    conv1_gradient: float,
    wall_seconds: float,
    all_gradients_finite: bool,
) -> NCAIterationRecord:
    design = forward.continuous_design.detach()
    material_logit = forward.rollout.material_logit.detach()
    with torch.no_grad():
        binary = binary_design(design)
    solve_records = forward.solve_trace.records
    all_cg_converged = True
    values = (
        float(forward.objective.total.detach().item()),
        float(forward.objective.thermal_smooth.detach().item()),
        float(forward.objective.exact_peak.detach().item()),
        float(forward.objective.total_variation.detach().item()),
        float(forward.objective.binarization_penalty.detach().item()),
        float(design.mean().item()),
        float(binary.mean().item()),
        float(material_logit.mean().item()),
        float(material_logit.std(correction=0).item()),
        float(material_logit.min().item()),
        float(material_logit.max().item()),
        float(design.std(correction=0).item()),
        forward.rollout.hidden_state_rms,
        forward.rollout.delta_state_rms,
        forward.rollout.maximum_absolute_delta,
        forward.rollout.maximum_absolute_state,
        gradient_norm_before,
        gradient_norm_after,
        wall_seconds,
    )
    finite = all(math.isfinite(value) for value in values)
    if not finite:
        raise _TrainingInvalidError("NONFINITE_TRAINING_STATE")
    return NCAIterationRecord(
        iteration=iteration,
        total_objective=values[0],
        thermal_smooth=values[1],
        exact_continuous_tmax=values[2],
        total_variation=values[3],
        binarization_penalty=values[4],
        continuous_material_fraction=values[5],
        binary_material_fraction=values[6],
        projection_absolute_error=forward.projection.absolute_error,
        material_logit_mean=values[7],
        material_logit_std=values[8],
        material_logit_minimum=values[9],
        material_logit_maximum=values[10],
        material_std=values[11],
        hidden_state_rms=values[12],
        delta_state_rms=values[13],
        maximum_absolute_delta=values[14],
        maximum_absolute_state=values[15],
        gradient_norm_before_clipping=values[16],
        gradient_norm_after_clipping=values[17],
        conv3x3_weight_gradient_norm=conv3_gradient,
        conv1x1_weight_gradient_norm=conv1_gradient,
        all_parameter_gradients_finite=all_gradients_finite,
        maximum_cg_iterations=max(record.iterations for record in solve_records),
        maximum_explicit_relative_residual=max(
            record.relative_residual for record in solve_records
        ),
        all_cg_converged=all_cg_converged,
        finite=finite,
        wall_seconds=values[18],
    )


def _write_checkpoint(
    output_dir: Path,
    *,
    model: PureNCA,
    optimizer: torch.optim.Optimizer,
    seed: int,
    initial_learning_rate: float,
    completed_updates: int,
) -> None:
    payload = {
        "schema_version": 1,
        "seed": seed,
        "initial_learning_rate": initial_learning_rate,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "completed_updates": completed_updates,
        "last_iteration": completed_updates - 1,
        "model_state_sha256": model_state_sha256(model),
        "model_state": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
    }
    temporary = output_dir / f"checkpoint_{completed_updates:06d}.pt.tmp"
    final = output_dir / f"checkpoint_{completed_updates:06d}.pt"
    torch.save(payload, temporary)
    temporary.replace(final)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_result_artifacts(result: NCARunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([asdict(record) for record in result.records])
    _atomic_text(output_dir / "optimization_metrics.csv", metrics.to_csv(index=False))
    solves = pd.DataFrame([asdict(record) for record in result.solve_records])
    _atomic_text(output_dir / "cg_records.csv", solves.to_csv(index=False))
    payload = {
        "schema_version": 1,
        "status": result.status.value,
        "reason_codes": list(result.reason_codes),
        "seed": result.seed,
        "mode": result.mode,
        "learning_rate": result.learning_rate,
        "requested_iterations": result.requested_iterations,
        "completed_iterations": result.completed_iterations,
        "initial_objective": result.initial_objective,
        "initial_model_sha256": result.initial_model_hash,
        "final_model_sha256": result.final_model_hash,
    }
    if (
        result.final_continuous_design is not None
        and result.final_binary_design is not None
    ):
        np.save(
            output_dir / "design_continuous_64.npy",
            result.final_continuous_design.numpy(),
            allow_pickle=False,
        )
        np.save(
            output_dir / "design_binary_64.npy",
            result.final_binary_design.numpy(),
            allow_pickle=False,
        )
    _atomic_text(
        output_dir / "nca_run_result.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _failed_result(
    *,
    reason_code: str,
    seed: int,
    mode: NCARunMode,
    learning_rate: float,
    requested_iterations: int,
    initial_objective: float | None,
    records: list[NCAIterationRecord],
    solve_records: list[SolveRecord],
    initial_model_hash: str,
    model: PureNCA,
) -> NCARunResult:
    return NCARunResult(
        status=NCARunStatus.INVALID_RUN,
        reason_codes=(reason_code,),
        seed=seed,
        mode=mode,
        learning_rate=learning_rate,
        requested_iterations=requested_iterations,
        completed_iterations=len(records),
        initial_objective=initial_objective,
        records=tuple(records),
        solve_records=tuple(solve_records),
        initial_model_hash=initial_model_hash,
        final_model_hash=model_state_sha256(model),
        final_continuous_design=None,
        final_binary_design=None,
    )


def run_nca_training(
    sources: Tensor,
    *,
    seed: int,
    learning_rate: float,
    iterations: int,
    mode: NCARunMode,
    output_dir: Path | None,
    evaluator: NCAEvaluator = evaluate_nca,
    allow_cpu_unit_test: bool = False,
    checkpoint_interval: int = 100,
    synchronize: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    iteration_start_hook: Callable[[int], None] | None = None,
    iteration_configurator: (
        Callable[[int, torch.optim.Optimizer], None] | None
    ) = None,
) -> NCARunResult:
    """Run fixed Adam updates, fail closed, and freeze the post-update design."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if learning_rate <= 0.0 or checkpoint_interval < 1:
        raise ValueError("learning rate and checkpoint interval must be positive")
    if mode != "unit" and allow_cpu_unit_test:
        raise ValueError("CPU permission is restricted to unit mode")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    model = initialize_nca(seed, sources.device)
    initial_model_hash = model_state_sha256(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    parameters = list(model.parameters())
    records: list[NCAIterationRecord] = []
    solve_records: list[SolveRecord] = []
    initial_objective: float | None = None
    active_trace: SolveTrace | None = None
    failure_reason: str | None = None

    try:
        for iteration in range(iterations):
            if iteration_configurator is not None:
                iteration_configurator(iteration, optimizer)
            if synchronize is not None:
                synchronize()
            if iteration_start_hook is not None:
                iteration_start_hook(iteration)
            started = clock()
            optimizer.zero_grad(set_to_none=True)
            active_trace = SolveTrace()
            forward = evaluator(
                model,
                sources,
                trace=active_trace,
                allow_cpu_unit_test=allow_cpu_unit_test,
            )
            _validate_forward(forward)
            _validate_cg_trace(active_trace, after_backward=False)
            if initial_objective is None:
                initial_objective = float(forward.objective.total.detach().item())
            forward.objective.total.backward()
            all_gradients_finite = _validate_gradients(parameters)
            _validate_cg_trace(active_trace, after_backward=True)
            gradient_norm_before = _gradient_norm(parameters)
            conv3_gradient = _weight_gradient_norm(model.perception.weight)
            conv1_gradient = _weight_gradient_norm(model.update.weight)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            gradient_norm_after = _gradient_norm(parameters)
            optimizer.step()
            if synchronize is not None:
                synchronize()
            elapsed = clock() - started
            record = _iteration_record(
                iteration,
                forward,
                gradient_norm_before=gradient_norm_before,
                gradient_norm_after=gradient_norm_after,
                conv3_gradient=conv3_gradient,
                conv1_gradient=conv1_gradient,
                wall_seconds=elapsed,
                all_gradients_finite=all_gradients_finite,
            )
            records.append(record)
            solve_records.extend(active_trace.records)
            active_trace = None
            completed_updates = iteration + 1
            if output_dir is not None and completed_updates % checkpoint_interval == 0:
                _write_checkpoint(
                    output_dir,
                    model=model,
                    optimizer=optimizer,
                    seed=seed,
                    initial_learning_rate=learning_rate,
                    completed_updates=completed_updates,
                )

        with torch.no_grad():
            final = evaluator(
                model,
                sources,
                trace=SolveTrace(),
                allow_cpu_unit_test=allow_cpu_unit_test,
                include_binary_diagnostic=True,
            )
        if final.binary_design is None:
            raise _TrainingInvalidError("CORRUPTED_FINAL_DESIGN")
        result = NCARunResult(
            status=NCARunStatus.PASS,
            reason_codes=(),
            seed=seed,
            mode=mode,
            learning_rate=learning_rate,
            requested_iterations=iterations,
            completed_iterations=len(records),
            initial_objective=initial_objective,
            records=tuple(records),
            solve_records=tuple(solve_records),
            initial_model_hash=initial_model_hash,
            final_model_hash=model_state_sha256(model),
            final_continuous_design=final.continuous_design.detach().cpu(),
            final_binary_design=final.binary_design.detach().cpu(),
        )
    except torch.OutOfMemoryError:
        failure_reason = "CUDA_OOM"
    except CGConvergenceError:
        failure_reason = "CG_NONCONVERGENCE"
    except VolumeProjectionError:
        failure_reason = "INVALID_VOLUME_PROJECTION"
    except FloatingPointError:
        failure_reason = "NONFINITE_TRAINING_STATE"
    except _TrainingInvalidError as error:
        failure_reason = error.reason_code

    if failure_reason is not None:
        if active_trace is not None:
            solve_records.extend(active_trace.records)
        result = _failed_result(
            reason_code=failure_reason,
            seed=seed,
            mode=mode,
            learning_rate=learning_rate,
            requested_iterations=iterations,
            initial_objective=initial_objective,
            records=records,
            solve_records=solve_records,
            initial_model_hash=initial_model_hash,
            model=model,
        )
    if output_dir is not None:
        _write_result_artifacts(result, output_dir)
    return result
