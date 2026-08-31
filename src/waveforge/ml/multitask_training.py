"""Sequential microbatch training for one shared generative NCA."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from waveforge.design.differentiable_solver import SolveTrace
from waveforge.design.parameterization import VolumeProjectionError
from waveforge.ml.multitask_protocol import MultitaskStage, settings_at
from waveforge.ml.multitask_tasks import SourceLayoutTask, sample_primary_task
from waveforge.ml.nca_training import evaluate_nca, initialize_nca, model_state_sha256
from waveforge.physics.cg import CGConvergenceError

MultitaskMode = Literal["unit", "benchmark", "pilot", "production"]


class MultitaskRunStatus(StrEnum):
    """Machine-readable outcome of one training invocation."""

    PASS = "PASS"
    INCOMPLETE = "INCOMPLETE"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class MultitaskRunConfig:
    """Immutable settings that define one training trajectory."""

    model_seed: int
    task_seed: int
    total_updates: int
    microbatch_size: int
    checkpoint_interval: int = 250
    mode: MultitaskMode = "production"
    device: str = "cuda"
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.model_seed,
            self.task_seed,
            self.total_updates,
            self.microbatch_size,
            self.checkpoint_interval,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields
        ):
            raise ValueError("seeds and counts must be integers")
        if self.total_updates < 1:
            raise ValueError("total_updates must be positive")
        if self.microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.mode != "unit" and torch.device(self.device).type != "cuda":
            raise ValueError("non-unit multi-task training requires CUDA")


@dataclass(frozen=True)
class MultitaskForward:
    """Minimal differentiable result required from one sampled task."""

    loss: Tensor
    thermal_smooth: float
    exact_tmax: float
    continuous_material_fraction: float
    projection_absolute_error: float
    solve_trace: SolveTrace | None


@dataclass(frozen=True)
class MultitaskIterationRecord:
    """Aggregated diagnostics for one optimizer update."""

    update: int
    stage_id: int
    learning_rate: float
    task_exposures: int
    mean_total_objective: float
    mean_thermal_smooth: float
    mean_exact_tmax: float
    maximum_projection_absolute_error: float
    maximum_material_fraction_error: float
    gradient_norm_before_clipping: float
    gradient_norm_after_clipping: float
    all_gradients_finite: bool
    wall_seconds: float


@dataclass(frozen=True)
class MultitaskRunResult:
    """Training state and provenance after one complete or interrupted call."""

    status: MultitaskRunStatus
    reason_codes: tuple[str, ...]
    model_seed: int
    task_seed: int
    requested_updates: int
    completed_updates: int
    microbatch_size: int
    records: tuple[MultitaskIterationRecord, ...]
    initial_model_hash: str
    final_model_hash: str
    last_checkpoint: Path | None


class _MultitaskInvalidError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


TaskProvider = Callable[[int, int, int], SourceLayoutTask]
MultitaskEvaluator = Callable[..., MultitaskForward]
ModelFactory = Callable[[int, torch.device], nn.Module]


def _default_task_provider(
    seed: int, update: int, microbatch_index: int
) -> SourceLayoutTask:
    index = update * 1_000_000 + microbatch_index
    return sample_primary_task(seed, index)


def _default_evaluator(
    model: nn.Module,
    sources: Tensor,
    stage: MultitaskStage,
    *,
    allow_cpu_unit_test: bool,
) -> MultitaskForward:
    trace = SolveTrace()
    forward = evaluate_nca(
        model,  # type: ignore[arg-type]
        sources,
        trace=trace,
        allow_cpu_unit_test=allow_cpu_unit_test,
        projection_beta=stage.beta,
        smooth_max_alpha=stage.alpha,
        tv_weight=stage.tv_weight,
        binarization_weight=stage.binary_weight,
    )
    return MultitaskForward(
        loss=forward.objective.total,
        thermal_smooth=float(forward.objective.thermal_smooth.detach().item()),
        exact_tmax=float(forward.objective.exact_peak.detach().item()),
        continuous_material_fraction=float(
            forward.continuous_design.detach().mean().item()
        ),
        projection_absolute_error=float(forward.projection.absolute_error),
        solve_trace=trace,
    )


def _default_model_factory(seed: int, device: torch.device) -> nn.Module:
    return initialize_nca(seed, device)


def _validate_trace(trace: SolveTrace | None, *, after_backward: bool) -> None:
    if trace is None:
        return
    expected_roles = ["forward"] * 3 + (["adjoint"] * 3 if after_backward else [])
    if len(trace.records) != len(expected_roles):
        raise _MultitaskInvalidError("CG_NONCONVERGENCE")
    if [record.role for record in trace.records] != expected_roles:
        raise _MultitaskInvalidError("CG_NONCONVERGENCE")
    if any(
        not record.converged or record.relative_residual > 1.0e-6
        for record in trace.records
    ):
        raise _MultitaskInvalidError("CG_NONCONVERGENCE")


def _validate_forward(forward: MultitaskForward) -> None:
    values = (
        float(forward.loss.detach().item()),
        forward.thermal_smooth,
        forward.exact_tmax,
        forward.continuous_material_fraction,
        forward.projection_absolute_error,
    )
    if not all(math.isfinite(value) for value in values):
        raise _MultitaskInvalidError("NONFINITE_TRAINING_STATE")
    if not forward.loss.requires_grad:
        raise _MultitaskInvalidError("BROKEN_AUTOGRAD_GRAPH")
    if forward.projection_absolute_error > 1.0e-6:
        raise _MultitaskInvalidError("INVALID_VOLUME_PROJECTION")
    if abs(forward.continuous_material_fraction - 0.25) > 1.0e-6:
        raise _MultitaskInvalidError("INVALID_VOLUME_PROJECTION")


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    squared = sum(
        float(torch.sum(parameter.grad.detach().double().pow(2)).item())
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def _validate_gradients(parameters: list[nn.Parameter]) -> None:
    if any(parameter.grad is None for parameter in parameters):
        raise _MultitaskInvalidError("BROKEN_AUTOGRAD_GRAPH")
    if any(
        not bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
        if parameter.grad is not None
    ):
        raise _MultitaskInvalidError("NONFINITE_TRAINING_STATE")


def _rng_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng(payload: dict[str, object]) -> None:
    random.setstate(payload["python"])  # type: ignore[arg-type]
    np.random.set_state(payload["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(payload["torch_cpu"])  # type: ignore[arg-type]
    if "torch_cuda" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda"])  # type: ignore[arg-type]


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_checkpoint(
    output_dir: Path,
    *,
    config: MultitaskRunConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    records: list[MultitaskIterationRecord],
    initial_model_hash: str,
) -> Path:
    completed_updates = len(records)
    path = output_dir / f"checkpoint_{completed_updates:06d}.pt"
    payload: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "completed_updates": completed_updates,
        "initial_model_hash": initial_model_hash,
        "model_state_sha256": model_state_sha256(model),  # type: ignore[arg-type]
        "model_state": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "records": [asdict(record) for record in records],
        "rng_state": _rng_payload(),
    }
    _atomic_torch_save(payload, path)
    return path


def _load_checkpoint(
    path: Path,
    *,
    config: MultitaskRunConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[list[MultitaskIterationRecord], str]:
    payload = torch.load(
        path, map_location=torch.device(config.device), weights_only=False
    )
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported multi-task checkpoint schema")
    stored_config = payload["config"]
    locked_fields = ("model_seed", "task_seed", "total_updates", "microbatch_size")
    if any(stored_config[field] != getattr(config, field) for field in locked_fields):
        raise ValueError("resume checkpoint does not match locked run configuration")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng(payload["rng_state"])
    records = [MultitaskIterationRecord(**record) for record in payload["records"]]
    if payload["completed_updates"] != len(records):
        raise ValueError("corrupted checkpoint record count")
    if payload["model_state_sha256"] != model_state_sha256(model):  # type: ignore[arg-type]
        raise ValueError("checkpoint model hash mismatch")
    return records, str(payload["initial_model_hash"])


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_run_artifacts(result: MultitaskRunResult, output_dir: Path) -> None:
    metrics = pd.DataFrame([asdict(record) for record in result.records])
    _atomic_text(output_dir / "training_metrics.csv", metrics.to_csv(index=False))
    payload = {
        "schema_version": 1,
        "status": result.status.value,
        "reason_codes": list(result.reason_codes),
        "model_seed": result.model_seed,
        "task_seed": result.task_seed,
        "requested_updates": result.requested_updates,
        "completed_updates": result.completed_updates,
        "microbatch_size": result.microbatch_size,
        "initial_model_sha256": result.initial_model_hash,
        "final_model_sha256": result.final_model_hash,
        "last_checkpoint": (
            result.last_checkpoint.name if result.last_checkpoint is not None else None
        ),
    }
    _atomic_text(
        output_dir / "multitask_run_result.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def run_multitask_training(
    *,
    config: MultitaskRunConfig,
    output_dir: Path | None,
    task_provider: TaskProvider = _default_task_provider,
    evaluator: MultitaskEvaluator = _default_evaluator,
    model_factory: ModelFactory = _default_model_factory,
    resume_checkpoint: Path | None = None,
    maximum_updates_this_call: int | None = None,
    synchronize: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> MultitaskRunResult:
    """Train on independent tasks with immediate loss/M backward accumulation."""
    if maximum_updates_this_call is not None and maximum_updates_this_call < 1:
        raise ValueError("maximum_updates_this_call must be positive")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)
    model = model_factory(config.model_seed, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    parameters = list(model.parameters())
    records: list[MultitaskIterationRecord] = []
    initial_model_hash = model_state_sha256(model)  # type: ignore[arg-type]
    if resume_checkpoint is not None:
        records, initial_model_hash = _load_checkpoint(
            resume_checkpoint,
            config=config,
            model=model,
            optimizer=optimizer,
        )

    start_update = len(records)
    end_update = config.total_updates
    if maximum_updates_this_call is not None:
        end_update = min(end_update, start_update + maximum_updates_this_call)
    last_checkpoint = resume_checkpoint
    if output_dir is not None and resume_checkpoint is None:
        last_checkpoint = _write_checkpoint(
            output_dir,
            config=config,
            model=model,
            optimizer=optimizer,
            records=records,
            initial_model_hash=initial_model_hash,
        )
    failure_reason: str | None = None

    try:
        for update in range(start_update, end_update):
            stage = settings_at(update, config.total_updates)
            optimizer.param_groups[0]["lr"] = stage.learning_rate
            if synchronize is not None:
                synchronize()
            started = clock()
            optimizer.zero_grad(set_to_none=True)
            total_losses: list[float] = []
            thermal_values: list[float] = []
            peak_values: list[float] = []
            projection_errors: list[float] = []
            fraction_errors: list[float] = []

            for microbatch_index in range(config.microbatch_size):
                task = task_provider(config.task_seed, update, microbatch_index)
                sources = torch.as_tensor(
                    task.sources, dtype=torch.float64, device=device
                )
                forward = evaluator(
                    model,
                    sources,
                    stage,
                    allow_cpu_unit_test=config.mode == "unit",
                )
                _validate_forward(forward)
                _validate_trace(forward.solve_trace, after_backward=False)
                total_losses.append(float(forward.loss.detach().item()))
                thermal_values.append(forward.thermal_smooth)
                peak_values.append(forward.exact_tmax)
                projection_errors.append(forward.projection_absolute_error)
                fraction_errors.append(abs(forward.continuous_material_fraction - 0.25))
                (forward.loss / config.microbatch_size).backward()
                _validate_trace(forward.solve_trace, after_backward=True)

            _validate_gradients(parameters)
            gradient_before = _gradient_norm(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
            gradient_after = _gradient_norm(parameters)
            optimizer.step()
            if synchronize is not None:
                synchronize()
            records.append(
                MultitaskIterationRecord(
                    update=update,
                    stage_id=stage.stage_id,
                    learning_rate=stage.learning_rate,
                    task_exposures=config.microbatch_size,
                    mean_total_objective=float(np.mean(total_losses)),
                    mean_thermal_smooth=float(np.mean(thermal_values)),
                    mean_exact_tmax=float(np.mean(peak_values)),
                    maximum_projection_absolute_error=max(projection_errors),
                    maximum_material_fraction_error=max(fraction_errors),
                    gradient_norm_before_clipping=gradient_before,
                    gradient_norm_after_clipping=gradient_after,
                    all_gradients_finite=True,
                    wall_seconds=clock() - started,
                )
            )
            if (
                output_dir is not None
                and len(records) % config.checkpoint_interval == 0
            ):
                last_checkpoint = _write_checkpoint(
                    output_dir,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    records=records,
                    initial_model_hash=initial_model_hash,
                )
    except torch.OutOfMemoryError:
        failure_reason = "CUDA_OOM"
    except CGConvergenceError:
        failure_reason = "CG_NONCONVERGENCE"
    except VolumeProjectionError:
        failure_reason = "INVALID_VOLUME_PROJECTION"
    except FloatingPointError:
        failure_reason = "NONFINITE_TRAINING_STATE"
    except _MultitaskInvalidError as error:
        failure_reason = error.reason_code

    last_checkpoint_update = (
        int(last_checkpoint.stem.split("_")[-1])
        if last_checkpoint is not None
        else None
    )
    if (
        output_dir is not None
        and failure_reason is None
        and len(records) > 0
        and last_checkpoint_update != len(records)
    ):
        last_checkpoint = _write_checkpoint(
            output_dir,
            config=config,
            model=model,
            optimizer=optimizer,
            records=records,
            initial_model_hash=initial_model_hash,
        )

    if failure_reason is not None:
        status = MultitaskRunStatus.INVALID_RUN
        reason_codes = (failure_reason,)
    elif len(records) == config.total_updates:
        status = MultitaskRunStatus.PASS
        reason_codes = ()
    else:
        status = MultitaskRunStatus.INCOMPLETE
        reason_codes = ("INTERRUPTED_WITH_VALID_CHECKPOINT",)

    result = MultitaskRunResult(
        status=status,
        reason_codes=reason_codes,
        model_seed=config.model_seed,
        task_seed=config.task_seed,
        requested_updates=config.total_updates,
        completed_updates=len(records),
        microbatch_size=config.microbatch_size,
        records=tuple(records),
        initial_model_hash=initial_model_hash,
        final_model_hash=model_state_sha256(model),  # type: ignore[arg-type]
        last_checkpoint=last_checkpoint,
    )
    if output_dir is not None:
        _write_run_artifacts(result, output_dir)
    return result
