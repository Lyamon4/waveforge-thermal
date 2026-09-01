"""Matched FIELD/SENS training primitives for the MT3 experiment."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

import pandas as pd
import torch
from torch import Tensor, nn

from waveforge.design.batched_differentiable_solver import (
    BatchedSolveTrace,
    solve_steady_implicit_batched,
)
from waveforge.design.objectives import objective_components
from waveforge.ml.mt2b_tasks import balanced_task_batch
from waveforge.ml.mt3_conditioning import (
    MT3ConditioningVariant,
    build_mt3_conditioning,
    compute_initial_probe,
)
from waveforge.ml.mt3_loss import mt3_best_of_four_loss
from waveforge.ml.mt3_protocol import MT3Stage, training_settings_at
from waveforge.ml.mt3_unet import MT3UNet, project_mt3_candidates
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.physics.grid import Grid2D
from waveforge.reproducibility import set_deterministic_seed

MT3Variant: TypeAlias = MT3ConditioningVariant
MT3Mode: TypeAlias = Literal["unit", "qualification", "production"]
_GRID = Grid2D(nx=64, ny=64)


class MT3RunStatus(StrEnum):
    PASS = "PASS"
    INCOMPLETE = "INCOMPLETE"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class MT3BatchForward:
    loss: Tensor
    candidate_thermal_smooth: Tensor
    candidate_exact_tmax: Tensor
    mean_total_variation: float
    mean_binarization_penalty: float
    softmin: float
    diversity_penalty: float
    maximum_projection_absolute_error: float
    candidate_trace: BatchedSolveTrace | None
    probe_trace: BatchedSolveTrace | None


@dataclass(frozen=True)
class MT3RunConfig:
    variant: MT3Variant
    model_seed: int
    task_seed: int
    base_learning_rate: float
    total_updates: int
    batch_size: int
    checkpoint_interval: int
    mode: MT3Mode = "production"
    device: str = "cuda"
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.variant not in {"FIELD_UNET", "SENS_UNET"}:
            raise ValueError("unknown MT3 training variant")
        counts = (
            self.model_seed,
            self.task_seed,
            self.total_updates,
            self.batch_size,
            self.checkpoint_interval,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in counts
        ):
            raise ValueError("MT3 seeds and counts must be integers")
        if any(value < 1 for value in counts):
            raise ValueError("MT3 seeds and counts must be positive")
        if self.base_learning_rate not in (1.0e-4, 3.0e-4):
            raise ValueError("MT3 base learning rate is not registered")
        if self.gradient_clip_norm != 1.0:
            raise ValueError("MT3 gradient clip norm must equal 1.0")
        if self.mode != "unit" and torch.device(self.device).type != "cuda":
            raise ValueError("non-unit MT3 training requires CUDA")
        if self.mode == "production" and (
            self.total_updates != 4000
            or self.batch_size != 4
            or self.checkpoint_interval != 500
        ):
            raise ValueError("production MT3 training must use the locked budget")
        if self.mode == "qualification" and (
            self.total_updates != 500 or self.batch_size != 4
        ):
            raise ValueError("MT3 qualification must use 500 updates and batch four")


@dataclass(frozen=True)
class MT3IterationRecord:
    update: int
    learning_rate: float
    mean_loss: float
    mean_candidate_thermal_smooth: float
    best_candidate_thermal_smooth: float
    worst_candidate_exact_tmax: float
    mean_total_variation: float
    mean_binarization_penalty: float
    softmin: float
    diversity_penalty: float
    maximum_projection_absolute_error: float
    gradient_norm_before_clipping: float
    gradient_norm_after_clipping: float
    wall_seconds: float


@dataclass(frozen=True)
class MT3RunResult:
    status: MT3RunStatus
    reason_codes: tuple[str, ...]
    config: MT3RunConfig
    completed_updates: int
    records: tuple[MT3IterationRecord, ...]
    initial_model_hash: str
    final_model_hash: str
    last_checkpoint: Path | None


@dataclass(frozen=True)
class MT3QualificationRun:
    learning_rate: float
    model_seed: int
    valid: bool
    median_best4_r25_gap: float
    p90_best4_r25_gap: float


@dataclass(frozen=True)
class MT3QualificationVerdict:
    production_authorized: bool
    selected_learning_rate: float | None
    reason: str
    rows: tuple[MT3QualificationRun, ...]


MT3Evaluator = Callable[..., MT3BatchForward]
MT3ModelFactory = Callable[[int, torch.device], nn.Module]
MT3TaskBatchProvider = Callable[[int, int], tuple[SourceLayoutTask, ...]]


def initialize_mt3_model(seed: int, device: torch.device) -> MT3UNet:
    """Create either matched variant from byte-identical seeded parameters."""
    set_deterministic_seed(seed)
    return MT3UNet().to(device=device, dtype=torch.float32)


def mt3_model_state_sha256(model: nn.Module) -> str:
    """Hash every named state tensor, including its dtype and exact shape."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def mt3_task_batch(seed: int, update: int) -> tuple[SourceLayoutTask, ...]:
    """Return the deterministic four-stratum task batch for one update."""
    return balanced_task_batch(update, seed=seed, excluded_task_ids=frozenset())


def _validate_batched_trace(
    trace: BatchedSolveTrace | None,
    *,
    forward_count: int,
    after_backward: bool,
) -> None:
    if trace is None:
        return
    roles = ["forward"] * forward_count
    if after_backward:
        roles += ["adjoint"] * forward_count
    if [record.role for record in trace.records] != roles:
        raise RuntimeError("MT3 batched physics trace is incomplete")
    if any(
        not record.converged or record.relative_residual > 1.0e-6
        for record in trace.records
    ):
        raise RuntimeError("MT3 batched physics residual exceeds tolerance")


def evaluate_mt3_batch(
    model: nn.Module,
    sources: Tensor,
    stage: MT3Stage,
    *,
    variant: MT3Variant,
    allow_cpu_unit_test: bool,
) -> MT3BatchForward:
    """Evaluate one true task/candidate/scenario-vectorized MT3 batch."""
    parameter = next(model.parameters())
    if parameter.dtype is not torch.float32:
        raise ValueError("MT3 model parameters must use float32")
    if parameter.device.type != "cuda" and not allow_cpu_unit_test:
        raise ValueError("production MT3 batch evaluation requires CUDA")
    if (
        sources.ndim != 4
        or tuple(sources.shape[1:]) != (3, 64, 64)
        or sources.dtype is not torch.float64
        or sources.device != parameter.device
    ):
        raise ValueError("sources must be float64 [batch,3,64,64] on model device")
    if variant not in {"FIELD_UNET", "SENS_UNET"}:
        raise ValueError("unknown MT3 conditioning variant")

    probe = compute_initial_probe(
        sources,
        allow_cpu_unit_test=allow_cpu_unit_test,
    )
    condition = build_mt3_conditioning(probe, sources, variant=variant)
    logits = model(condition)
    candidates = project_mt3_candidates(logits, beta=stage.projection_beta)
    batch = sources.shape[0]
    flattened_designs = candidates.designs.reshape(batch * 4, 64, 64)
    conductivity = 1.0 + 19.0 * flattened_designs.to(torch.float64).pow(3)
    candidate_sources = (
        sources[:, None].expand(batch, 4, 3, 64, 64).reshape(batch * 4, 3, 64, 64)
    )
    candidate_trace = BatchedSolveTrace()
    temperatures = solve_steady_implicit_batched(
        conductivity,
        candidate_sources,
        _GRID,
        trace=candidate_trace,
    )

    components = [
        objective_components(
            temperatures[index],
            flattened_designs[index],
            alpha=stage.smooth_max_alpha,
            tv_weight=0.0,
            binarization_weight=0.0,
        )
        for index in range(batch * 4)
    ]
    thermal = torch.stack([item.thermal_smooth for item in components]).reshape(
        batch, 4
    )
    exact = torch.stack([item.exact_peak for item in components]).reshape(batch, 4)
    variations = torch.stack([item.total_variation for item in components])
    binarization = torch.stack([item.binarization_penalty for item in components])
    candidate_loss = mt3_best_of_four_loss(thermal, candidates.designs)
    regularizers = (
        stage.tv_weight * variations.mean()
        + stage.binarization_weight * binarization.mean()
    ).to(dtype=candidate_loss.total.dtype)
    loss = candidate_loss.total + regularizers
    if not torch.isfinite(loss) or not loss.requires_grad:
        raise FloatingPointError("MT3 batch loss is non-finite or detached")
    return MT3BatchForward(
        loss=loss,
        candidate_thermal_smooth=thermal,
        candidate_exact_tmax=exact,
        mean_total_variation=float(variations.detach().mean().item()),
        mean_binarization_penalty=float(binarization.detach().mean().item()),
        softmin=float(candidate_loss.softmin_per_task.detach().mean().item()),
        diversity_penalty=float(candidate_loss.diversity_penalty.detach().item()),
        maximum_projection_absolute_error=max(candidates.projection_errors),
        candidate_trace=candidate_trace,
        probe_trace=probe.trace,
    )


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    squared = sum(
        float(parameter.grad.detach().double().square().sum().item())
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_checkpoint(
    output_dir: Path,
    *,
    config: MT3RunConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    records: list[MT3IterationRecord],
    initial_model_hash: str,
) -> Path:
    completed = len(records)
    path = output_dir / f"checkpoint_{completed:06d}.pt"
    payload: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "completed_updates": completed,
        "initial_model_hash": initial_model_hash,
        "model_state_sha256": mt3_model_state_sha256(model),
        "model_state": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "records": [asdict(record) for record in records],
    }
    _atomic_torch_save(payload, path)
    return path


def _load_checkpoint(
    path: Path,
    *,
    config: MT3RunConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[list[MT3IterationRecord], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("config") != asdict(config):
        raise ValueError("resume checkpoint does not match locked MT3 configuration")
    records = [MT3IterationRecord(**row) for row in payload["records"]]
    if payload.get("completed_updates") != len(records):
        raise ValueError("MT3 checkpoint record count is corrupted")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if mt3_model_state_sha256(model) != payload.get("model_state_sha256"):
        raise ValueError("MT3 checkpoint model hash mismatch")
    return records, str(payload["initial_model_hash"])


def _write_run_artifacts(result: MT3RunResult, output_dir: Path) -> None:
    metrics = pd.DataFrame([asdict(record) for record in result.records])
    temporary_csv = output_dir / "training_metrics.csv.tmp"
    metrics.to_csv(temporary_csv, index=False, lineterminator="\n")
    temporary_csv.replace(output_dir / "training_metrics.csv")
    payload = {
        "schema_version": 1,
        "status": result.status.value,
        "reason_codes": list(result.reason_codes),
        "config": asdict(result.config),
        "completed_updates": result.completed_updates,
        "initial_model_sha256": result.initial_model_hash,
        "final_model_sha256": result.final_model_hash,
        "last_checkpoint": (
            result.last_checkpoint.name if result.last_checkpoint is not None else None
        ),
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }
    temporary_json = output_dir / "mt3_run_result.json.tmp"
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_json.replace(output_dir / "mt3_run_result.json")


def run_mt3_training(
    *,
    config: MT3RunConfig,
    output_dir: Path | None,
    evaluator: MT3Evaluator = evaluate_mt3_batch,
    model_factory: MT3ModelFactory = initialize_mt3_model,
    task_batch_provider: MT3TaskBatchProvider = mt3_task_batch,
    resume_checkpoint: Path | None = None,
    maximum_updates_this_call: int | None = None,
    synchronize: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> MT3RunResult:
    """Run or resume one deterministic matched MT3 training trajectory."""
    if maximum_updates_this_call is not None and maximum_updates_this_call < 1:
        raise ValueError("maximum updates this call must be positive")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    model = model_factory(config.model_seed, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.base_learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    parameters = list(model.parameters())
    initial_hash = mt3_model_state_sha256(model)
    records: list[MT3IterationRecord] = []
    if resume_checkpoint is not None:
        records, initial_hash = _load_checkpoint(
            resume_checkpoint,
            config=config,
            model=model,
            optimizer=optimizer,
        )
    last_checkpoint = resume_checkpoint
    if output_dir is not None and resume_checkpoint is None:
        last_checkpoint = _write_checkpoint(
            output_dir,
            config=config,
            model=model,
            optimizer=optimizer,
            records=records,
            initial_model_hash=initial_hash,
        )

    start = len(records)
    stop = config.total_updates
    if maximum_updates_this_call is not None:
        stop = min(stop, start + maximum_updates_this_call)
    failure: str | None = None
    try:
        for update in range(start, stop):
            stage = training_settings_at(update)
            learning_rate = config.base_learning_rate * stage.learning_rate_multiplier
            optimizer.param_groups[0]["lr"] = learning_rate
            tasks = task_batch_provider(config.task_seed, update)
            if len(tasks) != config.batch_size:
                raise RuntimeError("MT3 task provider returned the wrong batch size")
            sources = torch.stack(
                [torch.from_numpy(task.sources) for task in tasks]
            ).to(device=device, dtype=torch.float64)
            if synchronize is not None:
                synchronize()
            started = clock()
            optimizer.zero_grad(set_to_none=True)
            forward = evaluator(
                model,
                sources,
                stage,
                variant=config.variant,
                allow_cpu_unit_test=config.mode == "unit",
            )
            _validate_batched_trace(
                forward.candidate_trace,
                forward_count=config.batch_size * 4 * 3,
                after_backward=False,
            )
            forward.loss.backward()
            _validate_batched_trace(
                forward.candidate_trace,
                forward_count=config.batch_size * 4 * 3,
                after_backward=True,
            )
            if any(parameter.grad is None for parameter in parameters):
                raise RuntimeError("MT3 model has a missing parameter gradient")
            if any(
                not torch.isfinite(parameter.grad).all()
                for parameter in parameters
                if parameter.grad is not None
            ):
                raise FloatingPointError("MT3 model gradient contains NaN or Inf")
            gradient_before = _gradient_norm(parameters)
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            gradient_after = _gradient_norm(parameters)
            optimizer.step()
            if synchronize is not None:
                synchronize()
            records.append(
                MT3IterationRecord(
                    update=update,
                    learning_rate=learning_rate,
                    mean_loss=float(forward.loss.detach().item()),
                    mean_candidate_thermal_smooth=float(
                        forward.candidate_thermal_smooth.detach().mean().item()
                    ),
                    best_candidate_thermal_smooth=float(
                        forward.candidate_thermal_smooth.detach()
                        .min(dim=1)
                        .values.mean()
                        .item()
                    ),
                    worst_candidate_exact_tmax=float(
                        forward.candidate_exact_tmax.detach().max().item()
                    ),
                    mean_total_variation=forward.mean_total_variation,
                    mean_binarization_penalty=forward.mean_binarization_penalty,
                    softmin=forward.softmin,
                    diversity_penalty=forward.diversity_penalty,
                    maximum_projection_absolute_error=(
                        forward.maximum_projection_absolute_error
                    ),
                    gradient_norm_before_clipping=gradient_before,
                    gradient_norm_after_clipping=gradient_after,
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
                    initial_model_hash=initial_hash,
                )
    except torch.OutOfMemoryError:
        failure = "CUDA_OOM"
    except FloatingPointError:
        failure = "NONFINITE_TRAINING_STATE"
    except RuntimeError:
        failure = "INVALID_TRAINING_STATE"

    if (
        output_dir is not None
        and failure is None
        and len(records) > 0
        and (
            last_checkpoint is None
            or int(last_checkpoint.stem.rsplit("_", 1)[1]) != len(records)
        )
    ):
        last_checkpoint = _write_checkpoint(
            output_dir,
            config=config,
            model=model,
            optimizer=optimizer,
            records=records,
            initial_model_hash=initial_hash,
        )
    if failure is not None:
        status = MT3RunStatus.INVALID_RUN
        reasons = (failure,)
    elif len(records) == config.total_updates:
        status = MT3RunStatus.PASS
        reasons = ()
    else:
        status = MT3RunStatus.INCOMPLETE
        reasons = ("INTERRUPTED_WITH_VALID_CHECKPOINT",)
    result = MT3RunResult(
        status=status,
        reason_codes=reasons,
        config=config,
        completed_updates=len(records),
        records=tuple(records),
        initial_model_hash=initial_hash,
        final_model_hash=mt3_model_state_sha256(model),
        last_checkpoint=last_checkpoint,
    )
    if output_dir is not None:
        _write_run_artifacts(result, output_dir)
    return result


def select_mt3_learning_rate(
    rows: tuple[MT3QualificationRun, ...],
) -> MT3QualificationVerdict:
    """Apply the locked two-rate qualification ranking without rounded values."""
    candidates = (1.0e-4, 3.0e-4)
    grouped = {
        lr: tuple(row for row in rows if row.learning_rate == lr) for lr in candidates
    }
    if any(len(grouped[lr]) != 2 for lr in candidates) or len(rows) != 4:
        raise ValueError("qualification requires two seeds for each registered rate")

    summaries: dict[float, tuple[int, float, float]] = {}
    for learning_rate, group in grouped.items():
        valid = tuple(row for row in group if row.valid)
        if valid:
            median_gap = statistics.median(row.median_best4_r25_gap for row in valid)
            p90_gap = statistics.median(row.p90_best4_r25_gap for row in valid)
        else:
            median_gap = math.inf
            p90_gap = math.inf
        summaries[learning_rate] = (len(valid), median_gap, p90_gap)

    selected = min(
        candidates,
        key=lambda lr: (
            -summaries[lr][0],
            summaries[lr][1],
            summaries[lr][2],
            lr,
        ),
    )
    if summaries[selected][0] == 0:
        return MT3QualificationVerdict(False, None, "no_valid_runs", rows)
    other = candidates[1] if selected == candidates[0] else candidates[0]
    selected_summary = summaries[selected]
    other_summary = summaries[other]
    if selected_summary[0] != other_summary[0]:
        reason = "more_valid_runs"
    elif selected_summary[1] != other_summary[1]:
        reason = "lower_median_best4_r25_gap"
    elif selected_summary[2] != other_summary[2]:
        reason = "lower_p90_best4_r25_gap"
    else:
        reason = "smaller_learning_rate"
    return MT3QualificationVerdict(True, selected, reason, rows)
