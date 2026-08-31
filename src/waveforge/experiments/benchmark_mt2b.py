"""Bounded, non-training benchmark for the prospective NCA-MT2B protocol."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from waveforge.design.batched_differentiable_solver import (
    BatchedSolveTrace,
    solve_steady_implicit_batched,
)
from waveforge.design.differentiable_solver import solve_steady_implicit
from waveforge.design.objectives import objective_components
from waveforge.ml.mt2b_conditioning import build_mt2b_conditioning
from waveforge.ml.mt2b_nca import MT2BNCA
from waveforge.ml.mt2b_tasks import balanced_task_batch
from waveforge.ml.nca import project_nca_material
from waveforge.physics.cg import CGConfig
from waveforge.physics.fixed_operator import UniformPlateFactorization
from waveforge.physics.grid import Grid2D


@dataclass(frozen=True)
class BatchMeasurement:
    mode: Literal["sequential", "vectorized", "scenario_vectorized_sequential"]
    variant: Literal["RAW", "PHYSICS"]
    batch_size: int
    median_seconds_per_update: float
    tasks_per_second: float
    peak_memory_bytes: int
    agreement_pass: bool
    temperature_max_absolute_error: float = 0.0
    temperature_max_relative_error: float = 0.0
    loss_relative_error: float = 0.0
    gradient_relative_l2_error: float = 0.0
    gradient_cosine: float = 1.0


@dataclass(frozen=True)
class FixedOperatorMeasurement:
    rhs_count: int
    ordinary_seconds: float
    reusable_seconds: float
    speedup: float
    maximum_absolute_error: float
    maximum_relative_error: float
    agreement_pass: bool


@dataclass(frozen=True)
class RuntimeProjection:
    raw_training_hours: float
    physics_training_hours: float
    validation_hours: float
    reference_hours: float
    total_hours: float


def project_paired_runtime(
    *,
    raw_tasks_per_second: float,
    physics_tasks_per_second: float,
    validation_seconds: float,
    reference_seconds: float,
) -> RuntimeProjection:
    """Project the locked 8,000 task exposures per matched variant."""
    if raw_tasks_per_second <= 0.0 or physics_tasks_per_second <= 0.0:
        raise ValueError("tasks per second must be positive")
    if validation_seconds < 0.0 or reference_seconds < 0.0:
        raise ValueError("runtime additions must be non-negative")
    raw_seconds = 8_000 / raw_tasks_per_second
    physics_seconds = 8_000 / physics_tasks_per_second
    total_seconds = (
        raw_seconds + physics_seconds + validation_seconds + reference_seconds
    )
    return RuntimeProjection(
        raw_training_hours=raw_seconds / 3600.0,
        physics_training_hours=physics_seconds / 3600.0,
        validation_hours=validation_seconds / 3600.0,
        reference_hours=reference_seconds / 3600.0,
        total_hours=total_seconds / 3600.0,
    )


def build_benchmark_report(
    *,
    batch_measurements: tuple[BatchMeasurement, ...],
    fixed_operator: FixedOperatorMeasurement,
    environment: dict[str, object],
    runtime_projection: RuntimeProjection | None,
) -> dict[str, object]:
    """Build the fail-closed benchmark artifact without test-split access."""
    variants = {item.variant for item in batch_measurements}
    modes = {item.mode for item in batch_measurements}
    eligible_modes = [
        mode
        for mode in modes
        if all(
            any(
                item.mode == mode
                and item.variant == variant
                and item.batch_size == 4
                and item.agreement_pass
                for item in batch_measurements
            )
            for variant in variants
        )
    ]
    selected_mode = (
        max(
            eligible_modes,
            key=lambda mode: min(
                item.tasks_per_second
                for item in batch_measurements
                if item.mode == mode and item.variant in variants
            ),
        )
        if eligible_modes
        else None
    )
    benchmark_pass = fixed_operator.agreement_pass and (
        not batch_measurements or selected_mode is not None
    )
    return {
        "schema_version": 1,
        "status": "PASS" if benchmark_pass else "FAIL_NUMERICAL_AGREEMENT",
        "environment": environment,
        "batch_measurements": [asdict(item) for item in batch_measurements],
        "selected_training_mode": selected_mode,
        "rejected_training_modes": sorted(modes - set(eligible_modes)),
        "fixed_operator": asdict(fixed_operator),
        "runtime_projection": (
            asdict(runtime_projection) if runtime_projection is not None else None
        ),
        "long_training_started": False,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }


def _source_array(batch_size: int) -> np.ndarray:
    tasks = []
    batch_index = 0
    while len(tasks) < batch_size:
        tasks.extend(
            balanced_task_batch(
                batch_index,
                seed=2026092201,
                excluded_task_ids=frozenset(),
            )
        )
        batch_index += 1
    return np.stack([task.sources for task in tasks[:batch_size]])


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fixed_operator_measurement(
    *,
    device: torch.device,
    warmups: int,
    measured: int,
) -> FixedOperatorMeasurement:
    source_array = _source_array(4)
    sources = torch.as_tensor(source_array, dtype=torch.float64, device=device)
    conductivity = torch.ones((4, 64, 64), dtype=torch.float64, device=device)
    grid = Grid2D(nx=64, ny=64)
    config = CGConfig(relative_residual_tolerance=1.0e-12, maximum_iterations=4000)
    reusable = UniformPlateFactorization(64, 1.0)

    ordinary_times: list[float] = []
    reusable_times: list[float] = []
    ordinary_fields: np.ndarray | None = None
    reusable_fields: np.ndarray | None = None
    for index in range(warmups + measured):
        _synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            ordinary = solve_steady_implicit_batched(
                conductivity,
                sources,
                grid,
                config=config,
                trace=BatchedSolveTrace(),
            )
        _synchronize(device)
        ordinary_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        reusable_result = reusable.solve_many(source_array.reshape(-1, 64, 64))
        reusable_elapsed = time.perf_counter() - started
        if index >= warmups:
            ordinary_times.append(ordinary_elapsed)
            reusable_times.append(reusable_elapsed)
        ordinary_fields = ordinary.detach().cpu().numpy()
        reusable_fields = reusable_result.temperature.reshape(4, 3, 64, 64)

    assert ordinary_fields is not None and reusable_fields is not None
    maximum_absolute_error = float(np.max(np.abs(ordinary_fields - reusable_fields)))
    maximum_relative_error = maximum_absolute_error / max(
        float(np.max(np.abs(reusable_fields))), 1.0e-12
    )
    ordinary_seconds = statistics.median(ordinary_times)
    reusable_seconds = statistics.median(reusable_times)
    return FixedOperatorMeasurement(
        rhs_count=12,
        ordinary_seconds=ordinary_seconds,
        reusable_seconds=reusable_seconds,
        speedup=ordinary_seconds / reusable_seconds,
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        agreement_pass=(
            maximum_absolute_error <= 1.0e-9 and maximum_relative_error <= 1.0e-8
        ),
    )


def _project_batch(material_logits: Tensor) -> Tensor:
    return torch.stack(
        [
            project_nca_material(material_logits[index : index + 1], beta=2.0).design
            for index in range(material_logits.shape[0])
        ]
    )


def _vectorized_forward(
    model: MT2BNCA,
    condition: Tensor,
    sources: Tensor,
) -> tuple[Tensor, Tensor]:
    rollout = model.rollout(condition)
    designs = _project_batch(rollout.material_logit)
    conductivity = 1.0 + 19.0 * designs.to(torch.float64).pow(3)
    temperatures = solve_steady_implicit_batched(
        conductivity,
        sources,
        Grid2D(nx=64, ny=64),
        trace=BatchedSolveTrace(),
    )
    losses = [
        objective_components(
            temperatures[index],
            designs[index],
            alpha=100.0,
            tv_weight=0.001,
            binarization_weight=0.0,
        ).total
        for index in range(designs.shape[0])
    ]
    return torch.stack(losses).mean(), temperatures


def _sequential_forward(
    model: MT2BNCA,
    condition: Tensor,
    sources: Tensor,
) -> tuple[Tensor, Tensor]:
    losses: list[Tensor] = []
    temperature_fields: list[Tensor] = []
    for index in range(sources.shape[0]):
        rollout = model.rollout(condition[index : index + 1])
        design = project_nca_material(rollout.material_logit, beta=2.0).design
        temperatures = solve_steady_implicit(
            1.0 + 19.0 * design.to(torch.float64).pow(3),
            sources[index],
            Grid2D(nx=64, ny=64),
        )
        losses.append(
            objective_components(
                temperatures,
                design,
                alpha=100.0,
                tv_weight=0.001,
                binarization_weight=0.0,
            ).total
        )
        temperature_fields.append(temperatures)
    return torch.stack(losses).mean(), torch.stack(temperature_fields)


def _flatten_gradients(model: MT2BNCA) -> Tensor:
    gradients = [
        parameter.grad.reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return torch.cat(gradients)


def _agreement(
    condition: Tensor,
    sources: Tensor,
    *,
    device: torch.device,
) -> tuple[float, float, float, float, float, bool]:
    torch.manual_seed(2026092202)
    sequential_model = MT2BNCA().to(device=device, dtype=torch.float32)
    vectorized_model = MT2BNCA().to(device=device, dtype=torch.float32)
    vectorized_model.load_state_dict(sequential_model.state_dict())

    sequential_loss, sequential_temperature = _sequential_forward(
        sequential_model, condition, sources
    )
    sequential_loss.backward()
    sequential_gradient = _flatten_gradients(sequential_model)
    vectorized_loss, vectorized_temperature = _vectorized_forward(
        vectorized_model, condition, sources
    )
    vectorized_loss.backward()
    vectorized_gradient = _flatten_gradients(vectorized_model)

    temperature_error = float(
        torch.max(torch.abs(sequential_temperature - vectorized_temperature)).item()
    )
    temperature_relative = temperature_error / max(
        float(torch.max(torch.abs(sequential_temperature)).item()), 1.0e-12
    )
    loss_relative = float(
        (
            torch.abs(sequential_loss - vectorized_loss)
            / torch.clamp(torch.abs(sequential_loss), min=1.0e-12)
        )
        .detach()
        .item()
    )
    gradient_relative = float(
        (
            torch.linalg.vector_norm(sequential_gradient - vectorized_gradient)
            / torch.clamp(torch.linalg.vector_norm(sequential_gradient), min=1.0e-12)
        ).item()
    )
    gradient_cosine = float(
        torch.nn.functional.cosine_similarity(
            sequential_gradient.reshape(1, -1),
            vectorized_gradient.reshape(1, -1),
        ).item()
    )
    passed = bool(
        temperature_error <= 1.0e-9
        and temperature_relative <= 1.0e-8
        and loss_relative <= 1.0e-8
        and gradient_relative <= 1.0e-6
        and gradient_cosine >= 0.9999999
    )
    return (
        temperature_error,
        temperature_relative,
        loss_relative,
        gradient_relative,
        gradient_cosine,
        passed,
    )


def _condition(
    sources: Tensor,
    variant: Literal["RAW", "PHYSICS"],
    factorization: UniformPlateFactorization,
) -> Tensor:
    if variant == "RAW":
        return build_mt2b_conditioning(sources, variant="RAW")

    def solve(source_array: np.ndarray) -> np.ndarray:
        shape = source_array.shape
        result = factorization.solve_many(source_array.reshape(-1, 64, 64))
        return result.temperature.reshape(shape)

    return build_mt2b_conditioning(
        sources,
        variant="PHYSICS",
        temperature_solver=solve,
    )


def _measure_training_mode(
    *,
    mode: Literal["sequential", "vectorized"],
    variant: Literal["RAW", "PHYSICS"],
    batch_size: int,
    device: torch.device,
    warmups: int,
    measured: int,
    factorization: UniformPlateFactorization,
) -> BatchMeasurement:
    sources = torch.as_tensor(
        _source_array(batch_size), dtype=torch.float64, device=device
    )
    condition = _condition(sources, variant, factorization)
    agreement = _agreement(condition, sources, device=device)
    torch.manual_seed(2026092202)
    model = MT2BNCA().to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    forward = _sequential_forward if mode == "sequential" else _vectorized_forward
    timings: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(warmups + measured):
        _synchronize(device)
        started = time.perf_counter()
        current_condition = _condition(sources, variant, factorization)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward(model, current_condition, sources)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _synchronize(device)
        elapsed = time.perf_counter() - started
        if index >= warmups:
            timings.append(elapsed)
    median_seconds = statistics.median(timings)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return BatchMeasurement(
        mode=mode,
        variant=variant,
        batch_size=batch_size,
        median_seconds_per_update=median_seconds,
        tasks_per_second=batch_size / median_seconds,
        peak_memory_bytes=peak_memory,
        agreement_pass=agreement[5],
        temperature_max_absolute_error=agreement[0],
        temperature_max_relative_error=agreement[1],
        loss_relative_error=agreement[2],
        gradient_relative_l2_error=agreement[3],
        gradient_cosine=agreement[4],
    )


def _environment(device: torch.device) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "cuda": torch.version.cuda,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fixed-operator-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    warmups = 0 if args.quick else 2
    measured = 1 if args.quick else 5
    fixed = _fixed_operator_measurement(
        device=device,
        warmups=warmups,
        measured=measured,
    )
    measurements: list[BatchMeasurement] = []
    projection: RuntimeProjection | None = None
    if not args.fixed_operator_only:
        factorization = UniformPlateFactorization(64, 1.0)
        for variant in ("RAW", "PHYSICS"):
            for batch_size in (1, 2, 4, 8):
                for mode in ("sequential", "vectorized"):
                    measurements.append(
                        _measure_training_mode(
                            mode=mode,
                            variant=variant,
                            batch_size=batch_size,
                            device=device,
                            warmups=warmups,
                            measured=measured,
                            factorization=factorization,
                        )
                    )
        raw = next(
            item
            for item in measurements
            if item.variant == "RAW"
            and item.mode == "vectorized"
            and item.batch_size == 4
        )
        physics = next(
            item
            for item in measurements
            if item.variant == "PHYSICS"
            and item.mode == "vectorized"
            and item.batch_size == 4
        )
        projection = project_paired_runtime(
            raw_tasks_per_second=raw.tasks_per_second,
            physics_tasks_per_second=physics.tasks_per_second,
            validation_seconds=0.0,
            reference_seconds=0.0,
        )
    report = build_benchmark_report(
        batch_measurements=tuple(measurements),
        fixed_operator=fixed,
        environment=_environment(device),
        runtime_projection=projection,
    )
    _write_json(args.output, report)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
