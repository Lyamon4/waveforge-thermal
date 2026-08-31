"""Paid-A100 gates for the prospective multi-task generative NCA campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch

from waveforge.design.binary_readout import exact_cardinality_binary
from waveforge.design.differentiable_solver import SolveTrace, solve_steady_implicit
from waveforge.design.optimize import OptimizationConfig, optimize_design
from waveforge.ml.multitask_evaluation import (
    condition_causality_summary,
    evaluate_frozen_checkpoint,
    pairwise_binary_diversity,
    select_validation_checkpoint,
    summarize_against_reference,
)
from waveforge.ml.multitask_protocol import (
    DEVELOPMENT_SEED,
    PILOT_UPDATES,
    PRODUCTION_SEEDS,
)
from waveforge.ml.multitask_provenance import (
    build_hash_manifest,
    create_production_registry,
    validate_backup_readiness,
    validate_production_registry,
)
from waveforge.ml.multitask_tasks import (
    FrozenTaskSplits,
    SourceLayoutTask,
    build_frozen_splits,
    sample_primary_task,
    sample_training_task,
    write_split_manifest,
)
from waveforge.ml.multitask_training import (
    MultitaskRunConfig,
    MultitaskRunStatus,
    run_multitask_training,
)
from waveforge.physics.grid import Grid2D
from waveforge.reproducibility import artifact_sha256, configure_cuda_reproducibility
from waveforge.verification.compare import Gate2Status
from waveforge.verification.multitask_verification import (
    classify_campaign,
    summarize_seed,
    verify_binary_task,
)


class MultitaskGateError(RuntimeError):
    """Raised when a paid campaign phase is not prospectively authorized."""


class PilotStatus(StrEnum):
    """Machine-readable development pilot outcomes."""

    PILOT_GO = "PILOT_GO"
    PILOT_CONDITIONAL = "PILOT_CONDITIONAL"
    PILOT_KILL = "PILOT_KILL"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class BenchmarkCandidate:
    """Measured throughput and validity for one sequential microbatch."""

    microbatch_size: int
    median_seconds_per_update: float
    p90_seconds_per_update: float
    tasks_per_second: float
    scientifically_valid: bool
    peak_cuda_memory_bytes: int


@dataclass(frozen=True)
class RuntimeGate:
    """Affordable production length under the locked three-seed cap."""

    seconds_per_update: float
    remaining_hours: float
    updates_per_seed: int
    production_authorized: bool


def calculate_runtime_gate(
    *,
    seconds_per_update: float,
    remaining_hours: float,
) -> RuntimeGate:
    """Allocate a prospective time budget equally and require 5k updates per seed."""
    if (
        not math.isfinite(seconds_per_update)
        or seconds_per_update <= 0.0
        or not math.isfinite(remaining_hours)
        or remaining_hours <= 0.0
    ):
        raise ValueError("runtime inputs must be finite and positive")
    updates = min(
        15_000,
        math.floor(remaining_hours * 3600.0 / (3.0 * seconds_per_update)),
    )
    return RuntimeGate(
        seconds_per_update=seconds_per_update,
        remaining_hours=remaining_hours,
        updates_per_seed=updates,
        production_authorized=updates >= 5_000,
    )


def select_microbatch(candidates: list[BenchmarkCandidate]) -> BenchmarkCandidate:
    """Select maximum throughput; within 2% prefer the smaller microbatch."""
    eligible = [candidate for candidate in candidates if candidate.scientifically_valid]
    if not eligible:
        raise MultitaskGateError("no scientifically valid microbatch candidate")
    maximum_throughput = max(candidate.tasks_per_second for candidate in eligible)
    tied = [
        candidate
        for candidate in eligible
        if candidate.tasks_per_second >= 0.98 * maximum_throughput
    ]
    return min(tied, key=lambda candidate: candidate.microbatch_size)


def classify_pilot(
    *,
    numerically_valid: bool,
    projection_valid: bool,
    binary_budget_valid: bool,
    validation_improved: bool,
    matched_condition_wins: int,
    source_independent: bool,
    median_gradient_gap: float,
) -> PilotStatus:
    """Apply the locked 15%/20% pilot decision without result tuning."""
    if (
        not numerically_valid
        or not projection_valid
        or not binary_budget_valid
        or not math.isfinite(median_gradient_gap)
    ):
        return PilotStatus.INVALID_RUN
    if (
        not validation_improved
        or matched_condition_wins < 23
        or source_independent
        or median_gradient_gap > 0.20
    ):
        return PilotStatus.PILOT_KILL
    if median_gradient_gap <= 0.15:
        return PilotStatus.PILOT_GO
    return PilotStatus.PILOT_CONDITIONAL


def registered_test_baseline_jobs() -> tuple[dict[str, object], ...]:
    """Return the immutable single- and four-start comparator registry."""
    splits = build_frozen_splits()
    jobs: list[dict[str, object]] = []
    for task in splits.test_id:
        jobs.append(
            {
                "family": "single_start",
                "split": "test_id",
                "task_id": task.task_id,
                "start_index": 0,
            }
        )
    for split_name, tasks in (
        ("test_id", splits.test_id[:8]),
        ("test_ood", splits.test_ood[:8]),
    ):
        for task in tasks:
            for start_index in range(4):
                jobs.append(
                    {
                        "family": "multistart_challenge",
                        "split": split_name,
                        "task_id": task.task_id,
                        "start_index": start_index,
                    }
                )
    return tuple(jobs)


def assemble_production_payload(
    shards: list[dict[str, object]],
    *,
    updates_per_seed: int,
    microbatch_size: int,
    training_hours_cap: float,
    worker_count: int,
) -> dict[str, object]:
    """Assemble immutable independently written production seed shards."""
    seeds = [int(shard.get("seed", -1)) for shard in shards]
    if seeds != list(PRODUCTION_SEEDS):
        raise MultitaskGateError(
            "production shards must contain exact registered seeds"
        )
    if any(
        shard.get("status") != "PASS"
        or int(shard.get("completed_updates", -1)) != updates_per_seed
        for shard in shards
    ):
        raise MultitaskGateError("a production shard is incomplete or invalid")
    if worker_count not in (1, 2, 3):
        raise MultitaskGateError("production worker count must lie in [1,3]")
    return {
        "schema_version": 2,
        "status": "PASS",
        "production_seeds": list(PRODUCTION_SEEDS),
        "updates_per_seed": updates_per_seed,
        "microbatch_size": microbatch_size,
        "training_hours_cap": training_hours_cap,
        "worker_count": worker_count,
        "maximum_seed_training_wall_seconds": max(
            float(shard["training_wall_seconds"]) for shard in shards
        ),
        "summed_seed_training_wall_seconds": sum(
            float(shard["training_wall_seconds"]) for shard in shards
        ),
        "frozen_models": shards,
        "test_sets_accessed": False,
    }


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise MultitaskGateError(f"required artifact is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def validate_production_gate(output_dir: Path) -> None:
    """Require both an affordable benchmark and an actual PILOT_GO."""
    benchmark_path = output_dir / "benchmark_verdict.json"
    if not benchmark_path.is_file():
        raise MultitaskGateError("benchmark verdict is missing")
    benchmark = _read_json(benchmark_path)
    if benchmark.get("production_runtime_authorized") is not True:
        raise MultitaskGateError("benchmark does not authorize production runtime")
    pilot_path = output_dir / "pilot_verdict.json"
    if not pilot_path.is_file():
        raise MultitaskGateError("pilot verdict is missing")
    pilot = _read_json(pilot_path)
    if pilot.get("status") != PilotStatus.PILOT_GO.value:
        raise MultitaskGateError("production requires PILOT_GO")


def run_preflight(output_dir: Path) -> dict[str, object]:
    """Fail before paid training unless the A100 software environment is exact."""
    cuda_available = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda_available else None
    python_ok = sys.version_info[:2] == (3, 11)
    torch_ok = torch.__version__.split("+")[0].startswith("2.13.")
    gpu_ok = gpu is not None and "A100" in gpu
    status = (
        "PASS"
        if cuda_available and python_ok and torch_ok and gpu_ok
        else "INVALID_RUN"
    )
    determinism = None
    if status == "PASS":
        determinism = asdict(configure_cuda_reproducibility(DEVELOPMENT_SEED))
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_available else None
        ),
        "determinism": determinism,
    }
    _write_json(output_dir / "environment.json", payload)
    if status != "PASS":
        raise MultitaskGateError("A100 preflight is INVALID_RUN")
    return payload


def _fixed_benchmark_task(
    seed: int,
    update: int,
    microbatch_index: int,
) -> SourceLayoutTask:
    return sample_primary_task(DEVELOPMENT_SEED, 0)


def run_benchmark(
    output_dir: Path,
    *,
    remaining_hours: float = 6.0,
    warmup_updates: int = 20,
    measured_updates: int = 200,
) -> dict[str, object]:
    """Benchmark fixed-task complete updates for M=1,2,4 on the A100."""
    if not (output_dir / "environment.json").is_file():
        run_preflight(output_dir)
    configure_cuda_reproducibility(DEVELOPMENT_SEED)
    candidates: list[BenchmarkCandidate] = []
    for microbatch_size in (1, 2, 4):
        candidate_dir = output_dir / "benchmark" / f"microbatch_{microbatch_size}"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = run_multitask_training(
            config=MultitaskRunConfig(
                model_seed=DEVELOPMENT_SEED,
                task_seed=DEVELOPMENT_SEED,
                total_updates=warmup_updates + measured_updates,
                microbatch_size=microbatch_size,
                checkpoint_interval=warmup_updates + measured_updates,
                mode="benchmark",
                device="cuda",
            ),
            task_provider=_fixed_benchmark_task,
            output_dir=candidate_dir,
            synchronize=torch.cuda.synchronize,
        )
        measured = result.records[warmup_updates:]
        seconds = np.asarray([record.wall_seconds for record in measured])
        valid = (
            result.status is MultitaskRunStatus.PASS
            and len(measured) == measured_updates
            and np.isfinite(seconds).all()
            and all(
                record.maximum_projection_absolute_error <= 1.0e-6
                and record.maximum_material_fraction_error <= 1.0e-6
                for record in measured
            )
        )
        median_seconds = float(np.median(seconds)) if len(seconds) else math.inf
        p90_seconds = float(np.quantile(seconds, 0.9)) if len(seconds) else math.inf
        candidates.append(
            BenchmarkCandidate(
                microbatch_size=microbatch_size,
                median_seconds_per_update=median_seconds,
                p90_seconds_per_update=p90_seconds,
                tasks_per_second=(
                    microbatch_size / median_seconds if median_seconds > 0.0 else 0.0
                ),
                scientifically_valid=valid,
                peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated()),
            )
        )
    selected = select_microbatch(candidates)
    runtime = calculate_runtime_gate(
        seconds_per_update=selected.median_seconds_per_update,
        remaining_hours=remaining_hours,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "warmup_updates": warmup_updates,
        "measured_updates": measured_updates,
        "candidates": [asdict(candidate) for candidate in candidates],
        "selected_microbatch_size": selected.microbatch_size,
        "selected_median_seconds_per_update": selected.median_seconds_per_update,
        "production_updates_per_seed": runtime.updates_per_seed,
        "production_runtime_authorized": runtime.production_authorized,
        "remaining_training_hours": remaining_hours,
    }
    _write_json(output_dir / "benchmark_verdict.json", payload)
    return payload


def lock_runtime_budget_amendment(
    output_dir: Path,
    *,
    production_training_hours: float,
    maximum_campaign_cost_usd: float,
    hourly_cost_usd: float,
) -> dict[str, object]:
    """Prospectively revise only runtime after measurement and before pilot."""
    if any(
        (output_dir / name).exists()
        for name in (
            "pilot_verdict.json",
            "production_registry.json",
            "production_verdict.json",
        )
    ):
        raise MultitaskGateError("runtime budget must be locked before pilot")
    values = (
        production_training_hours,
        maximum_campaign_cost_usd,
        hourly_cost_usd,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("runtime budget values must be finite and positive")
    production_cost = production_training_hours * hourly_cost_usd
    if production_cost > maximum_campaign_cost_usd:
        raise ValueError("production allocation exceeds maximum campaign cost")

    verdict_path = output_dir / "benchmark_verdict.json"
    original = _read_json(verdict_path)
    if original.get("status") != "PASS":
        raise MultitaskGateError("runtime amendment requires a PASS benchmark")
    original_path = output_dir / "benchmark_verdict_original.json"
    if not original_path.exists():
        _write_json(original_path, original)
    else:
        original = _read_json(original_path)

    seconds_per_update = float(original["selected_median_seconds_per_update"])
    runtime = calculate_runtime_gate(
        seconds_per_update=seconds_per_update,
        remaining_hours=production_training_hours,
    )
    amended = dict(original)
    amended.update(
        {
            "schema_version": 2,
            "production_updates_per_seed": runtime.updates_per_seed,
            "production_runtime_authorized": runtime.production_authorized,
            "remaining_training_hours": production_training_hours,
            "runtime_budget_amended": True,
            "runtime_budget_amendment_artifact": "runtime_budget_amendment.json",
        }
    )
    amendment: dict[str, object] = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_PILOT",
        "prospective_before_pilot": True,
        "scientific_parameters_changed": False,
        "original_training_hours": float(original["remaining_training_hours"]),
        "production_training_hours": production_training_hours,
        "hourly_cost_usd": hourly_cost_usd,
        "maximum_campaign_cost_usd": maximum_campaign_cost_usd,
        "maximum_production_cost_usd": production_cost,
        "selected_median_seconds_per_update": seconds_per_update,
        "production_updates_per_seed": runtime.updates_per_seed,
        "production_runtime_authorized": runtime.production_authorized,
        "reason": (
            "Measured A100 throughput made the original six-hour allocation "
            "insufficient for the preregistered 5000-update minimum."
        ),
    }
    _write_json(output_dir / "runtime_budget_amendment.json", amendment)
    _write_json(verdict_path, amended)
    return amended


def _score_binary_design(
    binary_design: torch.Tensor,
    task: SourceLayoutTask,
    *,
    device: torch.device,
) -> float:
    binary = binary_design.to(device=device, dtype=torch.float64)
    sources = torch.as_tensor(task.sources, dtype=torch.float64, device=device)
    trace = SolveTrace()
    with torch.no_grad():
        temperatures = solve_steady_implicit(
            1.0 + 19.0 * binary,
            sources,
            Grid2D(nx=64, ny=64),
            trace=trace,
        )
    if len(trace.records) != 3 or any(not record.converged for record in trace.records):
        raise MultitaskGateError("direct-gradient verification CG failed")
    return float(torch.max(temperatures).item())


def _pilot_gradient_references(
    output_dir: Path,
    tasks: tuple[SourceLayoutTask, ...],
) -> dict[str, float]:
    reference_path = output_dir / "pilot_gradient_references.json"
    existing: dict[str, float] = {}
    if reference_path.is_file():
        payload = _read_json(reference_path)
        existing = {
            str(task_id): float(value)
            for task_id, value in dict(payload.get("peaks", {})).items()
        }
    device = torch.device("cuda")
    for index, task in enumerate(tasks):
        if task.task_id in existing:
            continue
        sources = torch.as_tensor(task.sources, dtype=torch.float64, device=device)
        result = optimize_design(
            sources,
            seed=2026083200 + index,
            config=OptimizationConfig(enforce_final_binary_budget=False),
            output_dir=None,
        )
        if result.status is Gate2Status.INVALID_RUN or result.continuous_design is None:
            raise MultitaskGateError("pilot direct-gradient optimizer was INVALID_RUN")
        exact_binary, _ = exact_cardinality_binary(result.continuous_design)
        existing[task.task_id] = _score_binary_design(
            exact_binary,
            task,
            device=device,
        )
        _write_json(
            reference_path,
            {
                "schema_version": 1,
                "optimizer": "WaveForge_direct_gradient_600_steps",
                "primary_binary_readout": "exact_top_1024",
                "peaks": existing,
            },
        )
    return existing


def _summary_payload(summary: object) -> dict[str, object]:
    return asdict(summary)  # type: ignore[arg-type]


def _evaluate_pilot_checkpoints(
    pilot_dir: Path,
    splits: FrozenTaskSplits,
    gradient_references: dict[str, float],
) -> tuple[Path, list[dict[str, object]]]:
    summaries = []
    rows: list[dict[str, object]] = []
    checkpoints = sorted(pilot_dir.glob("checkpoint_*.pt"))
    for checkpoint in checkpoints:
        completed = int(checkpoint.stem.split("_")[-1])
        if completed != 0 and completed % 250 != 0:
            continue
        evaluation = evaluate_frozen_checkpoint(
            checkpoint,
            splits.validation,
            split_name="validation",
            device=torch.device("cuda"),
        )
        task_ids = tuple(item.task_id for item in evaluation.tasks)
        peaks = tuple(item.peak_temperature for item in evaluation.tasks)
        dummy_references = {task_id: 1.0 for task_id in task_ids}
        peak_summary = summarize_against_reference(
            completed_updates=completed,
            split_name="validation",
            task_ids=task_ids,
            candidate_peaks=peaks,
            reference_peaks=dummy_references,
        )
        summaries.append(peak_summary)
        comparison_ids = tuple(gradient_references)
        comparison_peaks = tuple(
            next(
                item.peak_temperature
                for item in evaluation.tasks
                if item.task_id == task_id
            )
            for task_id in comparison_ids
        )
        gap_summary = summarize_against_reference(
            completed_updates=completed,
            split_name="validation",
            task_ids=comparison_ids,
            candidate_peaks=comparison_peaks,
            reference_peaks=gradient_references,
        )
        rows.append(
            {
                "checkpoint": checkpoint.name,
                "peak_summary": _summary_payload(peak_summary),
                "gradient_gap_summary": _summary_payload(gap_summary),
            }
        )
    selected_summary = select_validation_checkpoint(summaries)
    selected = pilot_dir / f"checkpoint_{selected_summary.completed_updates:06d}.pt"
    _write_json(
        pilot_dir / "pilot_checkpoint_validation.json",
        {"schema_version": 1, "rows": rows, "selected_checkpoint": selected.name},
    )
    return selected, rows


def run_pilot(output_dir: Path) -> dict[str, object]:
    """Run the locked 1,500-update pilot and its prospective causal gate."""
    verdict_path = output_dir / "pilot_verdict.json"
    if verdict_path.is_file():
        return _read_json(verdict_path)
    benchmark = _read_json(output_dir / "benchmark_verdict.json")
    if benchmark.get("status") != "PASS":
        raise MultitaskGateError("pilot requires a PASS benchmark")
    configure_cuda_reproducibility(DEVELOPMENT_SEED)
    microbatch_size = int(benchmark["selected_microbatch_size"])
    splits = build_frozen_splits()
    write_split_manifest(output_dir / "split_manifest.json", splits)
    comparison_tasks = splits.validation[:8]
    references = _pilot_gradient_references(output_dir, comparison_tasks)
    pilot_dir = output_dir / "pilot"
    existing_checkpoints = sorted(pilot_dir.glob("checkpoint_*.pt"))
    resume: Path | None = existing_checkpoints[-1] if existing_checkpoints else None
    while True:
        result = run_multitask_training(
            config=MultitaskRunConfig(
                model_seed=DEVELOPMENT_SEED,
                task_seed=DEVELOPMENT_SEED,
                total_updates=PILOT_UPDATES,
                microbatch_size=microbatch_size,
                checkpoint_interval=250,
                mode="pilot",
                device="cuda",
            ),
            output_dir=pilot_dir,
            resume_checkpoint=resume,
            maximum_updates_this_call=250,
            synchronize=torch.cuda.synchronize,
        )
        if result.status is MultitaskRunStatus.INVALID_RUN:
            payload = {
                "schema_version": 1,
                "status": PilotStatus.INVALID_RUN.value,
                "reason_codes": list(result.reason_codes),
            }
            _write_json(verdict_path, payload)
            return payload
        resume = result.last_checkpoint
        if result.completed_updates == PILOT_UPDATES:
            break

    selected, validation_rows = _evaluate_pilot_checkpoints(
        pilot_dir,
        splits,
        references,
    )
    selected_evaluation = evaluate_frozen_checkpoint(
        selected,
        splits.validation,
        split_name="validation",
        device=torch.device("cuda"),
    )
    shuffled_conditions = splits.validation[1:] + splits.validation[:1]
    shuffled_evaluation = evaluate_frozen_checkpoint(
        selected,
        splits.validation,
        split_name="validation",
        conditioning_tasks=shuffled_conditions,
        device=torch.device("cuda"),
    )
    causality = condition_causality_summary(
        matched=[item.peak_temperature for item in selected_evaluation.tasks],
        shuffled=[item.peak_temperature for item in shuffled_evaluation.tasks],
    )
    diversity = pairwise_binary_diversity(
        [item.binary_design for item in selected_evaluation.tasks]
    )
    selected_row = next(
        row for row in validation_rows if row["checkpoint"] == selected.name
    )
    initial_row = next(
        row for row in validation_rows if row["checkpoint"] == "checkpoint_000000.pt"
    )
    gap = float(selected_row["gradient_gap_summary"]["median_relative_gap"])
    selected_peak = float(selected_row["peak_summary"]["median_peak"])
    initial_peak = float(initial_row["peak_summary"]["median_peak"])
    binary_budget_valid = all(
        item.binary_material_fraction == 0.25 for item in selected_evaluation.tasks
    )
    status = classify_pilot(
        numerically_valid=True,
        projection_valid=True,
        binary_budget_valid=binary_budget_valid,
        validation_improved=selected_peak < initial_peak,
        matched_condition_wins=causality.matched_win_count,
        source_independent=diversity.mean_hamming_fraction == 0.0,
        median_gradient_gap=gap,
    )
    payload = {
        "schema_version": 1,
        "status": status.value,
        "development_seed": DEVELOPMENT_SEED,
        "microbatch_size": microbatch_size,
        "completed_updates": PILOT_UPDATES,
        "selected_checkpoint": str(selected.relative_to(output_dir)),
        "initial_validation_median_tmax": initial_peak,
        "selected_validation_median_tmax": selected_peak,
        "median_gap_to_direct_gradient": gap,
        "condition_causality": asdict(causality),
        "binary_diversity": asdict(diversity),
        "binary_budget_valid": binary_budget_valid,
    }
    _write_json(verdict_path, payload)
    return payload


def _aggregate_source_hash(repository_root: Path) -> str:
    digest = hashlib.sha256()
    source_paths = sorted((repository_root / "src").rglob("*.py"))
    if not source_paths:
        raise MultitaskGateError("source tree is empty")
    for path in source_paths:
        digest.update(path.relative_to(repository_root).as_posix().encode("utf-8"))
        digest.update(artifact_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _production_registry(output_dir: Path) -> dict[str, object]:
    benchmark = _read_json(output_dir / "benchmark_verdict.json")
    repository_root = Path(__file__).resolve().parents[3]
    registry = create_production_registry(
        updates_per_seed=int(benchmark["production_updates_per_seed"]),
        microbatch_size=int(benchmark["selected_microbatch_size"]),
        training_hours_cap=float(benchmark["remaining_training_hours"]),
        source_sha256=_aggregate_source_hash(repository_root),
        spec_sha256=artifact_sha256(
            repository_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-08-31-multitask-generative-nca-design.md"
        ),
        config_sha256=artifact_sha256(
            repository_root / "configs" / "multitask_nca.yaml"
        ),
    )
    registry_path = output_dir / "production_registry.json"
    if registry_path.is_file():
        existing = _read_json(registry_path)
        validate_production_registry(existing)
        if existing != registry:
            raise MultitaskGateError("production registry changed after lock")
    else:
        _write_json(registry_path, registry)
    return registry


def _leak_safe_provider(
    splits: FrozenTaskSplits,
):
    blocked = frozenset(task.task_id for task in splits.all_tasks)

    def provider(seed: int, update: int, microbatch_index: int) -> SourceLayoutTask:
        return sample_training_task(
            seed,
            update,
            microbatch_index,
            blocked_task_ids=blocked,
        )

    return provider


def _select_production_checkpoint(
    seed_dir: Path,
    splits: FrozenTaskSplits,
) -> tuple[Path, list[dict[str, object]]]:
    summaries = []
    rows: list[dict[str, object]] = []
    for checkpoint in sorted(seed_dir.glob("checkpoint_*.pt")):
        completed = int(checkpoint.stem.split("_")[-1])
        if completed == 0 or completed % 250 != 0:
            continue
        evaluation = evaluate_frozen_checkpoint(
            checkpoint,
            splits.validation,
            split_name="validation",
            device=torch.device("cuda"),
        )
        task_ids = tuple(item.task_id for item in evaluation.tasks)
        peaks = tuple(item.peak_temperature for item in evaluation.tasks)
        summary = summarize_against_reference(
            completed_updates=completed,
            split_name="validation",
            task_ids=task_ids,
            candidate_peaks=peaks,
            reference_peaks={task_id: 1.0 for task_id in task_ids},
        )
        summaries.append(summary)
        rows.append({"checkpoint": checkpoint.name, "summary": asdict(summary)})
    selected_summary = select_validation_checkpoint(summaries)
    selected = seed_dir / f"checkpoint_{selected_summary.completed_updates:06d}.pt"
    _write_json(
        seed_dir / "validation_selection.json",
        {"schema_version": 1, "selected_checkpoint": selected.name, "rows": rows},
    )
    return selected, rows


def run_production(output_dir: Path) -> dict[str, object]:
    """Train exactly three registered models, then freeze by validation only."""
    validate_production_gate(output_dir)
    configure_cuda_reproducibility(PRODUCTION_SEEDS[0])
    registry = _production_registry(output_dir)
    total_updates = int(registry["updates_per_seed"])
    microbatch_size = int(registry["microbatch_size"])
    training_hours_cap = float(registry["training_hours_cap"])
    splits = build_frozen_splits()
    write_split_manifest(output_dir / "split_manifest.json", splits)
    provider = _leak_safe_provider(splits)
    frozen_dir = output_dir / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    campaign_started = time.perf_counter()
    frozen_rows: list[dict[str, object]] = []
    for seed in PRODUCTION_SEEDS:
        frozen_path = frozen_dir / f"frozen_seed_{seed}.pt"
        selection_path = (
            output_dir / "production" / f"seed_{seed}" / "validation_selection.json"
        )
        if frozen_path.is_file() and selection_path.is_file():
            selection = _read_json(selection_path)
            frozen_rows.append(
                {
                    "seed": seed,
                    "selected_checkpoint": selection["selected_checkpoint"],
                    "frozen_checkpoint": str(frozen_path.relative_to(output_dir)),
                    "frozen_sha256": artifact_sha256(frozen_path),
                }
            )
            continue
        seed_dir = output_dir / "production" / f"seed_{seed}"
        resume: Path | None = None
        existing = sorted(seed_dir.glob("checkpoint_*.pt"))
        if existing:
            resume = existing[-1]
        while True:
            result = run_multitask_training(
                config=MultitaskRunConfig(
                    model_seed=seed,
                    task_seed=seed,
                    total_updates=total_updates,
                    microbatch_size=microbatch_size,
                    checkpoint_interval=250,
                    mode="production",
                    device="cuda",
                ),
                output_dir=seed_dir,
                task_provider=provider,
                resume_checkpoint=resume,
                maximum_updates_this_call=250,
                synchronize=torch.cuda.synchronize,
            )
            if result.status is MultitaskRunStatus.INVALID_RUN:
                payload = {
                    "schema_version": 1,
                    "status": "INVALID_RUN",
                    "failed_seed": seed,
                    "reason_codes": list(result.reason_codes),
                }
                _write_json(output_dir / "production_verdict.json", payload)
                return payload
            if time.perf_counter() - campaign_started > training_hours_cap * 3600.0:
                payload = {
                    "schema_version": 1,
                    "status": "INVALID_RUN",
                    "failed_seed": seed,
                    "reason_codes": ["LOCKED_TRAINING_CAP_EXCEEDED"],
                }
                _write_json(output_dir / "production_verdict.json", payload)
                return payload
            resume = result.last_checkpoint
            if result.completed_updates == total_updates:
                break
        selected, _ = _select_production_checkpoint(seed_dir, splits)
        shutil.copy2(selected, frozen_path)
        frozen_rows.append(
            {
                "seed": seed,
                "selected_checkpoint": selected.name,
                "frozen_checkpoint": str(frozen_path.relative_to(output_dir)),
                "frozen_sha256": artifact_sha256(frozen_path),
            }
        )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "production_seeds": list(PRODUCTION_SEEDS),
        "updates_per_seed": total_updates,
        "microbatch_size": microbatch_size,
        "training_wall_seconds": time.perf_counter() - campaign_started,
        "training_hours_cap": training_hours_cap,
        "frozen_models": frozen_rows,
        "test_sets_accessed": False,
    }
    _write_json(output_dir / "production_verdict.json", payload)
    return payload


def lock_production_registry(output_dir: Path) -> dict[str, object]:
    """Freeze production identity once benchmark and pilot authorize it."""
    validate_production_gate(output_dir)
    return _production_registry(output_dir)


def run_production_seed(output_dir: Path, seed: int) -> dict[str, object]:
    """Train and validation-freeze exactly one registered production seed."""
    validate_production_gate(output_dir)
    if seed not in PRODUCTION_SEEDS:
        raise MultitaskGateError("requested production seed is not registered")
    registry_path = output_dir / "production_registry.json"
    if not registry_path.is_file():
        raise MultitaskGateError("production registry must be locked before shards")
    registry = _read_json(registry_path)
    validate_production_registry(registry)
    total_updates = int(registry["updates_per_seed"])
    microbatch_size = int(registry["microbatch_size"])
    training_hours_cap = float(registry["training_hours_cap"])
    splits = build_frozen_splits()
    provider = _leak_safe_provider(splits)
    seed_dir = output_dir / "production" / f"seed_{seed}"
    verdict_path = seed_dir / "production_seed_verdict.json"
    if verdict_path.is_file():
        return _read_json(verdict_path)

    configure_cuda_reproducibility(seed)
    started = time.perf_counter()
    existing = sorted(seed_dir.glob("checkpoint_*.pt"))
    resume: Path | None = existing[-1] if existing else None
    while True:
        result = run_multitask_training(
            config=MultitaskRunConfig(
                model_seed=seed,
                task_seed=seed,
                total_updates=total_updates,
                microbatch_size=microbatch_size,
                checkpoint_interval=250,
                mode="production",
                device="cuda",
            ),
            output_dir=seed_dir,
            task_provider=provider,
            resume_checkpoint=resume,
            maximum_updates_this_call=250,
            synchronize=torch.cuda.synchronize,
        )
        if result.status is MultitaskRunStatus.INVALID_RUN:
            payload = {
                "schema_version": 1,
                "status": "INVALID_RUN",
                "seed": seed,
                "completed_updates": result.completed_updates,
                "reason_codes": list(result.reason_codes),
            }
            _write_json(verdict_path, payload)
            return payload
        if time.perf_counter() - started > training_hours_cap * 3600.0:
            payload = {
                "schema_version": 1,
                "status": "INVALID_RUN",
                "seed": seed,
                "completed_updates": result.completed_updates,
                "reason_codes": ["LOCKED_TRAINING_CAP_EXCEEDED"],
            }
            _write_json(verdict_path, payload)
            return payload
        resume = result.last_checkpoint
        if result.completed_updates == total_updates:
            break

    selected, _ = _select_production_checkpoint(seed_dir, splits)
    frozen_dir = output_dir / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = frozen_dir / f"frozen_seed_{seed}.pt"
    shutil.copy2(selected, frozen_path)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "seed": seed,
        "completed_updates": total_updates,
        "training_wall_seconds": time.perf_counter() - started,
        "selected_checkpoint": str(selected.relative_to(output_dir)),
        "frozen_checkpoint": str(frozen_path.relative_to(output_dir)),
        "frozen_sha256": artifact_sha256(frozen_path),
    }
    _write_json(verdict_path, payload)
    return payload


def finalize_parallel_production(
    output_dir: Path,
    *,
    worker_count: int,
) -> dict[str, object]:
    """Fail closed unless all registered seed shards are complete and frozen."""
    registry = _read_json(output_dir / "production_registry.json")
    validate_production_registry(registry)
    shards = [
        _read_json(
            output_dir / "production" / f"seed_{seed}" / "production_seed_verdict.json"
        )
        for seed in PRODUCTION_SEEDS
    ]
    payload = assemble_production_payload(
        shards,
        updates_per_seed=int(registry["updates_per_seed"]),
        microbatch_size=int(registry["microbatch_size"]),
        training_hours_cap=float(registry["training_hours_cap"]),
        worker_count=worker_count,
    )
    _write_json(output_dir / "production_verdict.json", payload)
    return payload


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise MultitaskGateError(f"cannot write empty metrics table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _baseline_seed(split_name: str, task_index: int, start_index: int) -> int:
    split_offset = 0 if split_name == "test_id" else 1000
    return 2026083400 + split_offset + task_index * 4 + start_index


def _run_gradient_candidate(
    output_dir: Path,
    *,
    split_name: str,
    task_index: int,
    task: SourceLayoutTask,
    start_index: int,
) -> dict[str, object]:
    candidate_dir = (
        output_dir
        / "test_baselines"
        / split_name
        / task.task_id
        / f"start_{start_index}"
    )
    result_path = candidate_dir / "result.json"
    design_path = candidate_dir / "binary_design_64.npy"
    if result_path.is_file() and design_path.is_file():
        return _read_json(result_path)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    seed = _baseline_seed(split_name, task_index, start_index)
    configure_cuda_reproducibility(seed)
    sources = torch.as_tensor(task.sources, dtype=torch.float64, device="cuda")
    started = time.perf_counter()
    result = optimize_design(
        sources,
        seed=seed,
        config=OptimizationConfig(enforce_final_binary_budget=False),
        output_dir=None,
    )
    if result.status is Gate2Status.INVALID_RUN or result.continuous_design is None:
        raise MultitaskGateError(
            f"direct-gradient candidate failed for {task.task_id} start {start_index}"
        )
    binary, budget = exact_cardinality_binary(result.continuous_design)
    peak_64 = _score_binary_design(binary, task, device=torch.device("cuda"))
    design = binary.detach().cpu().numpy().astype(np.float64, copy=False)
    np.save(design_path, design, allow_pickle=False)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "optimizer": "WaveForge_direct_gradient_600_steps",
        "split": split_name,
        "task_id": task.task_id,
        "task_index": task_index,
        "start_index": start_index,
        "optimizer_seed": seed,
        "completed_iterations": result.completed_iterations,
        "final_logits_hash": result.final_logits_hash,
        "config_sha256": result.config_sha256,
        "binary_material_fraction": budget.material_fraction,
        "peak_64": peak_64,
        "wall_seconds": time.perf_counter() - started,
        "binary_design": str(design_path.relative_to(output_dir)),
        "binary_design_sha256": artifact_sha256(design_path),
    }
    _write_json(result_path, payload)
    return payload


def _run_registered_test_baselines(
    output_dir: Path,
    splits: FrozenTaskSplits,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    registry = registered_test_baseline_jobs()
    _write_json(
        output_dir / "test_baseline_registry.json",
        {"schema_version": 1, "jobs": list(registry)},
    )
    task_lookup = {
        (split_name, task.task_id): (index, task)
        for split_name, tasks in (
            ("test_id", splits.test_id),
            ("test_ood", splits.test_ood),
        )
        for index, task in enumerate(tasks)
    }
    candidates: dict[tuple[str, str, int], dict[str, object]] = {}
    for job in registry:
        split_name = str(job["split"])
        task_id = str(job["task_id"])
        start_index = int(job["start_index"])
        task_index, task = task_lookup[(split_name, task_id)]
        key = (split_name, task_id, start_index)
        if key not in candidates:
            candidates[key] = _run_gradient_candidate(
                output_dir,
                split_name=split_name,
                task_index=task_index,
                task=task,
                start_index=start_index,
            )

    single: dict[str, dict[str, object]] = {}
    for task in splits.test_id:
        single[task.task_id] = candidates[("test_id", task.task_id, 0)]

    multistart_rows: list[dict[str, object]] = []
    for split_name, tasks in (
        ("test_id", splits.test_id[:8]),
        ("test_ood", splits.test_ood[:8]),
    ):
        for task in tasks:
            group = [
                candidates[(split_name, task.task_id, start_index)]
                for start_index in range(4)
            ]
            winner = min(group, key=lambda item: float(item["peak_64"]))
            multistart_rows.append(
                {
                    "split": split_name,
                    "task_id": task.task_id,
                    "selected_start_index": winner["start_index"],
                    "selected_peak_64": winner["peak_64"],
                    "candidate_peaks_64": json.dumps(
                        [float(item["peak_64"]) for item in group]
                    ),
                    "selection_rule": "minimum_unrounded_64x64_peak",
                }
            )
    _write_csv(output_dir / "multistart_selection.csv", multistart_rows)
    return single, multistart_rows


def _independent_reference_peaks(
    output_dir: Path,
    tasks: tuple[SourceLayoutTask, ...],
    candidates: dict[str, dict[str, object]],
) -> dict[str, float]:
    path = output_dir / "test_gradient_verified_256.json"
    if path.is_file():
        payload = _read_json(path)
        return {
            str(task_id): float(value)
            for task_id, value in dict(payload["worst_peaks"]).items()
        }
    peaks: dict[str, float] = {}
    records: list[dict[str, object]] = []
    for task in tasks:
        design_path = output_dir / str(candidates[task.task_id]["binary_design"])
        design = np.load(design_path, allow_pickle=False)
        verified = verify_binary_task(design, task, resolution=256)
        if verified.material_fraction != 0.25:
            raise MultitaskGateError("gradient comparator material budget changed")
        peaks[task.task_id] = verified.worst_peak
        records.append(asdict(verified))
    _write_json(
        path,
        {
            "schema_version": 1,
            "solver": "independent_CPU_SciPy_256",
            "worst_peaks": peaks,
            "records": records,
        },
    )
    return peaks


def run_unseen_test(output_dir: Path) -> dict[str, object]:
    """Evaluate frozen production NCA models on untouched ID/OOD layouts."""
    verdict_path = output_dir / "campaign_verdict.json"
    if verdict_path.is_file():
        return _read_json(verdict_path)
    production = _read_json(output_dir / "production_verdict.json")
    if (
        production.get("status") != "PASS"
        or production.get("test_sets_accessed") is not False
    ):
        raise MultitaskGateError("test phase requires untouched PASS production")
    splits = build_frozen_splits()
    write_split_manifest(output_dir / "split_manifest.json", splits)
    single_references, multistart_rows = _run_registered_test_baselines(
        output_dir,
        splits,
    )
    gradient_peaks = _independent_reference_peaks(
        output_dir,
        splits.test_id,
        single_references,
    )

    seed_summaries = []
    id_rows: list[dict[str, object]] = []
    ood_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(PRODUCTION_SEEDS):
        checkpoint = output_dir / "frozen" / f"frozen_seed_{seed}.pt"
        if not checkpoint.is_file():
            raise MultitaskGateError(f"frozen production seed is missing: {seed}")
        configure_cuda_reproducibility(seed)
        id_evaluation = evaluate_frozen_checkpoint(
            checkpoint,
            splits.test_id,
            split_name="test_id",
            device=torch.device("cuda"),
        )
        ood_evaluation = evaluate_frozen_checkpoint(
            checkpoint,
            splits.test_ood,
            split_name="test_ood",
            device=torch.device("cuda"),
        )
        shuffled = evaluate_frozen_checkpoint(
            checkpoint,
            splits.test_id,
            split_name="test_id",
            conditioning_tasks=splits.test_id[1:] + splits.test_id[:1],
            device=torch.device("cuda"),
        )
        causality = condition_causality_summary(
            matched=[item.peak_temperature for item in id_evaluation.tasks],
            shuffled=[item.peak_temperature for item in shuffled.tasks],
        )
        np.savez_compressed(
            output_dir / f"frozen_seed_{seed}_test_designs.npz",
            test_id=np.stack([item.binary_design for item in id_evaluation.tasks]),
            test_ood=np.stack([item.binary_design for item in ood_evaluation.tasks]),
        )
        nca_verified_peaks: list[float] = []
        valid = True
        for task, item in zip(splits.test_id, id_evaluation.tasks, strict=True):
            verified = verify_binary_task(item.binary_design, task, resolution=256)
            valid = bool(
                valid
                and item.binary_material_fraction == 0.25
                and verified.material_fraction == 0.25
                and verified.maximum_normalized_residual <= 1.0e-8
            )
            nca_verified_peaks.append(verified.worst_peak)
            reference = gradient_peaks[task.task_id]
            id_rows.append(
                {
                    "seed": seed,
                    "task_id": task.task_id,
                    "nca_tmax_256": verified.worst_peak,
                    "gradient_tmax_256": reference,
                    "relative_gap": (verified.worst_peak - reference) / reference,
                    "nca_material_fraction": verified.material_fraction,
                    "nca_design_hash_64": verified.design_hash_64,
                    "maximum_normalized_residual": verified.maximum_normalized_residual,
                }
            )
        for task, item in zip(splits.test_ood, ood_evaluation.tasks, strict=True):
            verified = verify_binary_task(item.binary_design, task, resolution=256)
            valid = bool(
                valid
                and item.binary_material_fraction == 0.25
                and verified.material_fraction == 0.25
                and verified.maximum_normalized_residual <= 1.0e-8
            )
            ood_rows.append(
                {
                    "seed": seed,
                    "task_id": task.task_id,
                    "nca_tmax_256": verified.worst_peak,
                    "nca_material_fraction": verified.material_fraction,
                    "nca_design_hash_64": verified.design_hash_64,
                    "maximum_normalized_residual": verified.maximum_normalized_residual,
                }
            )
        seed_summaries.append(
            summarize_seed(
                seed=seed,
                nca_peaks=tuple(nca_verified_peaks),
                gradient_peaks=tuple(
                    gradient_peaks[task.task_id] for task in splits.test_id
                ),
                bootstrap_seed=2026083500 + seed_index,
                bootstrap_resamples=10_000,
                condition_matched_wins=causality.matched_win_count,
                valid=valid,
            )
        )
    _write_csv(output_dir / "test_id_verified_metrics.csv", id_rows)
    _write_csv(output_dir / "test_ood_verified_metrics.csv", ood_rows)
    campaign = classify_campaign(seed_summaries)
    payload = {
        "schema_version": 1,
        "status": campaign.status.value,
        "passing_seed_count": campaign.passing_seed_count,
        "better_tested_gradient_seed_count": (
            campaign.better_tested_gradient_seed_count
        ),
        "seeds": [asdict(item) for item in campaign.seeds],
        "primary_split": "test_id",
        "primary_task_count": len(splits.test_id),
        "secondary_ood_task_count": len(splits.test_ood),
        "direct_gradient_comparator": "WaveForge_direct_gradient_600_steps",
        "multistart_challenge_task_count": len(multistart_rows),
        "final_authority": "independent_CPU_SciPy_256",
    }
    _write_json(verdict_path, payload)
    production["test_sets_accessed"] = True
    production["test_verdict_artifact"] = "campaign_verdict.json"
    _write_json(output_dir / "production_verdict.json", production)
    return payload


def write_campaign_hashes(output_dir: Path) -> dict[str, object]:
    """Hash all completed compact and frozen artifacts before download."""
    required = [
        output_dir / "benchmark_verdict.json",
        output_dir / "pilot_verdict.json",
        output_dir / "production_registry.json",
        output_dir / "production_verdict.json",
        output_dir / "split_manifest.json",
    ] + [output_dir / "frozen" / f"frozen_seed_{seed}.pt" for seed in PRODUCTION_SEEDS]
    readiness = validate_backup_readiness(required)
    paths = [path for path in output_dir.rglob("*") if path.is_file()]
    manifest = build_hash_manifest(paths, root=output_dir)
    _write_json(
        output_dir / "hash_manifest.json",
        {"schema_version": 1, "artifacts": manifest},
    )
    readiness["hash_manifest"] = "hash_manifest.json"
    _write_json(output_dir / "backup_ready.json", readiness)
    return readiness


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit phase-only command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "preflight",
            "benchmark",
            "budget",
            "pilot",
            "production-lock",
            "production-seed",
            "production-finalize",
            "production",
            "test",
            "hashes",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remaining-hours", type=float, default=8.0)
    parser.add_argument("--maximum-campaign-cost-usd", type=float, default=7.0)
    parser.add_argument("--hourly-cost-usd", type=float, default=0.633)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker-count", type=int, default=1)
    return parser


def main() -> None:
    """Execute exactly one guarded campaign phase."""
    args = build_parser().parse_args()
    if args.phase == "preflight":
        run_preflight(args.output)
    elif args.phase == "benchmark":
        run_benchmark(args.output, remaining_hours=args.remaining_hours)
    elif args.phase == "budget":
        lock_runtime_budget_amendment(
            args.output,
            production_training_hours=args.remaining_hours,
            maximum_campaign_cost_usd=args.maximum_campaign_cost_usd,
            hourly_cost_usd=args.hourly_cost_usd,
        )
    elif args.phase == "pilot":
        run_pilot(args.output)
    elif args.phase == "production-lock":
        lock_production_registry(args.output)
    elif args.phase == "production-seed":
        if args.seed is None:
            raise MultitaskGateError("production-seed requires --seed")
        run_production_seed(args.output, args.seed)
    elif args.phase == "production-finalize":
        finalize_parallel_production(
            args.output,
            worker_count=args.worker_count,
        )
    elif args.phase == "production":
        run_production(args.output)
    elif args.phase == "test":
        run_unseen_test(args.output)
    elif args.phase == "hashes":
        write_campaign_hashes(args.output)
    else:
        raise MultitaskGateError(f"phase {args.phase} is not implemented yet")


if __name__ == "__main__":
    main()
