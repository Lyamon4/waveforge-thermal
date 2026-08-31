"""Paid-A100 gates for the prospective multi-task generative NCA campaign."""

from __future__ import annotations

import argparse
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
    """Allocate at most six hours equally and require 5k updates per seed."""
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
            if time.perf_counter() - campaign_started > 6.0 * 3600.0:
                payload = {
                    "schema_version": 1,
                    "status": "INVALID_RUN",
                    "failed_seed": seed,
                    "reason_codes": ["SIX_HOUR_TRAINING_CAP_EXCEEDED"],
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
        "frozen_models": frozen_rows,
        "test_sets_accessed": False,
    }
    _write_json(output_dir / "production_verdict.json", payload)
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
        choices=("preflight", "benchmark", "pilot", "production", "test", "hashes"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remaining-hours", type=float, default=6.0)
    return parser


def main() -> None:
    """Execute exactly one guarded campaign phase."""
    args = build_parser().parse_args()
    if args.phase == "preflight":
        run_preflight(args.output)
    elif args.phase == "benchmark":
        run_benchmark(args.output, remaining_hours=args.remaining_hours)
    elif args.phase == "pilot":
        run_pilot(args.output)
    elif args.phase == "production":
        run_production(args.output)
    elif args.phase == "hashes":
        write_campaign_hashes(args.output)
    else:
        raise MultitaskGateError(f"phase {args.phase} is not implemented yet")


if __name__ == "__main__":
    main()
