"""Generate solver-consistent MT2B validation reference designs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from waveforge.design.binary_readout import exact_cardinality_binary
from waveforge.design.optimize import OptimizationConfig, optimize_design
from waveforge.ml.mt2b_evaluation import independent_scipy64_tmax
from waveforge.ml.mt2b_reference_optimization import (
    ReferenceOptimizationResult,
    optimize_reference_scenario_batched,
)
from waveforge.ml.multitask_tasks import (
    VALIDATION_SEED,
    SourceLayoutTask,
    sample_primary_task,
)
from waveforge.reproducibility import (
    artifact_sha256,
    configure_cuda_reproducibility,
)

PROTOCOL_BUNDLE_SHA256 = (
    "567606c870720ca48001868efa9db1c6918e42345a1892932826c1ab0691d103"
)
ReferenceBackend = Literal[
    "legacy_sequential_implicit", "scenario_vectorized_b1_implicit"
]


class MT2BReferenceError(RuntimeError):
    """Fail-closed reference provenance or numerical error."""


@dataclass(frozen=True)
class ReferenceJob:
    split_name: str
    task_index: int
    task_id: str
    optimizer_seed: int
    task: SourceLayoutTask


@dataclass(frozen=True)
class _ReferenceOutcome:
    completed_iterations: int
    final_logits_sha256: str
    continuous_design: torch.Tensor
    binary_design: torch.Tensor
    binary_material_fraction: float
    records: tuple[dict[str, object], ...]
    wall_seconds: float


def build_reference_job(task_index: int) -> ReferenceJob:
    """Build one of the exact 32 preregistered validation jobs."""
    if (
        isinstance(task_index, bool)
        or not isinstance(task_index, int)
        or not 0 <= task_index < 32
    ):
        raise ValueError("reference task_index must lie in [0,32)")
    task = sample_primary_task(VALIDATION_SEED, task_index)
    return ReferenceJob(
        split_name="validation",
        task_index=task_index,
        task_id=task.task_id,
        optimizer_seed=2026083200 + task_index,
        task=task,
    )


def _validate_binary(design: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(design, dtype=np.float64)
    if array.shape != (64, 64) or not np.isin(array, (0.0, 1.0)).all():
        raise MT2BReferenceError("reference design must be binary with shape [64,64]")
    if int(np.count_nonzero(array)) != 1024:
        raise MT2BReferenceError("reference design must contain exactly 1024 cells")
    return array


def load_reference_artifact(
    result_path: Path,
    design_path: Path,
    *,
    expected: ReferenceJob,
) -> dict[str, object]:
    """Validate and load one complete reference result without partial reuse."""
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        design = _validate_binary(np.load(design_path, allow_pickle=False))
    except (OSError, ValueError) as error:
        raise MT2BReferenceError("reference artifact is unreadable") from error
    exact_fields = {
        "schema_version": 1,
        "status": "PASS",
        "split": expected.split_name,
        "task_index": expected.task_index,
        "task_id": expected.task_id,
        "optimizer_seed": expected.optimizer_seed,
        "completed_iterations": 600,
        "binary_cell_count": 1024,
        "binary_material_fraction": 0.25,
    }
    if any(payload.get(key) != value for key, value in exact_fields.items()):
        raise MT2BReferenceError("reference artifact identity is invalid")
    if payload.get("binary_design_sha256") != artifact_sha256(design_path):
        raise MT2BReferenceError("reference design hash mismatch")
    return {**payload, "binary_design": design}


def validate_acceleration_qualification(
    sequential_design: NDArray[np.float64],
    scenario_batched_design: NDArray[np.float64],
    *,
    sequential_tmax: float,
    scenario_batched_tmax: float,
) -> dict[str, object]:
    """Accept acceleration only when the final scientific design is identical."""
    sequential = _validate_binary(sequential_design)
    batched = _validate_binary(scenario_batched_design)
    if (
        not math.isfinite(sequential_tmax)
        or not math.isfinite(scenario_batched_tmax)
        or sequential_tmax <= 0.0
        or scenario_batched_tmax <= 0.0
    ):
        raise MT2BReferenceError(
            "qualification Tmax values must be finite and positive"
        )
    exact_match = bool(np.array_equal(sequential, batched))
    tmax_error = abs(sequential_tmax - scenario_batched_tmax)
    return {
        "schema_version": 1,
        "qualification_task_index": 0,
        "sequential_backend": "legacy_sequential_implicit",
        "candidate_backend": "scenario_vectorized_b1_implicit",
        "binary_design_exact_match": exact_match,
        "scipy64_tmax_absolute_error": tmax_error,
        "scipy64_tmax_tolerance": 1.0e-12,
        "accepted": bool(exact_match and tmax_error <= 1.0e-12),
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _atomic_npy(path: Path, array: NDArray[np.float64]) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(path)


def _atomic_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    if not rows:
        raise MT2BReferenceError("reference metrics cannot be empty")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sequential_outcome(job: ReferenceJob, device: torch.device) -> _ReferenceOutcome:
    sources = torch.as_tensor(job.task.sources, dtype=torch.float64, device=device)
    started = time.perf_counter()
    result = optimize_design(
        sources,
        seed=job.optimizer_seed,
        config=OptimizationConfig(enforce_final_binary_budget=False),
        output_dir=None,
    )
    elapsed = time.perf_counter() - started
    if (
        result.status.value == "INVALID_RUN"
        or result.completed_iterations != 600
        or result.continuous_design is None
    ):
        raise MT2BReferenceError("legacy sequential reference became invalid")
    binary, budget = exact_cardinality_binary(result.continuous_design, count=1024)
    return _ReferenceOutcome(
        completed_iterations=result.completed_iterations,
        final_logits_sha256=result.final_logits_hash,
        continuous_design=result.continuous_design,
        binary_design=binary,
        binary_material_fraction=budget.material_fraction,
        records=tuple(asdict(record) for record in result.records),
        wall_seconds=elapsed,
    )


def _scenario_batched_outcome(
    job: ReferenceJob, device: torch.device
) -> _ReferenceOutcome:
    sources = torch.as_tensor(job.task.sources, dtype=torch.float64, device=device)
    started = time.perf_counter()
    result: ReferenceOptimizationResult = optimize_reference_scenario_batched(
        sources,
        seed=job.optimizer_seed,
    )
    elapsed = time.perf_counter() - started
    if result.completed_iterations != 600:
        raise MT2BReferenceError("scenario-vectorized reference became incomplete")
    return _ReferenceOutcome(
        completed_iterations=result.completed_iterations,
        final_logits_sha256=result.final_logits_sha256,
        continuous_design=result.continuous_design,
        binary_design=result.binary_design,
        binary_material_fraction=result.binary_material_fraction,
        records=tuple(asdict(record) for record in result.records),
        wall_seconds=elapsed,
    )


def _run_backend(job: ReferenceJob, backend: ReferenceBackend) -> _ReferenceOutcome:
    configure_cuda_reproducibility(job.optimizer_seed)
    device = torch.device("cuda")
    if backend == "legacy_sequential_implicit":
        return _sequential_outcome(job, device)
    if backend == "scenario_vectorized_b1_implicit":
        return _scenario_batched_outcome(job, device)
    raise ValueError(f"unsupported reference backend {backend!r}")


def _write_outcome(
    directory: Path,
    *,
    job: ReferenceJob,
    backend: ReferenceBackend,
    outcome: _ReferenceOutcome,
    execution_source_sha: str,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    continuous_path = directory / "continuous_design_64.npy"
    binary_path = directory / "binary_design_64.npy"
    result_path = directory / "reference_result.json"
    metrics_path = directory / "optimization_metrics.csv"
    continuous = outcome.continuous_design.numpy().astype(np.float64, copy=False)
    binary = _validate_binary(
        outcome.binary_design.numpy().astype(np.float64, copy=False)
    )
    _atomic_npy(continuous_path, continuous)
    _atomic_npy(binary_path, binary)
    _atomic_csv(metrics_path, outcome.records)
    peak = independent_scipy64_tmax(binary, job.task)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "split": "validation",
        "task_index": job.task_index,
        "task_id": job.task_id,
        "optimizer_seed": job.optimizer_seed,
        "optimizer": "WaveForge_direct_gradient_600_steps",
        "physics_backend": backend,
        "completed_iterations": outcome.completed_iterations,
        "binary_cell_count": 1024,
        "binary_material_fraction": outcome.binary_material_fraction,
        "final_logits_sha256": outcome.final_logits_sha256,
        "continuous_design_sha256": artifact_sha256(continuous_path),
        "binary_design_sha256": artifact_sha256(binary_path),
        "optimization_metrics_sha256": artifact_sha256(metrics_path),
        "independent_scipy64_tmax": peak,
        "wall_seconds": outcome.wall_seconds,
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA256,
        "execution_source_sha": execution_source_sha,
        "validation_accessed": True,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }
    _atomic_json(result_path, payload)
    return payload


def _existing_reference(
    directory: Path, *, job: ReferenceJob
) -> dict[str, object] | None:
    result_path = directory / "reference_result.json"
    binary_path = directory / "binary_design_64.npy"
    if not result_path.exists() and not binary_path.exists():
        return None
    return load_reference_artifact(result_path, binary_path, expected=job)


def _git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_acceleration_qualification(
    output_root: Path,
    *,
    execution_source_sha: str,
) -> dict[str, object]:
    """Compare full 600-step backends before using the faster candidate."""
    path = output_root / "reference_acceleration_qualification.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    job = build_reference_job(0)
    root = output_root / "reference_qualification" / "task_00"
    sequential = _run_backend(job, "legacy_sequential_implicit")
    sequential_payload = _write_outcome(
        root / "sequential",
        job=job,
        backend="legacy_sequential_implicit",
        outcome=sequential,
        execution_source_sha=execution_source_sha,
    )
    candidate = _run_backend(job, "scenario_vectorized_b1_implicit")
    candidate_payload = _write_outcome(
        root / "scenario_batched",
        job=job,
        backend="scenario_vectorized_b1_implicit",
        outcome=candidate,
        execution_source_sha=execution_source_sha,
    )
    qualification = validate_acceleration_qualification(
        sequential.binary_design.numpy(),
        candidate.binary_design.numpy(),
        sequential_tmax=float(sequential_payload["independent_scipy64_tmax"]),
        scenario_batched_tmax=float(candidate_payload["independent_scipy64_tmax"]),
    )
    selected: ReferenceBackend = (
        "scenario_vectorized_b1_implicit"
        if qualification["accepted"]
        else "legacy_sequential_implicit"
    )
    payload = {
        **qualification,
        "selected_backend": selected,
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA256,
        "execution_source_sha": execution_source_sha,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }
    _atomic_json(path, payload)
    official = output_root / "references" / "task_00"
    _write_outcome(
        official,
        job=job,
        backend="legacy_sequential_implicit",
        outcome=sequential,
        execution_source_sha=execution_source_sha,
    )
    return payload


def run_all_references(
    output_root: Path,
    *,
    execution_source_sha: str,
) -> dict[str, object]:
    """Generate each missing validation reference exactly once."""
    qualification_path = output_root / "reference_acceleration_qualification.json"
    if not qualification_path.exists():
        raise MT2BReferenceError("reference acceleration qualification is missing")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    backend = str(qualification.get("selected_backend"))
    if backend not in {
        "legacy_sequential_implicit",
        "scenario_vectorized_b1_implicit",
    }:
        raise MT2BReferenceError("qualified reference backend is invalid")
    rows: list[dict[str, object]] = []
    for task_index in range(32):
        job = build_reference_job(task_index)
        directory = output_root / "references" / f"task_{task_index:02d}"
        existing = _existing_reference(directory, job=job)
        if existing is None:
            outcome = _run_backend(job, backend)  # type: ignore[arg-type]
            existing = _write_outcome(
                directory,
                job=job,
                backend=backend,  # type: ignore[arg-type]
                outcome=outcome,
                execution_source_sha=execution_source_sha,
            )
        rows.append(
            {key: value for key, value in existing.items() if key != "binary_design"}
        )
        print(
            f"MT2B_REFERENCE_PROGRESS tasks={task_index + 1}/32 backend={backend}",
            flush=True,
        )
    registry = {
        "schema_version": 1,
        "status": "PASS",
        "task_count": 32,
        "split": "validation",
        "selected_backend": backend,
        "jobs": rows,
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA256,
        "execution_source_sha": execution_source_sha,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }
    _atomic_json(output_root / "reference_registry.json", registry)
    return registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("qualify", "references"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    root = arguments.root.resolve()
    output = arguments.output.resolve()
    if _git_sha(root) != arguments.source_sha:
        raise MT2BReferenceError("working tree HEAD does not match source SHA")
    if arguments.phase == "qualify":
        run_acceleration_qualification(
            output,
            execution_source_sha=arguments.source_sha,
        )
    else:
        run_all_references(
            output,
            execution_source_sha=arguments.source_sha,
        )


if __name__ == "__main__":
    main()
