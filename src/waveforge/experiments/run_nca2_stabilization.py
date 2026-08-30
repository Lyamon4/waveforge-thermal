"""Prospective stabilized pure-NCA campaign orchestration."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from waveforge.environment import collect_environment
from waveforge.experiments.run_inverse_design import gate2_source_batch
from waveforge.ml.nca2_training import ScheduledNCAController
from waveforge.ml.nca_training import NCARunResult, NCARunStatus, run_nca_training
from waveforge.reproducibility import (
    artifact_sha256,
    configure_cuda_reproducibility,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs/nca2_stabilization.yaml"
SPEC_PATH = (
    PROJECT_ROOT
    / "docs/superpowers/specs/2026-08-30-nca2-stabilized-training-design.md"
)
OLD_VERDICT_PATH = PROJECT_ROOT / "artifacts/pure_nca_spike/nca_spike_verdict.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def benchmark_revised_loop(
    *,
    sources: object,
    training_runner: Callable[..., NCARunResult] = run_nca_training,
    synchronizer: Callable[[], None] = torch.cuda.synchronize,
    reset_peak_memory: Callable[[], None] = torch.cuda.reset_peak_memory_stats,
    peak_allocated_memory: Callable[[], int] = torch.cuda.max_memory_allocated,
    peak_reserved_memory: Callable[[], int] = torch.cuda.max_memory_reserved,
) -> dict[str, Any]:
    """Measure the final objective stage and project the full locked campaign."""
    warmup_steps = 3
    measured_steps = 10
    expected_steps = warmup_steps + measured_steps
    controller = ScheduledNCAController("B")

    def configure_final_stage(
        iteration: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        controller.configure(500 + iteration, optimizer)

    def iteration_start_hook(iteration: int) -> None:
        if iteration == warmup_steps:
            reset_peak_memory()

    result = training_runner(
        sources=sources,
        seed=20260910,
        learning_rate=1.0e-4,
        iterations=expected_steps,
        mode="benchmark",
        output_dir=None,
        evaluator=controller.evaluate,
        checkpoint_interval=50,
        synchronize=synchronizer,
        iteration_start_hook=iteration_start_hook,
        iteration_configurator=configure_final_stage,
    )
    if (
        result.status is not NCARunStatus.PASS
        or result.completed_iterations != expected_steps
    ):
        raise RuntimeError(
            f"revised-loop benchmark requires exactly {expected_steps} PASS records"
        )
    samples = np.asarray(
        [record.wall_seconds for record in result.records[warmup_steps:]],
        dtype=np.float64,
    )
    if samples.shape != (measured_steps,) or not np.isfinite(samples).all():
        raise RuntimeError(
            "revised-loop benchmark samples are incomplete or non-finite"
        )
    mean = float(np.mean(samples))
    qualification_updates = 4200
    production_updates = 4500
    total_updates = qualification_updates + production_updates
    return {
        "schema_version": 1,
        "status": "PASS",
        "benchmark_seed": 20260910,
        "schedule_iteration_offset": 500,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "samples_seconds": samples.tolist(),
        "median_step_seconds": float(np.median(samples)),
        "p90_step_seconds": float(np.percentile(samples, 90)),
        "mean_step_seconds": mean,
        "standard_deviation_seconds": float(np.std(samples)),
        "peak_allocated_bytes": int(peak_allocated_memory()),
        "peak_reserved_bytes": int(peak_reserved_memory()),
        "qualification_updates": qualification_updates,
        "production_updates": production_updates,
        "total_updates": total_updates,
        "projected_qualification_hours": mean * qualification_updates / 3600.0,
        "projected_production_hours": mean * production_updates / 3600.0,
        "projected_gpu_hours": mean * total_updates / 3600.0,
    }


def validate_runtime_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the inclusive 6.6-hour stop boundary without changing protocol."""
    projected = float(report["projected_gpu_hours"])
    maximum = 6.6
    authorized = bool(np.isfinite(projected) and projected <= maximum)
    return {
        **report,
        "status": "PASS" if authorized else "NCA2_RUNTIME_REVIEW_REQUIRED",
        "qualification_authorized": authorized,
        "maximum_projected_gpu_hours": maximum,
    }


def build_protocol_manifest(
    *,
    benchmark: dict[str, Any],
    config_path: Path,
    spec_path: Path,
    old_verdict_path: Path,
    implementation_git_sha: str,
    determinism: dict[str, Any],
) -> dict[str, Any]:
    """Bind the runtime decision to locked inputs and immutable Experiment 1."""
    old_verdict = json.loads(old_verdict_path.read_text(encoding="utf-8"))
    if old_verdict.get("status") != "NCA_NO_GO_EFFECT":
        raise RuntimeError("old pure-NCA verdict is not the immutable NO-GO result")
    return {
        "schema_version": 1,
        "status": benchmark["status"],
        "qualification_authorized": benchmark["qualification_authorized"],
        "config_sha256": artifact_sha256(config_path),
        "spec_sha256": artifact_sha256(spec_path),
        "old_experiment_status": old_verdict["status"],
        "old_verdict_sha256": artifact_sha256(old_verdict_path),
        "implementation_git_sha": implementation_git_sha,
        "determinism": determinism,
    }


def run_benchmark_phase(output_dir: Path) -> dict[str, Any]:
    """Run the blocking CUDA benchmark and persist the runtime decision."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; NCA-2 cannot benchmark on CPU")
    determinism = configure_cuda_reproducibility(20260910)
    sources = gate2_source_batch(device=torch.device("cuda"))
    gated = validate_runtime_gate(benchmark_revised_loop(sources=sources))
    environment = collect_environment(
        "pip3 install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu130",
        determinism=determinism,
    )
    manifest = build_protocol_manifest(
        benchmark=gated,
        config_path=CONFIG_PATH,
        spec_path=SPEC_PATH,
        old_verdict_path=OLD_VERDICT_PATH,
        implementation_git_sha=_git_sha(),
        determinism=asdict(determinism),
    )
    _write_json(output_dir / "revised_loop_benchmark.json", gated)
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "protocol_manifest.json", manifest)
    return gated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("benchmark",))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/nca2_stabilization",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase == "benchmark":
        run_benchmark_phase(args.output)


if __name__ == "__main__":
    main()
