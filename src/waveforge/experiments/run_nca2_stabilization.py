"""Prospective stabilized pure-NCA campaign orchestration."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from waveforge.environment import collect_environment
from waveforge.experiments.run_inverse_design import gate2_source_batch
from waveforge.ml.nca import PureNCA, build_static_condition, project_nca_material
from waveforge.ml.nca2_qualification import (
    DevelopmentSeedMetrics,
    NCA2QualificationVerdict,
    classify_development_seed,
    select_nca2_protocol,
    summarize_protocol,
)
from waveforge.ml.nca2_schedule import objective_settings_at
from waveforge.ml.nca2_training import ScheduledNCAController
from waveforge.ml.nca_training import (
    NCARunResult,
    NCARunStatus,
    model_state_sha256,
    run_nca_training,
)
from waveforge.reproducibility import (
    artifact_sha256,
    configure_cuda_reproducibility,
)
from waveforge.verification.high_fidelity import verify_candidate
from waveforge.verification.nca_verification import (
    NCAConnectivityDiagnostic,
    connectivity_diagnostic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs/nca2_stabilization.yaml"
SPEC_PATH = (
    PROJECT_ROOT
    / "docs/superpowers/specs/2026-08-30-nca2-stabilized-training-design.md"
)
OLD_VERDICT_PATH = PROJECT_ROOT / "artifacts/pure_nca_spike/nca_spike_verdict.json"
DEVELOPMENT_SEEDS = (20260901, 20260902, 20260903)
QUALIFICATION_CHECKPOINTS = (500, 550, 600, 650, 700)


@dataclass(frozen=True)
class QualificationRun:
    """One registered development run and its immutable location."""

    protocol_id: str
    seed: int
    run_dir: Path
    result: Any

    @property
    def initial_model_hash(self) -> str:
        return str(self.result.initial_model_hash)


@dataclass(frozen=True)
class FrozenNCA2Design:
    """Strict post-update design regenerated from a registered checkpoint."""

    continuous_design: np.ndarray
    binary_design: np.ndarray
    checkpoint_model_hash: str


@dataclass(frozen=True)
class QualificationCheckpointDiagnostic:
    """Independent low-grid thermal and engineering checkpoint diagnostic."""

    protocol_id: str
    seed: int
    completed_updates: int
    worst_peak: float
    binary_fraction: float
    connectivity: NCAConnectivityDiagnostic


class NCA2GateError(RuntimeError):
    """A locked phase input is missing, inconsistent or unauthorized."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(frame.to_csv(index=False), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


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


def execute_qualification_runs(
    *,
    output_dir: Path,
    sources: object,
    training_runner: Callable[..., NCARunResult] = run_nca_training,
    synchronizer: Callable[[], None] = torch.cuda.synchronize,
    seed_configurator: Callable[[int], Any] = configure_cuda_reproducibility,
) -> tuple[QualificationRun, ...]:
    """Execute the exact two-by-three prospective development registry."""
    runs: list[QualificationRun] = []
    for protocol_id in ("A", "B"):
        for seed in DEVELOPMENT_SEEDS:
            seed_configurator(seed)
            controller = ScheduledNCAController(protocol_id)
            run_dir = (
                output_dir / "qualification" / f"protocol_{protocol_id}" / str(seed)
            )
            result = training_runner(
                sources=sources,
                seed=seed,
                learning_rate=1.0e-3,
                iterations=700,
                mode="qualification",
                output_dir=run_dir,
                evaluator=controller.evaluate,
                checkpoint_interval=50,
                synchronize=synchronizer,
                iteration_configurator=controller.configure,
            )
            runs.append(
                QualificationRun(
                    protocol_id=protocol_id,
                    seed=seed,
                    run_dir=run_dir,
                    result=result,
                )
            )
    return tuple(runs)


def freeze_nca2_checkpoint(
    *,
    checkpoint_path: Path,
    sources: torch.Tensor,
    completed_updates: int,
    protocol_id: str,
) -> FrozenNCA2Design:
    """Regenerate the strict design for one exact post-update checkpoint."""
    del protocol_id
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("completed_updates") != completed_updates
        or checkpoint.get("last_iteration") != completed_updates - 1
    ):
        raise RuntimeError("qualification checkpoint metadata mismatch")
    model = PureNCA().to(device=sources.device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    checkpoint_hash = model_state_sha256(model)
    if checkpoint_hash != checkpoint.get("model_state_sha256"):
        raise RuntimeError("qualification checkpoint model hash mismatch")
    settings = objective_settings_at(completed_updates - 1)
    with torch.no_grad():
        rollout = model.rollout(build_static_condition(sources))
        projected = project_nca_material(
            rollout.material_logit,
            beta=settings.projection_beta,
        ).design
        binary = (projected >= 0.5).to(torch.float64)
    return FrozenNCA2Design(
        continuous_design=projected[0, 0].cpu().numpy(),
        binary_design=binary[0, 0].cpu().numpy(),
        checkpoint_model_hash=checkpoint_hash,
    )


def evaluate_qualification_checkpoints(
    *,
    protocol_id: str,
    seed: int,
    run_dir: Path,
    sources: object,
    finalizer: Callable[..., FrozenNCA2Design] = freeze_nca2_checkpoint,
    verifier: Callable[..., Any] = verify_candidate,
) -> tuple[QualificationCheckpointDiagnostic, ...]:
    """Evaluate all five registered checkpoints with independent SciPy physics."""
    expected_names = [
        f"checkpoint_{completed:06d}.pt" for completed in range(50, 701, 50)
    ]
    actual_names = sorted(path.name for path in run_dir.glob("checkpoint_*.pt"))
    if actual_names != expected_names:
        raise RuntimeError("qualification checkpoint registry is incomplete")
    rows: list[QualificationCheckpointDiagnostic] = []
    for completed_updates in QUALIFICATION_CHECKPOINTS:
        frozen = finalizer(
            checkpoint_path=run_dir / f"checkpoint_{completed_updates:06d}.pt",
            sources=sources,
            completed_updates=completed_updates,
            protocol_id=protocol_id,
        )
        candidate_id = f"{protocol_id}_{seed}_{completed_updates}"
        verification = verifier(
            candidate_id,
            frozen.binary_design,
            fidelity="low_64",
        )
        connectivity = connectivity_diagnostic(frozen.binary_design)
        rows.append(
            QualificationCheckpointDiagnostic(
                protocol_id=protocol_id,
                seed=seed,
                completed_updates=completed_updates,
                worst_peak=float(verification.worst_peak),
                binary_fraction=float(frozen.binary_design.mean()),
                connectivity=connectivity,
            )
        )
    return tuple(rows)


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


def _validate_qualification_gate(output_dir: Path) -> dict[str, Any]:
    benchmark_path = output_dir / "revised_loop_benchmark.json"
    manifest_path = output_dir / "protocol_manifest.json"
    if not benchmark_path.is_file() or not manifest_path.is_file():
        raise NCA2GateError("runtime benchmark gate artifacts are missing")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        benchmark.get("status") != "PASS"
        or benchmark.get("qualification_authorized") is not True
        or manifest.get("qualification_authorized") is not True
    ):
        raise NCA2GateError("runtime gate did not authorize qualification")
    expected = {
        "config_sha256": artifact_sha256(CONFIG_PATH),
        "spec_sha256": artifact_sha256(SPEC_PATH),
        "old_verdict_sha256": artifact_sha256(OLD_VERDICT_PATH),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise NCA2GateError("runtime gate provenance hash mismatch")
    return manifest


def _engineering_connectivity_pass(
    diagnostic: QualificationCheckpointDiagnostic,
) -> bool:
    intersections = diagnostic.connectivity.sink_component_source_intersections
    return all(intersections.get(scenario_id, False) for scenario_id in ("A", "B", "C"))


def run_qualification_phase(output_dir: Path) -> NCA2QualificationVerdict:
    """Run the locked six-run development comparison and freeze A/B selection."""
    manifest = _validate_qualification_gate(output_dir)
    if not torch.cuda.is_available():
        raise NCA2GateError("CUDA is unavailable; qualification cannot use CPU")
    sources = gate2_source_batch(device=torch.device("cuda"))
    runs = execute_qualification_runs(output_dir=output_dir, sources=sources)
    diagnostic_rows: list[QualificationCheckpointDiagnostic] = []
    seed_metrics: list[DevelopmentSeedMetrics] = []
    run_registry: list[dict[str, Any]] = []
    run_by_key = {(run.protocol_id, run.seed): run for run in runs}
    for run in runs:
        result = run.result
        numerically_valid = bool(
            result.status is NCARunStatus.PASS
            and result.completed_iterations == 700
            and [record.iteration for record in result.records] == list(range(700))
        )
        diagnostics: tuple[QualificationCheckpointDiagnostic, ...] = ()
        reason_codes = list(getattr(result, "reason_codes", ()))
        if numerically_valid:
            try:
                diagnostics = evaluate_qualification_checkpoints(
                    protocol_id=run.protocol_id,
                    seed=run.seed,
                    run_dir=run.run_dir,
                    sources=sources,
                )
            except (RuntimeError, ValueError) as error:
                numerically_valid = False
                reason_codes.append(type(error).__name__)
        diagnostic_rows.extend(diagnostics)
        initial_peer = run_by_key[("B" if run.protocol_id == "A" else "A", run.seed)]
        if run.initial_model_hash != initial_peer.initial_model_hash:
            numerically_valid = False
            reason_codes.append("INITIAL_MODEL_MISMATCH")
        final_diagnostic = diagnostics[-1] if diagnostics else None
        peaks = (
            tuple(row.worst_peak for row in diagnostics)
            if diagnostics
            else (math.nan,) * 5
        )
        metric = classify_development_seed(
            protocol_id=run.protocol_id,
            seed=run.seed,
            checkpoint_peaks=peaks,
            binary_fraction=(
                final_diagnostic.binary_fraction if final_diagnostic else math.nan
            ),
            numerically_valid=numerically_valid,
            connectivity_pass=(
                _engineering_connectivity_pass(final_diagnostic)
                if final_diagnostic
                else False
            ),
        )
        seed_metrics.append(metric)
        result_path = run.run_dir / "nca_run_result.json"
        run_registry.append(
            {
                "protocol_id": run.protocol_id,
                "seed": run.seed,
                "status": result.status.value,
                "completed_iterations": result.completed_iterations,
                "initial_model_sha256": run.initial_model_hash,
                "final_model_sha256": result.final_model_hash,
                "reason_codes": reason_codes,
                "run_result_sha256": (
                    artifact_sha256(result_path) if result_path.is_file() else None
                ),
            }
        )
    protocol_a = summarize_protocol(
        "A", tuple(metric for metric in seed_metrics if metric.protocol_id == "A")
    )
    protocol_b = summarize_protocol(
        "B", tuple(metric for metric in seed_metrics if metric.protocol_id == "B")
    )
    verdict = select_nca2_protocol(protocol_a, protocol_b)
    metrics_rows = []
    for row in diagnostic_rows:
        payload = asdict(row)
        payload["connectivity"] = json.dumps(
            payload["connectivity"], sort_keys=True, separators=(",", ":")
        )
        metrics_rows.append(payload)
    metrics_path = output_dir / "qualification_metrics.csv"
    _atomic_csv(metrics_path, pd.DataFrame(metrics_rows))
    payload = _json_safe(
        {
            "schema_version": 1,
            **asdict(verdict),
            "run_registry": run_registry,
            "artifact_hashes": {
                "config_sha256": artifact_sha256(CONFIG_PATH),
                "spec_sha256": artifact_sha256(SPEC_PATH),
                "old_verdict_sha256": artifact_sha256(OLD_VERDICT_PATH),
                "protocol_manifest_sha256": artifact_sha256(
                    output_dir / "protocol_manifest.json"
                ),
                "qualification_metrics_sha256": artifact_sha256(metrics_path),
            },
            "implementation_git_sha": _git_sha(),
            "runtime_gate_implementation_git_sha": manifest["implementation_git_sha"],
        }
    )
    _write_json(output_dir / "qualification_verdict.json", payload)
    return verdict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("benchmark", "qualification"))
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
    elif args.phase == "qualification":
        run_qualification_phase(args.output)


if __name__ == "__main__":
    main()
