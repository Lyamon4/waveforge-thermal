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
    content_hash,
)
from waveforge.verification.high_fidelity import verify_candidate
from waveforge.verification.nca2_verification import (
    PREVIOUS_WAVEFORGE_PEAKS,
    TREE_PEAK_256,
    NCA2SeedVerification,
    classify_nca2_campaign,
    verify_nca2_seed,
)
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
PRODUCTION_SEEDS = (20260911, 20260912, 20260913)
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
    snapshots: dict[int, np.ndarray]


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


def validate_production_seed(seed: int) -> int:
    """Reject any post-result replacement of the three registered seeds."""
    if seed not in PRODUCTION_SEEDS:
        raise ValueError(f"unregistered production seed: {seed}")
    return seed


def validate_production_checkpoint_registry(run_dir: Path) -> Path:
    """Require every 50-update checkpoint through the frozen update 1500."""
    expected = [f"checkpoint_{completed:06d}.pt" for completed in range(50, 1501, 50)]
    actual = sorted(path.name for path in run_dir.glob("checkpoint_*.pt"))
    if actual != expected:
        raise RuntimeError("production checkpoint registry is incomplete")
    return run_dir / "checkpoint_001500.pt"


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
    snapshot_steps: tuple[int, ...] = (),
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
        rollout = model.rollout(
            build_static_condition(sources),
            snapshot_steps=snapshot_steps,
        )
        projected = project_nca_material(
            rollout.material_logit,
            beta=settings.projection_beta,
        ).design
        binary = (projected >= 0.5).to(torch.float64)
    return FrozenNCA2Design(
        continuous_design=projected.cpu().numpy(),
        binary_design=binary.cpu().numpy(),
        checkpoint_model_hash=checkpoint_hash,
        snapshots={
            step: state[0].cpu().numpy() for step, state in rollout.snapshots.items()
        },
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


def _validate_production_gate(output_dir: Path) -> tuple[str, dict[str, Any]]:
    _validate_qualification_gate(output_dir)
    verdict_path = output_dir / "qualification_verdict.json"
    if not verdict_path.is_file():
        raise NCA2GateError("qualification verdict is missing")
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    selected = verdict.get("selected_protocol")
    if (
        verdict.get("status") != "PASS"
        or verdict.get("production_authorized") is not True
        or selected not in ("A", "B")
    ):
        raise NCA2GateError("qualification did not authorize production")
    artifact_hashes = verdict.get("artifact_hashes", {})
    expected = {
        "config_sha256": artifact_sha256(CONFIG_PATH),
        "spec_sha256": artifact_sha256(SPEC_PATH),
        "old_verdict_sha256": artifact_sha256(OLD_VERDICT_PATH),
        "protocol_manifest_sha256": artifact_sha256(
            output_dir / "protocol_manifest.json"
        ),
        "qualification_metrics_sha256": artifact_sha256(
            output_dir / "qualification_metrics.csv"
        ),
    }
    if any(artifact_hashes.get(key) != value for key, value in expected.items()):
        raise NCA2GateError("qualification provenance hash mismatch")
    return str(selected), verdict


def run_production_phase(
    output_dir: Path,
    *,
    seed: int,
    training_runner: Callable[..., NCARunResult] = run_nca_training,
    finalizer: Callable[..., FrozenNCA2Design] = freeze_nca2_checkpoint,
) -> dict[str, Any]:
    """Run and atomically freeze one untouched prospective production seed."""
    validate_production_seed(seed)
    selected_protocol, qualification = _validate_production_gate(output_dir)
    if not torch.cuda.is_available():
        raise NCA2GateError("CUDA is unavailable; production cannot use CPU")
    final_dir = output_dir / f"production_seed_{seed}"
    incomplete_dir = output_dir / f"production_seed_{seed}.incomplete"
    if final_dir.exists() or incomplete_dir.exists():
        raise NCA2GateError(f"production destination already exists for seed {seed}")
    incomplete_dir.mkdir(parents=True)
    configure_cuda_reproducibility(seed)
    sources = gate2_source_batch(device=torch.device("cuda"))
    controller = ScheduledNCAController(selected_protocol)
    result = training_runner(
        sources=sources,
        seed=seed,
        learning_rate=1.0e-3,
        iterations=1500,
        mode="production",
        output_dir=incomplete_dir,
        evaluator=controller.evaluate,
        checkpoint_interval=50,
        synchronize=torch.cuda.synchronize,
        iteration_configurator=controller.configure,
    )
    valid_records = (
        result.status is NCARunStatus.PASS
        and result.completed_iterations == 1500
        and [record.iteration for record in result.records] == list(range(1500))
    )
    if not valid_records:
        invalid = {
            "schema_version": 1,
            "status": "NCA2_INVALID_RUN",
            "seed": seed,
            "selected_protocol": selected_protocol,
            "completed_iterations": result.completed_iterations,
            "reason_codes": list(result.reason_codes),
        }
        _write_json(incomplete_dir / "production_manifest.json", invalid)
        raise NCA2GateError(f"production seed {seed} is numerically invalid")
    final_checkpoint = validate_production_checkpoint_registry(incomplete_dir)
    frozen = finalizer(
        checkpoint_path=final_checkpoint,
        sources=sources,
        completed_updates=1500,
        protocol_id=selected_protocol,
        snapshot_steps=(0, 1, 2, 4, 8, 16, 32, 48, 64),
    )
    if result.final_continuous_design is None or result.final_binary_design is None:
        raise NCA2GateError("training result did not retain final designs")
    result_continuous = result.final_continuous_design.numpy()
    result_binary = result.final_binary_design.numpy()
    if not np.array_equal(frozen.continuous_design, result_continuous):
        raise NCA2GateError("checkpoint-regenerated continuous design differs")
    if not np.array_equal(frozen.binary_design, result_binary):
        raise NCA2GateError("checkpoint-regenerated binary design differs")
    if not np.array_equal(
        frozen.binary_design,
        (frozen.continuous_design >= 0.5).astype(np.float64),
    ):
        raise NCA2GateError("final design violates strict D >= 0.5 threshold")
    if frozen.checkpoint_model_hash != result.final_model_hash:
        raise NCA2GateError("final checkpoint model hash differs from run result")
    np.save(
        incomplete_dir / "design_continuous_64.npy",
        frozen.continuous_design,
        allow_pickle=False,
    )
    np.save(
        incomplete_dir / "design_binary_64.npy",
        frozen.binary_design,
        allow_pickle=False,
    )
    np.savez(
        incomplete_dir / "rollout_snapshots.npz",
        **{f"step_{step}": state for step, state in frozen.snapshots.items()},
    )
    checkpoint_hashes = {
        path.name: artifact_sha256(path)
        for path in sorted(incomplete_dir.glob("checkpoint_*.pt"))
    }
    manifest = {
        "schema_version": 1,
        "status": "VALID_PRODUCTION_RUN",
        "seed": seed,
        "selected_protocol": selected_protocol,
        "requested_iterations": 1500,
        "completed_iterations": 1500,
        "final_iteration_index": 1499,
        "checkpoint_interval": 50,
        "initial_model_sha256": result.initial_model_hash,
        "final_model_sha256": result.final_model_hash,
        "continuous_design_sha256": content_hash(frozen.continuous_design),
        "binary_design_sha256": content_hash(frozen.binary_design),
        "binary_material_fraction": float(frozen.binary_design.mean()),
        "qualification_verdict_sha256": artifact_sha256(
            output_dir / "qualification_verdict.json"
        ),
        "qualification_selection_reason": qualification["selection_reason"],
        "config_sha256": artifact_sha256(CONFIG_PATH),
        "spec_sha256": artifact_sha256(SPEC_PATH),
        "implementation_git_sha": _git_sha(),
        "checkpoint_sha256": checkpoint_hashes,
        "optimization_metrics_sha256": artifact_sha256(
            incomplete_dir / "optimization_metrics.csv"
        ),
        "cg_records_sha256": artifact_sha256(incomplete_dir / "cg_records.csv"),
    }
    _write_json(incomplete_dir / "production_manifest.json", manifest)
    incomplete_dir.replace(final_dir)
    return manifest


def _scenario_peak_map(verification: Any) -> dict[str, float]:
    return {
        record.scenario_id: float(record.peak_temperature)
        for record in verification.scenario_records
    }


def run_verification_phase(
    output_dir: Path,
    *,
    verifier: Callable[..., NCA2SeedVerification] = verify_nca2_seed,
    gate_validator: Callable[[Path], Any] = _validate_production_gate,
) -> dict[str, Any]:
    """Independently verify all frozen seeds and publish the primary verdict."""
    gate_validator(output_dir)
    verifications: list[NCA2SeedVerification] = []
    rows_128: list[dict[str, Any]] = []
    rows_256: list[dict[str, Any]] = []
    connectivity_rows: list[dict[str, Any]] = []
    comparator_rows: list[dict[str, Any]] = []
    for seed in PRODUCTION_SEEDS:
        run_dir = output_dir / f"production_seed_{seed}"
        manifest_path = run_dir / "production_manifest.json"
        if not manifest_path.is_file():
            raise NCA2GateError(f"production manifest is missing for seed {seed}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "VALID_PRODUCTION_RUN"
            or manifest.get("seed") != seed
        ):
            raise NCA2GateError(f"production seed {seed} is not valid")
        continuous = np.load(run_dir / "design_continuous_64.npy", allow_pickle=False)
        binary = np.load(run_dir / "design_binary_64.npy", allow_pickle=False)
        result = verifier(
            seed=seed,
            binary_design=binary,
            continuous_design=continuous,
            expected_binary_content_hash=manifest["binary_design_sha256"],
            expected_continuous_content_hash=manifest["continuous_design_sha256"],
            numerically_valid=True,
        )
        verifications.append(result)
        for fidelity_result, destination in (
            (result.verification_128, rows_128),
            (result.verification_256, rows_256),
        ):
            peaks = _scenario_peak_map(fidelity_result)
            destination.append(
                {
                    "seed": seed,
                    "peak_A": peaks["A"],
                    "peak_B": peaks["B"],
                    "peak_C": peaks["C"],
                    "worst_peak": fidelity_result.worst_peak,
                    "average_peak": fidelity_result.average_peak,
                    "protected_zone_peak": fidelity_result.protected_zone_peak,
                    "binary_material_fraction": fidelity_result.material_fraction,
                    "total_wall_seconds": fidelity_result.total_wall_seconds,
                    "relative_128_to_256_change": (result.relative_128_to_256_change),
                }
            )
        connectivity = result.connectivity
        source_sink_intersections = connectivity.sink_component_source_intersections
        connectivity_rows.append(
            {
                "seed": seed,
                "engineering_connectivity_pass": (result.engineering_connectivity_pass),
                "component_count": connectivity.component_count,
                "conductive_cell_count": connectivity.conductive_cell_count,
                "sink_connected_cell_count": (connectivity.sink_connected_cell_count),
                "sink_connected_fraction": connectivity.sink_connected_fraction,
                **{
                    f"source_{scenario}_sink_connected": source_sink_intersections[
                        scenario
                    ]
                    for scenario in ("A", "B", "C")
                },
            }
        )
        comparator_rows.append(
            {
                "seed": seed,
                "comparator_id": "parametric_branching_tree",
                "comparator_tmax_256": TREE_PEAK_256,
                "nca_tmax_256": result.verdict.peak_256,
                "nca_relative_improvement": result.verdict.tree_improvement,
            }
        )
        for previous_seed, previous_peak in PREVIOUS_WAVEFORGE_PEAKS.items():
            comparator_rows.append(
                {
                    "seed": seed,
                    "comparator_id": f"waveforge_{previous_seed}",
                    "comparator_tmax_256": previous_peak,
                    "nca_tmax_256": result.verdict.peak_256,
                    "nca_relative_improvement": (
                        result.previous_waveforge_relative_differences[previous_seed]
                    ),
                }
            )
    campaign = classify_nca2_campaign(tuple(result.verdict for result in verifications))
    path_128 = output_dir / "verified_128_metrics.csv"
    path_256 = output_dir / "verified_256_metrics.csv"
    connectivity_path = output_dir / "connectivity_metrics.csv"
    comparator_path = output_dir / "comparator_metrics.csv"
    _atomic_csv(path_128, pd.DataFrame(rows_128))
    _atomic_csv(path_256, pd.DataFrame(rows_256))
    _atomic_csv(connectivity_path, pd.DataFrame(connectivity_rows))
    _atomic_csv(comparator_path, pd.DataFrame(comparator_rows))
    payload = {
        "schema_version": 1,
        "campaign": asdict(campaign),
        "seeds": [
            {
                "seed": result.seed,
                "verdict": asdict(result.verdict),
                "engineering_connectivity_pass": (result.engineering_connectivity_pass),
            }
            for result in verifications
        ],
        "artifact_hashes": {
            "verified_128_metrics_sha256": artifact_sha256(path_128),
            "verified_256_metrics_sha256": artifact_sha256(path_256),
            "connectivity_metrics_sha256": artifact_sha256(connectivity_path),
            "comparator_metrics_sha256": artifact_sha256(comparator_path),
        },
        "implementation_git_sha": _git_sha(),
    }
    _write_json(output_dir / "nca2_verdict.json", payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("benchmark", "qualification", "production", "verification"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/nca2_stabilization",
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase == "benchmark":
        run_benchmark_phase(args.output)
    elif args.phase == "qualification":
        run_qualification_phase(args.output)
    elif args.phase == "production":
        if args.seed is None:
            raise SystemExit("production phase requires --seed")
        run_production_phase(args.output, seed=args.seed)
    elif args.phase == "verification":
        run_verification_phase(args.output)


if __name__ == "__main__":
    main()
