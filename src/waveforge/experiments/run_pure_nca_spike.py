"""Phase-gated execution of the preregistered pure-NCA feasibility spike."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from waveforge.environment import collect_environment
from waveforge.experiments.run_inverse_design import gate2_source_batch
from waveforge.ml.nca import PureNCA, build_static_condition, project_nca_material
from waveforge.ml.nca_protocol import load_nca_protocol
from waveforge.ml.nca_qualification import (
    QualificationReason,
    QualificationVerdict,
    evaluate_lr_eligibility,
    select_learning_rate,
)
from waveforge.ml.nca_training import (
    NCARunResult,
    NCARunStatus,
    run_nca_training,
)
from waveforge.reproducibility import (
    artifact_sha256,
    configure_cuda_reproducibility,
    content_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "pure_nca_spike.yaml"
SPEC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "superpowers"
    / "specs"
    / "2026-08-29-pure-nca-physics-trained-spike-design.md"
)
PREFLIGHT_FILES = (
    "environment.json",
    "initial_state_sanity.json",
    "determinism_preflight.json",
    "preflight_report.json",
    "complete_step_benchmark.json",
    "protocol_manifest.json",
)


class PreflightGateError(RuntimeError):
    """A later experiment phase attempted to bypass blocking preflight."""


class QualificationGateError(RuntimeError):
    """Production attempted to bypass the locked LR selection."""


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
    temporary.write_text(
        frame.to_csv(index=False),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def validate_preflight_gate(output_dir: Path) -> dict[str, Any]:
    """Require every PASS artifact and exact current protocol hashes."""
    missing = [name for name in PREFLIGHT_FILES if not (output_dir / name).is_file()]
    if missing:
        raise PreflightGateError(f"preflight artifacts missing: {missing}")
    for name in PREFLIGHT_FILES:
        payload = _read_json(output_dir / name)
        if payload.get("status") != "PASS":
            raise PreflightGateError(f"preflight artifact is not PASS: {name}")
    manifest = _read_json(output_dir / "protocol_manifest.json")
    if manifest.get("config_sha256") != artifact_sha256(CONFIG_PATH):
        raise PreflightGateError("preflight config hash does not match locked config")
    if manifest.get("spec_sha256") != artifact_sha256(SPEC_PATH):
        raise PreflightGateError(
            "preflight spec hash does not match locked specification"
        )
    return manifest


def run_qualification_phase(
    output_dir: Path,
    *,
    training_runner: Callable[..., NCARunResult] = run_nca_training,
) -> QualificationVerdict:
    """Run all three registered candidates and lock the deterministic winner."""
    manifest = validate_preflight_gate(output_dir)
    protocol = load_nca_protocol(CONFIG_PATH)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; qualification cannot use CPU")
    configure_cuda_reproducibility(
        protocol.qualification.seed,
        warn_only=manifest.get("determinism_mode") == "topology_verdict",
    )
    sources = gate2_source_batch(device=torch.device("cuda"))
    run_results: list[NCARunResult] = []
    qualifications = []
    metric_rows: list[dict[str, Any]] = []
    run_hashes: dict[str, str] = {}
    for learning_rate in protocol.qualification.candidate_learning_rates:
        label = f"lr_{learning_rate:.0e}".replace("e-0", "e-")
        run_dir = output_dir / "qualification" / label
        result = training_runner(
            sources=sources,
            seed=protocol.qualification.seed,
            learning_rate=learning_rate,
            iterations=protocol.qualification.iterations,
            mode="qualification",
            output_dir=run_dir,
            checkpoint_interval=100,
            synchronize=torch.cuda.synchronize,
        )
        run_results.append(result)
        qualifications.append(
            evaluate_lr_eligibility(
                learning_rate,
                result.initial_objective,
                result.records,
            )
        )
        for record in result.records:
            metric_rows.append({"learning_rate": learning_rate, **asdict(record)})
        result_path = run_dir / "nca_run_result.json"
        if result_path.is_file():
            run_hashes[label] = artifact_sha256(result_path)

    initial_hashes = {result.initial_model_hash for result in run_results}
    if len(initial_hashes) != 1:
        qualifications = [
            replace(
                qualification,
                eligible=False,
                reason_codes=(
                    *qualification.reason_codes,
                    QualificationReason.INITIAL_MODEL_MISMATCH,
                ),
            )
            for qualification in qualifications
        ]
    verdict = select_learning_rate(qualifications)
    metrics_path = output_dir / "lr_qualification_metrics.csv"
    _atomic_csv(metrics_path, pd.DataFrame(metric_rows))
    payload = {
        "schema_version": 1,
        **asdict(verdict),
        "initial_model_hashes": [result.initial_model_hash for result in run_results],
        "artifact_hashes": {
            "config_sha256": artifact_sha256(CONFIG_PATH),
            "spec_sha256": artifact_sha256(SPEC_PATH),
            "preflight_manifest_sha256": artifact_sha256(
                output_dir / "protocol_manifest.json"
            ),
            "metrics_sha256": artifact_sha256(metrics_path),
            "run_result_sha256": run_hashes,
        },
    }
    _write_json(output_dir / "lr_qualification_verdict.json", payload)
    return verdict


def run_production_phase(output_dir: Path, *, seed: int) -> None:
    """Gate placeholder that rejects absent or mismatched selected LR artifacts."""
    validate_preflight_gate(output_dir)
    verdict_path = output_dir / "lr_qualification_verdict.json"
    if not verdict_path.is_file():
        raise QualificationGateError("selected learning rate artifact is missing")
    verdict = _read_json(verdict_path)
    if verdict.get("selected_learning_rate") is None:
        raise QualificationGateError("selected learning rate is absent")
    if seed not in (20260901, 20260902, 20260903):
        raise QualificationGateError(f"unregistered production seed: {seed}")
    raise NotImplementedError("production runner is implemented in Task 7")


def benchmark_complete_steps(
    *,
    sources: Tensor | object,
    training_runner: Callable[..., NCARunResult] = run_nca_training,
    synchronizer: Callable[[], None] = torch.cuda.synchronize,
    reset_peak_memory: Callable[[], None] = torch.cuda.reset_peak_memory_stats,
    peak_allocated_memory: Callable[[], int] = torch.cuda.max_memory_allocated,
    peak_reserved_memory: Callable[[], int] = torch.cuda.max_memory_reserved,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure ten complete updates after three untimed warmup updates."""
    warmup_runs = 3
    measured_runs = 10

    def iteration_start_hook(iteration: int) -> None:
        if iteration == warmup_runs:
            reset_peak_memory()

    result = training_runner(
        sources=sources,
        seed=20260830,
        learning_rate=1.0e-3,
        iterations=warmup_runs + measured_runs,
        mode="benchmark",
        output_dir=None,
        checkpoint_interval=100,
        synchronize=synchronizer,
        clock=timer,
        iteration_start_hook=iteration_start_hook,
    )
    expected = warmup_runs + measured_runs
    if (
        result.status is not NCARunStatus.PASS
        or result.completed_iterations != expected
    ):
        raise RuntimeError(
            f"complete-step benchmark requires exactly {expected} PASS records"
        )
    samples = np.asarray(
        [record.wall_seconds for record in result.records[warmup_runs:]],
        dtype=np.float64,
    )
    if samples.shape != (measured_runs,) or not np.isfinite(samples).all():
        raise RuntimeError(
            "complete-step benchmark samples are incomplete or non-finite"
        )
    median = float(np.median(samples))
    return {
        "schema_version": 1,
        "status": "PASS",
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "samples_seconds": samples.tolist(),
        "median_seconds": median,
        "p90_seconds": float(np.percentile(samples, 90)),
        "mean_seconds": float(np.mean(samples)),
        "standard_deviation_seconds": float(np.std(samples)),
        "peak_allocated_bytes": int(peak_allocated_memory()),
        "peak_reserved_bytes": int(peak_reserved_memory()),
        "projected_qualification_updates": 600,
        "projected_qualification_seconds": median * 600,
        "projected_production_updates": 6000,
        "projected_production_seconds": median * 6000,
    }


def _hash_state_tree(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(
                item, key=lambda candidate: (type(candidate).__name__, str(candidate))
            ):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                update(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))
        else:
            raise TypeError(f"unsupported optimizer-state value: {type(item)!r}")

    update(value)
    return digest.hexdigest()


def _checkpoint_optimizer_hash(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return _hash_state_tree(payload["optimizer_state"])


def _torch_install_command() -> str:
    gate1_environment = (
        PROJECT_ROOT / "artifacts" / "gate1_physics" / "environment.json"
    )
    return str(_read_json(gate1_environment)["torch_install_command"])


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _initial_state_sanity(device: torch.device) -> dict[str, Any]:
    model = PureNCA().to(device=device, dtype=torch.float32)
    sources = gate2_source_batch(device=device)
    condition = build_static_condition(sources)
    rollout = model.rollout(condition)
    projected = project_nca_material(rollout.material_logit)

    coordinates = torch.linspace(-1.0, 1.0, 64, device=device, dtype=torch.float32)
    perturbation = (coordinates[:, None] + coordinates[None, :])[None, None].clone()
    perturbation.requires_grad_(True)
    weights = torch.linspace(0.2, 1.0, 4096, device=device).reshape(64, 64)
    perturbed_design = project_nca_material(perturbation).design
    torch.sum(perturbed_design * weights).backward()
    gradient = perturbation.grad
    if gradient is None:
        raise RuntimeError("projection derivative is missing")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "initial_state_nonzero_count": int(
            torch.count_nonzero(rollout.final_state).item()
        ),
        "initial_material_logit_nonzero_count": int(
            torch.count_nonzero(rollout.material_logit).item()
        ),
        "projected_material_fraction": float(projected.design.mean().item()),
        "projection_absolute_error": projected.projection.absolute_error,
        "projection_gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
        "projection_gradient_finite": bool(torch.isfinite(gradient).all().item()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if (
        payload["initial_state_nonzero_count"] != 0
        or payload["initial_material_logit_nonzero_count"] != 0
        or abs(payload["projected_material_fraction"] - 0.25) > 1.0e-6
        or not payload["projection_gradient_finite"]
        or payload["projection_gradient_norm"] <= 0.0
        or payload["parameter_count"] != 11472
    ):
        payload["status"] = "INVALID_RUN"
        raise RuntimeError(f"initial-state sanity failed: {payload}")
    return payload


def _deterministic_two_step_replay(
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    sources = gate2_source_batch(device=device)
    runs: list[NCARunResult] = []
    optimizer_hashes: list[str] = []
    for index in (1, 2):
        run_dir = output_dir / "determinism" / f"run_{index}"
        result = run_nca_training(
            sources,
            seed=20260831,
            learning_rate=1.0e-3,
            iterations=2,
            mode="smoke",
            output_dir=run_dir,
            checkpoint_interval=2,
            synchronize=torch.cuda.synchronize,
        )
        if result.status is not NCARunStatus.PASS:
            raise RuntimeError(f"deterministic two-step run {index} failed")
        runs.append(result)
        optimizer_hashes.append(
            _checkpoint_optimizer_hash(run_dir / "checkpoint_000002.pt")
        )

    continuous_hashes = [
        content_hash(result.final_continuous_design.numpy()) for result in runs
    ]
    binary_hashes = [
        content_hash(result.final_binary_design.numpy()) for result in runs
    ]
    gradient_path_valid = all(
        result.records[0].conv1x1_weight_gradient_norm > 1.0e-12
        and result.records[0].conv3x3_weight_gradient_norm == 0.0
        and result.records[1].conv3x3_weight_gradient_norm > 1.0e-12
        for result in runs
    )
    exact = (
        len({result.final_model_hash for result in runs}) == 1
        and len(set(continuous_hashes)) == 1
        and len(set(binary_hashes)) == 1
        and len(set(optimizer_hashes)) == 1
    )
    payload = {
        "schema_version": 1,
        "status": "PASS" if exact and gradient_path_valid else "INVALID_RUN",
        "model_hashes": [result.final_model_hash for result in runs],
        "continuous_design_hashes": continuous_hashes,
        "binary_design_hashes": binary_hashes,
        "optimizer_state_hashes": optimizer_hashes,
        "gradient_path_valid": gradient_path_valid,
        "exact_replay": exact,
    }
    if payload["status"] != "PASS":
        raise RuntimeError(f"strict deterministic replay failed: {payload}")
    return payload


def run_preflight_phase(
    output_dir: Path,
    *,
    warn_only_determinism: bool = False,
) -> dict[str, Any]:
    """Execute all blocking CUDA sanity, replay, smoke and benchmark checks."""
    protocol = load_nca_protocol(CONFIG_PATH)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pure-NCA preflight cannot use CPU")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    determinism = configure_cuda_reproducibility(
        20260831,
        warn_only=warn_only_determinism,
    )
    environment = collect_environment(
        _torch_install_command(),
        determinism=determinism,
    )
    environment["status"] = "PASS"
    _write_json(output_dir / "environment.json", environment)

    sanity = _initial_state_sanity(device)
    _write_json(output_dir / "initial_state_sanity.json", sanity)

    replay = _deterministic_two_step_replay(output_dir, device)
    replay["determinism_mode"] = determinism.mode
    _write_json(output_dir / "determinism_preflight.json", replay)

    sources = gate2_source_batch(device=device)
    smoke = run_nca_training(
        sources,
        seed=20260830,
        learning_rate=1.0e-3,
        iterations=10,
        mode="smoke",
        output_dir=output_dir / "smoke",
        checkpoint_interval=10,
        synchronize=torch.cuda.synchronize,
    )
    if smoke.status is not NCARunStatus.PASS or smoke.completed_iterations != 10:
        raise RuntimeError(f"10-step CUDA smoke failed: {smoke.reason_codes}")

    benchmark = benchmark_complete_steps(sources=sources)
    _write_json(output_dir / "complete_step_benchmark.json", benchmark)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "scope": protocol.scope,
        "config_sha256": artifact_sha256(CONFIG_PATH),
        "spec_sha256": artifact_sha256(SPEC_PATH),
        "implementation_git_sha": _git_sha(),
        "determinism_mode": determinism.mode,
        "preflight_seed": 20260831,
        "smoke_seed": 20260830,
        "smoke_learning_rate": 1.0e-3,
    }
    _write_json(output_dir / "protocol_manifest.json", manifest)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "determinism_mode": determinism.mode,
        "initial_state_sanity": sanity["status"],
        "two_step_replay": replay["status"],
        "smoke_status": smoke.status.value,
        "smoke_completed_iterations": smoke.completed_iterations,
        "benchmark_status": benchmark["status"],
        "projected_qualification_seconds": benchmark["projected_qualification_seconds"],
        "projected_production_seconds": benchmark["projected_production_seconds"],
    }
    _write_json(output_dir / "preflight_report.json", report)
    validate_preflight_gate(output_dir)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("preflight", "qualification", "production", "verification", "report"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--warn-only-determinism", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase == "preflight":
        run_preflight_phase(
            args.output,
            warn_only_determinism=args.warn_only_determinism,
        )
    elif args.phase == "qualification":
        run_qualification_phase(args.output)
    elif args.phase == "production":
        if args.seed is None:
            raise ValueError("--seed is required for production")
        run_production_phase(args.output, seed=args.seed)
    else:
        raise NotImplementedError(f"phase {args.phase} is not implemented yet")


if __name__ == "__main__":
    main()
