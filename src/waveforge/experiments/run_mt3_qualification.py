"""Run the bounded two-rate/two-seed WaveForge MT3 qualification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from waveforge.experiments.run_mt2b_evaluation import (
    _load_references,
    validation_tasks,
)
from waveforge.experiments.run_mt3_training import (
    MT3ExecutionError,
    MT3QualificationEvaluation,
    MT3QualificationSpec,
    protocol_bundle_sha256,
    run_qualification_campaign,
)
from waveforge.ml.mt2b_evaluation import independent_scipy64_tmax
from waveforge.ml.mt3_conditioning import build_mt3_conditioning, compute_initial_probe
from waveforge.ml.mt3_protocol import assert_paid_runtime_authorized
from waveforge.ml.mt3_refinement import select_and_refine
from waveforge.ml.mt3_training import (
    MT3RunConfig,
    MT3RunStatus,
    initialize_mt3_model,
    mt3_model_state_sha256,
    run_mt3_training,
)
from waveforge.ml.mt3_unet import project_mt3_candidates
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.reproducibility import artifact_sha256


def build_qualification_config(spec: MT3QualificationSpec) -> MT3RunConfig:
    """Build the exact registered 500-update SENS qualification run."""
    return MT3RunConfig(
        variant="SENS_UNET",
        model_seed=spec.model_seed,
        task_seed=spec.task_stream_seed,
        base_learning_rate=spec.learning_rate,
        total_updates=500,
        batch_size=4,
        checkpoint_interval=500,
        mode="qualification",
        device="cuda",
    )


def qualification_gap_summary(
    *,
    candidate_tmax: list[float],
    reference_tmax: list[float],
) -> MT3QualificationEvaluation:
    """Summarize paired solver-consistent gaps without rounded values."""
    if len(candidate_tmax) != len(reference_tmax) or not candidate_tmax:
        raise ValueError("qualification requires equal nonempty paired arrays")
    candidate = np.asarray(candidate_tmax, dtype=np.float64)
    reference = np.asarray(reference_tmax, dtype=np.float64)
    if (
        not np.isfinite(candidate).all()
        or not np.isfinite(reference).all()
        or np.any(candidate <= 0.0)
        or np.any(reference <= 0.0)
    ):
        return MT3QualificationEvaluation(False, math.inf, math.inf)
    gaps = (candidate - reference) / reference
    return MT3QualificationEvaluation(
        True,
        float(np.median(gaps)),
        float(np.quantile(gaps, 0.9)),
    )


def assert_qualification_budget(
    *,
    projected_hours: float,
    hourly_usd: float,
    credit_usd: float,
) -> None:
    """Apply the locked paid-runtime guard with its keyword-only credit input."""
    assert_paid_runtime_authorized(
        projected_hours,
        hourly_usd,
        credit_usd=credit_usd,
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("qualification validation rows cannot be empty")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _checkpoint_completed_updates(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise MT3ExecutionError("malformed qualification checkpoint name") from error


def _latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = sorted(
        directory.glob("checkpoint_*.pt"), key=_checkpoint_completed_updates
    )
    return checkpoints[-1] if checkpoints else None


def _qualification_identity(
    spec: MT3QualificationSpec,
    *,
    source_sha: str,
    protocol_sha: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": "MT3_SENSITIVITY_LEARNED_WARMSTART_QUALIFICATION",
        "variant": "SENS_UNET",
        "learning_rate": spec.learning_rate,
        "model_seed": spec.model_seed,
        "task_stream_seed": spec.task_stream_seed,
        "updates": 500,
        "batch_size": 4,
        "execution_source_sha": source_sha,
        "protocol_bundle_sha256": protocol_sha,
        "validation_accessed": True,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }


def train_qualification_to_checkpoint(
    spec: MT3QualificationSpec,
    directory: Path,
    *,
    source_sha: str,
    protocol_sha: str,
    chunk_updates: int,
) -> Path:
    """Run or safely resume one registered 500-update qualification job."""
    if chunk_updates < 1 or chunk_updates > 500:
        raise ValueError("qualification chunks must lie in [1,500]")
    directory.mkdir(parents=True, exist_ok=True)
    identity = _qualification_identity(
        spec,
        source_sha=source_sha,
        protocol_sha=protocol_sha,
    )
    identity_path = directory / "qualification_identity.json"
    if identity_path.is_file():
        observed = json.loads(identity_path.read_text(encoding="utf-8"))
        if observed != identity:
            raise MT3ExecutionError("qualification identity mismatch")
    else:
        if any(directory.iterdir()):
            raise MT3ExecutionError("nonempty qualification directory lacks identity")
        _atomic_json(identity_path, identity)

    config = build_qualification_config(spec)
    while True:
        checkpoint = _latest_checkpoint(directory)
        completed = (
            0 if checkpoint is None else _checkpoint_completed_updates(checkpoint)
        )
        if completed >= 500:
            break
        result = run_mt3_training(
            config=config,
            output_dir=directory,
            resume_checkpoint=checkpoint,
            maximum_updates_this_call=min(chunk_updates, 500 - completed),
            synchronize=torch.cuda.synchronize,
        )
        if result.status is MT3RunStatus.INVALID_RUN:
            raise MT3ExecutionError(
                "qualification training became invalid: "
                + ",".join(result.reason_codes)
            )
        if result.completed_updates <= completed or result.last_checkpoint is None:
            raise MT3ExecutionError("qualification made no checkpointed progress")
        print(
            "MT3_QUALIFICATION_PROGRESS "
            f"lr={spec.learning_rate:.0e} seed={spec.model_seed} "
            f"updates={result.completed_updates}/500",
            flush=True,
        )
    if checkpoint is None:
        raise MT3ExecutionError("qualification ended without a checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("config") != asdict(config)
        or payload.get("completed_updates") != 500
    ):
        raise MT3ExecutionError("qualification checkpoint violates the locked config")
    return checkpoint


def _load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = MT3RunConfig(**payload["config"])
    if (
        config.variant != "SENS_UNET"
        or config.mode != "qualification"
        or payload.get("completed_updates") != 500
    ):
        raise MT3ExecutionError("qualification evaluator received a wrong checkpoint")
    model = initialize_mt3_model(config.model_seed, device)
    model.load_state_dict(payload["model_state"])
    if mt3_model_state_sha256(model) != payload.get("model_state_sha256"):
        raise MT3ExecutionError("qualification model hash mismatch")
    return model.eval()


def _generate_candidates(
    model: torch.nn.Module,
    task: SourceLayoutTask,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sources = torch.from_numpy(task.sources).to(device=device, dtype=torch.float64)
    probe = compute_initial_probe(
        sources.unsqueeze(0),
        allow_cpu_unit_test=False,
    )
    condition = build_mt3_conditioning(
        probe,
        sources.unsqueeze(0),
        variant="SENS_UNET",
    )
    with torch.no_grad():
        logits = model(condition)
        candidates = project_mt3_candidates(logits, beta=8.0)
    return logits[0], candidates.binary[0], sources


def evaluate_qualification_checkpoint(
    checkpoint: Path,
    spec: MT3QualificationSpec,
    *,
    tasks: tuple[SourceLayoutTask, ...],
    reference_peaks: dict[str, float],
    device: torch.device,
) -> MT3QualificationEvaluation:
    """Evaluate one trained run using SciPy64 for both candidate and reference."""
    if len(tasks) != 32 or set(reference_peaks) != {task.task_id for task in tasks}:
        raise ValueError("qualification evaluation requires 32 paired validation tasks")
    model = _load_model(checkpoint, device)
    metrics_path = checkpoint.parent / "qualification_validation_metrics.csv"
    stored = _read_csv(metrics_path)
    if any(int(row["task_index"]) != index for index, row in enumerate(stored)):
        raise MT3ExecutionError("qualification validation resume rows are corrupted")
    rows: list[dict[str, object]] = [dict(row) for row in stored]
    for index, task in enumerate(tasks[len(rows) :], start=len(rows)):
        logits, binary, sources = _generate_candidates(model, task, device)
        refined = select_and_refine(
            logits,
            binary,
            task,
            sources,
            scorer=independent_scipy64_tmax,
            steps=25,
        )
        candidate = independent_scipy64_tmax(
            refined.binary_design.numpy().astype(np.float64, copy=False),
            task,
        )
        reference = float(reference_peaks[task.task_id])
        row: dict[str, object] = {
            "task_index": index,
            "task_id": task.task_id,
            "candidate_tmax_scipy64": candidate,
            "reference_tmax_scipy64": reference,
            "relative_gap": (candidate - reference) / reference,
            "selected_head": refined.selected_head,
            "best4_tmax_scipy64": min(
                score.binary_tmax for score in refined.candidate_scores
            ),
            "binary_cell_count": int(refined.binary_design.sum().item()),
            "refinement_updates": refined.total_refinement_updates,
            "test_id_accessed": False,
            "test_ood_accessed": False,
        }
        rows.append(row)
        _atomic_csv(metrics_path, rows)
        print(
            "MT3_QUALIFICATION_VALIDATION "
            f"lr={spec.learning_rate:.0e} seed={spec.model_seed} tasks={index + 1}/32",
            flush=True,
        )
    if any(
        int(row["binary_cell_count"]) != 1024
        or int(row["refinement_updates"]) != 25
        or row["task_id"] != task.task_id
        for row, task in zip(rows, tasks, strict=True)
    ):
        return MT3QualificationEvaluation(False, math.inf, math.inf)
    result = qualification_gap_summary(
        candidate_tmax=[float(row["candidate_tmax_scipy64"]) for row in rows],
        reference_tmax=[float(row["reference_tmax_scipy64"]) for row in rows],
    )
    _atomic_json(
        checkpoint.parent / "qualification_evaluation.json",
        {
            "schema_version": 1,
            "valid": result.valid,
            "median_best4_r25_gap": result.median_best4_r25_gap,
            "p90_best4_r25_gap": result.p90_best4_r25_gap,
            "metrics_sha256": artifact_sha256(metrics_path),
            "candidate_solver": "independent_scipy_64",
            "reference_solver": "independent_scipy_64",
            "validation_accessed": True,
            "test_id_accessed": False,
            "test_ood_accessed": False,
        },
    )
    return result


def _current_source_sha(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_locked_qualification(
    *,
    root: Path,
    output_root: Path,
    reference_root: Path,
    source_sha: str,
    chunk_updates: int,
    projected_hours: float,
    hourly_usd: float,
    credit_usd: float,
):
    """Enforce provenance/budget gates and execute only MT3 qualification."""
    if _current_source_sha(root) != source_sha:
        raise MT3ExecutionError("working tree HEAD differs from execution source SHA")
    assert_qualification_budget(
        projected_hours=projected_hours,
        hourly_usd=hourly_usd,
        credit_usd=credit_usd,
    )
    protocol_sha = protocol_bundle_sha256(root)
    tasks = validation_tasks()
    _, reference_peaks = _load_references(reference_root, tasks)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("MT3 qualification requires CUDA")

    def trainer(spec: MT3QualificationSpec, directory: Path) -> Path:
        return train_qualification_to_checkpoint(
            spec,
            directory,
            source_sha=source_sha,
            protocol_sha=protocol_sha,
            chunk_updates=chunk_updates,
        )

    def evaluator(
        checkpoint: Path, spec: MT3QualificationSpec
    ) -> MT3QualificationEvaluation:
        return evaluate_qualification_checkpoint(
            checkpoint,
            spec,
            tasks=tasks,
            reference_peaks=reference_peaks,
            device=device,
        )

    return run_qualification_campaign(
        output_root=output_root,
        trainer=trainer,
        evaluator=evaluator,
        source_sha=source_sha,
        protocol_bundle_sha=protocol_sha,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--chunk-updates", type=int, default=100)
    parser.add_argument("--projected-hours", type=float, required=True)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--credit-usd", type=float, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    verdict = run_locked_qualification(
        root=arguments.root.resolve(),
        output_root=arguments.output.resolve(),
        reference_root=arguments.reference_root.resolve(),
        source_sha=arguments.source_sha,
        chunk_updates=arguments.chunk_updates,
        projected_hours=arguments.projected_hours,
        hourly_usd=arguments.hourly_usd,
        credit_usd=arguments.credit_usd,
    )
    print(
        "MT3_QUALIFICATION_AUTHORIZED"
        if verdict.production_authorized
        else "MT3_QUALIFICATION_NO_GO",
        flush=True,
    )


if __name__ == "__main__":
    main()
