"""Checkpointed execution boundary for matched WaveForge MT3 training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from waveforge.ml.mt3_training import (
    MT3QualificationRun,
    MT3QualificationVerdict,
    MT3RunConfig,
    MT3RunStatus,
    MT3Variant,
    run_mt3_training,
    select_mt3_learning_rate,
)

MODEL_SEED = 2026092311
TASK_STREAM_SEED = 2026092312
UPDATES = 4000
BATCH_SIZE = 4
CHECKPOINT_INTERVAL = 500
_VARIANTS: tuple[MT3Variant, ...] = ("FIELD_UNET", "SENS_UNET")
_QUALIFICATION_LEARNING_RATES = (1.0e-4, 3.0e-4)
_QUALIFICATION_MODEL_SEEDS = (2026092303, 2026092304)
_QUALIFICATION_TASK_SEED_BASE = 2026092305


class MT3ExecutionError(RuntimeError):
    """Fail-closed provenance, ordering, or campaign error."""


@dataclass(frozen=True)
class MT3QualificationSpec:
    """One of the four prospectively registered LR/seed runs."""

    learning_rate: float
    model_seed: int
    task_stream_seed: int


@dataclass(frozen=True)
class MT3QualificationEvaluation:
    """Solver-consistent validation result for one 500-update run."""

    valid: bool
    median_best4_r25_gap: float
    p90_best4_r25_gap: float


QualificationTrainer = Callable[[MT3QualificationSpec, Path], Path]
QualificationEvaluator = Callable[
    [Path, MT3QualificationSpec], MT3QualificationEvaluation
]


def qualification_specs() -> tuple[MT3QualificationSpec, ...]:
    """Return the exact ordered two-rate/two-seed qualification matrix."""
    return tuple(
        MT3QualificationSpec(
            learning_rate=learning_rate,
            model_seed=model_seed,
            task_stream_seed=_QUALIFICATION_TASK_SEED_BASE + seed_index,
        )
        for learning_rate in _QUALIFICATION_LEARNING_RATES
        for seed_index, model_seed in enumerate(_QUALIFICATION_MODEL_SEEDS)
    )


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("qualification metrics cannot be empty")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_qualification_campaign(
    *,
    output_root: Path,
    trainer: QualificationTrainer,
    evaluator: QualificationEvaluator,
    source_sha: str,
    protocol_bundle_sha: str,
) -> MT3QualificationVerdict:
    """Run exactly four registered qualification jobs and freeze their verdict."""
    if not _is_sha(source_sha):
        raise ValueError("source_sha must be a lowercase 40-character Git SHA")
    if len(protocol_bundle_sha) != 64 or any(
        character not in "0123456789abcdef" for character in protocol_bundle_sha
    ):
        raise ValueError("protocol bundle hash must be lowercase SHA-256")
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[MT3QualificationRun] = []
    for spec in qualification_specs():
        label = f"lr_{spec.learning_rate:.0e}_seed_{spec.model_seed}"
        checkpoint = trainer(spec, output_root / label)
        if not checkpoint.is_file():
            raise MT3ExecutionError("qualification trainer returned no checkpoint")
        evaluation = evaluator(checkpoint, spec)
        rows.append(
            MT3QualificationRun(
                learning_rate=spec.learning_rate,
                model_seed=spec.model_seed,
                valid=evaluation.valid,
                median_best4_r25_gap=evaluation.median_best4_r25_gap,
                p90_best4_r25_gap=evaluation.p90_best4_r25_gap,
            )
        )
    verdict = select_mt3_learning_rate(tuple(rows))
    metric_rows = [asdict(row) for row in rows]
    _atomic_csv(output_root / "qualification_metrics.csv", metric_rows)
    _atomic_json(
        output_root / "qualification_verdict.json",
        {
            "schema_version": 1,
            "production_authorized": verdict.production_authorized,
            "selected_learning_rate": verdict.selected_learning_rate,
            "exact_reason": verdict.reason,
            "rows": metric_rows,
            "execution_source_sha": source_sha,
            "protocol_bundle_sha256": protocol_bundle_sha,
            "validation_accessed": True,
            "test_id_accessed": False,
            "test_ood_accessed": False,
        },
    )
    return verdict


def _is_sha(value: str) -> bool:
    return len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def build_variant_identity(
    variant: MT3Variant,
    *,
    source_sha: str,
    selected_learning_rate: float,
) -> dict[str, object]:
    """Return every immutable field needed to reject an unsafe resume."""
    if variant not in _VARIANTS:
        raise ValueError("unsupported MT3 variant")
    if not _is_sha(source_sha):
        raise ValueError("source_sha must be a lowercase 40-character Git SHA")
    if selected_learning_rate not in (1.0e-4, 3.0e-4):
        raise ValueError("selected MT3 learning rate is not registered")
    return {
        "schema_version": 1,
        "experiment": "MT3_SENSITIVITY_LEARNED_WARMSTART",
        "variant": variant,
        "execution_source_sha": source_sha,
        "selected_learning_rate": selected_learning_rate,
        "model_seed": MODEL_SEED,
        "task_stream_seed": TASK_STREAM_SEED,
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "candidate_count": 4,
        "selected_candidates_for_refinement": 1,
        "validation_accessed": False,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }


def validate_variant_identity(path: Path, *, expected: dict[str, object]) -> None:
    """Reject a resume if any registered scientific identity field differs."""
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MT3ExecutionError("run identity is missing or unreadable") from error
    if observed != expected:
        raise MT3ExecutionError("run identity does not match the authorized campaign")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def protocol_bundle_sha256(root: Path) -> str:
    """Hash the locked YAML and approved prospective design as raw bytes."""
    paths = (
        root / "configs" / "mt3_sensitivity_warmstart.yaml",
        root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-09-01-mt3-sensitivity-learned-warmstart-design.md",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _current_source_sha(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _require_field_before_sens(output_root: Path) -> None:
    path = output_root / "field_unet" / "mt3_run_result.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MT3ExecutionError("FIELD_UNET must finish before SENS_UNET") from error
    if payload.get("status") != "PASS" or payload.get("completed_updates") != UPDATES:
        raise MT3ExecutionError("FIELD_UNET must finish before SENS_UNET")


def run_variant(
    variant: MT3Variant,
    *,
    root: Path,
    output_root: Path,
    source_sha: str,
    selected_learning_rate: float,
    chunk_updates: int,
) -> None:
    """Run or resume one locked production variant in bounded chunks."""
    if _current_source_sha(root) != source_sha:
        raise MT3ExecutionError("working tree HEAD differs from execution source SHA")
    if variant == "SENS_UNET":
        _require_field_before_sens(output_root)
    variant_dir = output_root / variant.lower()
    variant_dir.mkdir(parents=True, exist_ok=True)
    identity = build_variant_identity(
        variant,
        source_sha=source_sha,
        selected_learning_rate=selected_learning_rate,
    )
    identity["protocol_bundle_sha256"] = protocol_bundle_sha256(root)
    identity_path = variant_dir / "run_identity.json"
    if identity_path.exists():
        validate_variant_identity(identity_path, expected=identity)
    else:
        if any(variant_dir.iterdir()):
            raise MT3ExecutionError("nonempty MT3 directory lacks run identity")
        _atomic_json(identity_path, identity)
    config = MT3RunConfig(
        variant=variant,
        model_seed=MODEL_SEED,
        task_seed=TASK_STREAM_SEED,
        base_learning_rate=selected_learning_rate,
        total_updates=UPDATES,
        batch_size=BATCH_SIZE,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        mode="production",
        device="cuda",
    )
    while True:
        checkpoint = _latest_checkpoint(variant_dir)
        completed = 0 if checkpoint is None else int(checkpoint.stem.rsplit("_", 1)[1])
        if completed >= UPDATES:
            break
        result = run_mt3_training(
            config=config,
            output_dir=variant_dir,
            resume_checkpoint=checkpoint,
            maximum_updates_this_call=min(chunk_updates, UPDATES - completed),
            synchronize=torch.cuda.synchronize,
        )
        if result.status is MT3RunStatus.INVALID_RUN:
            raise MT3ExecutionError(
                f"{variant} became invalid: {','.join(result.reason_codes)}"
            )
        if result.completed_updates <= completed:
            raise MT3ExecutionError(f"{variant} made no checkpointed progress")
        print(
            f"MT3_PROGRESS variant={variant} "
            f"updates={result.completed_updates}/{UPDATES}",
            flush=True,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--chunk-updates", type=int, default=500)
    parser.add_argument("--variant", choices=_VARIANTS, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if not 1 <= arguments.chunk_updates <= CHECKPOINT_INTERVAL:
        raise ValueError("chunk updates must lie in [1,500]")
    run_variant(
        arguments.variant,
        root=arguments.root.resolve(),
        output_root=arguments.output.resolve(),
        source_sha=arguments.source_sha,
        selected_learning_rate=arguments.learning_rate,
        chunk_updates=arguments.chunk_updates,
    )


if __name__ == "__main__":
    main()
