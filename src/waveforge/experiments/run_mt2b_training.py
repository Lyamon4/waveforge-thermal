"""Run the authorized, checkpointed RAW then PHYSICS MT2B training campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from waveforge.ml.mt2b_training import (
    build_mt2b_evaluator,
    initialize_mt2b_model,
    mt2b_task_provider,
)
from waveforge.ml.multitask_training import (
    MultitaskRunConfig,
    MultitaskRunStatus,
    run_multitask_training,
)
from waveforge.reproducibility import (
    artifact_sha256,
    configure_cuda_reproducibility,
)

MT2BVariant: TypeAlias = Literal["RAW", "PHYSICS"]
PROTOCOL_BUNDLE_SHA256 = (
    "567606c870720ca48001868efa9db1c6918e42345a1892932826c1ab0691d103"
)
MODEL_SEED = 2026092202
TASK_STREAM_SEED = 2026092201
UPDATES = 2000
BATCH_SIZE = 4
CHECKPOINT_INTERVAL = 250
_VARIANTS: tuple[MT2BVariant, ...] = ("RAW", "PHYSICS")


class MT2BExecutionError(RuntimeError):
    """Fail-closed provenance, authorization, or campaign error."""


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def build_variant_identity(
    variant: MT2BVariant,
    *,
    source_sha: str,
) -> dict[str, object]:
    """Return every immutable field needed to reject an unsafe resume."""
    if variant not in _VARIANTS:
        raise ValueError(f"unsupported MT2B variant {variant!r}")
    if len(source_sha) != 40 or any(
        character not in "0123456789abcdef" for character in source_sha
    ):
        raise ValueError("source_sha must be a lowercase 40-character Git SHA")
    return {
        "schema_version": 1,
        "experiment": "NCA-MT2B",
        "variant": variant,
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA256,
        "execution_source_sha": source_sha,
        "model_seed": MODEL_SEED,
        "task_stream_seed": TASK_STREAM_SEED,
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "training_mode": "scenario_vectorized_sequential",
        "validation_accessed": False,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }


def validate_variant_identity(
    path: Path,
    *,
    expected: dict[str, object],
) -> None:
    """Reject any resume whose scientific or executable identity differs."""
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MT2BExecutionError("run identity is missing or unreadable") from error
    if observed != expected:
        raise MT2BExecutionError("run identity does not match the authorized campaign")


def _protocol_bundle_sha256(root: Path) -> str:
    config = root / "configs" / "nca_mt2b.yaml"
    specification = (
        root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-08-31-nca-mt2b-physics-conditioning-design.md"
    )
    payload = f"{artifact_sha256(config)}\n{artifact_sha256(specification)}\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _validate_authorization(root: Path) -> None:
    if _protocol_bundle_sha256(root) != PROTOCOL_BUNDLE_SHA256:
        raise MT2BExecutionError("locked protocol bundle hash mismatch")
    authorization_path = (
        root / "artifacts" / "nca_mt2b_protocol" / "training_authorization.json"
    )
    try:
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MT2BExecutionError(
            "paired-training authorization is unavailable"
        ) from error
    if (
        authorization.get("status") != "PAIRED_TRAINING_AUTHORIZED"
        or authorization.get("protocol_bundle_sha256") != PROTOCOL_BUNDLE_SHA256
        or authorization.get("variants_in_locked_order") != list(_VARIANTS)
        or authorization.get("updates_per_variant") != UPDATES
    ):
        raise MT2BExecutionError("paired-training authorization is invalid")


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


def _completed_updates(output_dir: Path) -> int:
    checkpoint = _latest_checkpoint(output_dir)
    if checkpoint is None:
        return 0
    try:
        return int(checkpoint.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise MT2BExecutionError("checkpoint filename is invalid") from error


def run_variant(
    variant: MT2BVariant,
    *,
    output_root: Path,
    source_sha: str,
    chunk_updates: int,
) -> dict[str, object]:
    """Run or resume one locked variant in checkpoint-sized chunks."""
    variant_dir = output_root / variant.lower()
    variant_dir.mkdir(parents=True, exist_ok=True)
    identity = build_variant_identity(variant, source_sha=source_sha)
    identity_path = variant_dir / "run_identity.json"
    if identity_path.exists():
        validate_variant_identity(identity_path, expected=identity)
    else:
        if any(variant_dir.iterdir()):
            raise MT2BExecutionError("nonempty variant directory lacks run identity")
        _atomic_json(identity_path, identity)

    determinism = configure_cuda_reproducibility(MODEL_SEED)
    evaluator = build_mt2b_evaluator(variant)
    config = MultitaskRunConfig(
        model_seed=MODEL_SEED,
        task_seed=TASK_STREAM_SEED,
        total_updates=UPDATES,
        microbatch_size=BATCH_SIZE,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        mode="production",
        device="cuda",
        gradient_clip_norm=1.0,
    )
    completed = _completed_updates(variant_dir)
    while completed < UPDATES:
        checkpoint = _latest_checkpoint(variant_dir)
        result = run_multitask_training(
            config=config,
            output_dir=variant_dir,
            task_provider=mt2b_task_provider,
            evaluator=evaluator,
            model_factory=initialize_mt2b_model,
            resume_checkpoint=checkpoint,
            maximum_updates_this_call=min(chunk_updates, UPDATES - completed),
            synchronize=torch.cuda.synchronize,
        )
        if result.status is MultitaskRunStatus.INVALID_RUN:
            raise MT2BExecutionError(
                f"{variant} became INVALID_RUN: {','.join(result.reason_codes)}"
            )
        if result.completed_updates <= completed:
            raise MT2BExecutionError(f"{variant} made no checkpointed progress")
        completed = result.completed_updates
        progress = {
            "schema_version": 1,
            "variant": variant,
            "completed_updates": completed,
            "requested_updates": UPDATES,
            "status": result.status.value,
            "last_checkpoint": (
                result.last_checkpoint.name
                if result.last_checkpoint is not None
                else None
            ),
            "model_state_sha256": result.final_model_hash,
            "validation_accessed": False,
            "test_id_accessed": False,
            "test_ood_accessed": False,
        }
        _atomic_json(variant_dir / "training_progress.json", progress)
        print(
            f"MT2B_PROGRESS variant={variant} updates={completed}/{UPDATES} "
            f"checkpoint={progress['last_checkpoint']}",
            flush=True,
        )

    final_result = json.loads(
        (variant_dir / "multitask_run_result.json").read_text(encoding="utf-8")
    )
    if (
        final_result.get("status") != "PASS"
        or final_result.get("completed_updates") != UPDATES
    ):
        raise MT2BExecutionError(f"{variant} final training artifact is incomplete")
    return {
        "identity": identity,
        "determinism": asdict(determinism),
        "training_result": final_result,
    }


def run_campaign(
    *,
    root: Path,
    output_root: Path,
    source_sha: str,
    chunk_updates: int,
    only_variant: MT2BVariant | None = None,
) -> None:
    """Validate provenance and run RAW then PHYSICS without validation access."""
    if chunk_updates < 1 or chunk_updates > CHECKPOINT_INTERVAL:
        raise ValueError("chunk_updates must lie in [1,250]")
    _validate_authorization(root)
    if _current_source_sha(root) != source_sha:
        raise MT2BExecutionError(
            "working tree HEAD does not match execution source_sha"
        )
    variants = (only_variant,) if only_variant is not None else _VARIANTS
    campaign: dict[str, object] = {
        "schema_version": 1,
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA256,
        "execution_source_sha": source_sha,
        "variant_order": list(variants),
        "validation_accessed": False,
        "test_id_accessed": False,
        "test_ood_accessed": False,
        "variants": {},
    }
    for variant in variants:
        campaign["variants"][variant] = run_variant(  # type: ignore[index]
            variant,
            output_root=output_root,
            source_sha=source_sha,
            chunk_updates=chunk_updates,
        )
        _atomic_json(output_root / "campaign_status.json", campaign)
    print("MT2B_PAIRED_TRAINING_COMPLETE", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--chunk-updates", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--variant", choices=_VARIANTS)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    run_campaign(
        root=arguments.root.resolve(),
        output_root=arguments.output.resolve(),
        source_sha=arguments.source_sha,
        chunk_updates=arguments.chunk_updates,
        only_variant=arguments.variant,
    )


if __name__ == "__main__":
    main()
