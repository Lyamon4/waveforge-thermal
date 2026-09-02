"""Create a frozen MT3 development verdict and sealed-test authorization."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from waveforge.experiments.run_mt2b_evaluation import (
    _load_references,
    validation_tasks,
)
from waveforge.ml.mt2b_evaluation import independent_scipy64_tmax
from waveforge.ml.mt3_conditioning import build_mt3_conditioning, compute_initial_probe
from waveforge.ml.mt3_evaluation import (
    MT3CheckpointSummary,
    build_test_authorization_bundle,
    classify_mt3_development,
    select_mt3_checkpoint,
    summarize_mt3_checkpoint_rows,
)
from waveforge.ml.mt3_refinement import select_and_refine
from waveforge.ml.mt3_training import (
    MT3RunConfig,
    initialize_mt3_model,
    mt3_model_state_sha256,
)
from waveforge.ml.mt3_unet import project_mt3_candidates
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.verification.multitask_verification import (
    IndependentTaskVerification,
    verify_binary_task,
)

MT3Variant = Literal["FIELD_UNET", "SENS_UNET"]


class MT3ProductionEvaluationError(RuntimeError):
    """Fail-closed production checkpoint or development-evaluation error."""


@dataclass(frozen=True)
class MT3EvaluatedTask:
    """One solver-matched development layout and its frozen design arrays."""

    row: dict[str, object]
    candidate_binary_designs: np.ndarray
    refined_continuous_design: np.ndarray
    refined_binary_design: np.ndarray
    refinement_trace: tuple[dict[str, object], ...]


MT3TaskEvaluator = Callable[
    [int, SourceLayoutTask, float, Path, MT3Variant, str | torch.device],
    MT3EvaluatedTask,
]


def _checkpoint_completed_updates(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise MT3ProductionEvaluationError("malformed MT3 checkpoint name") from error


def production_checkpoint_paths(
    training_root: Path,
    variant: MT3Variant,
) -> tuple[Path, ...]:
    """Return the eight registered frozen checkpoints after identity checks."""
    directory = training_root / variant.lower()
    result_path = directory / "mt3_run_result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MT3ProductionEvaluationError(
            "unreadable MT3 production result"
        ) from error
    config = result.get("config", {})
    if (
        result.get("status") != "PASS"
        or result.get("completed_updates") != 4000
        or config.get("variant") != variant
        or config.get("mode") != "production"
        or result.get("test_id_accessed") is not False
        or result.get("test_ood_accessed") is not False
    ):
        raise MT3ProductionEvaluationError("MT3 production identity is invalid")
    paths = tuple(
        directory / f"checkpoint_{completed:06d}.pt"
        for completed in range(500, 4001, 500)
    )
    if any(not path.is_file() for path in paths):
        raise MT3ProductionEvaluationError(
            f"{variant} must contain all eight registered checkpoints"
        )
    return paths


def _load_production_model(
    checkpoint: Path,
    *,
    variant: MT3Variant,
    device: torch.device,
) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    try:
        config = MT3RunConfig(**payload["config"])
    except (KeyError, TypeError) as error:
        raise MT3ProductionEvaluationError("malformed MT3 checkpoint config") from error
    completed = _checkpoint_completed_updates(checkpoint)
    if (
        config.variant != variant
        or config.mode != "production"
        or config.total_updates != 4000
        or config.batch_size != 4
        or config.checkpoint_interval != 500
        or payload.get("completed_updates") != completed
    ):
        raise MT3ProductionEvaluationError("checkpoint violates production protocol")
    model = initialize_mt3_model(config.model_seed, device)
    model.load_state_dict(payload["model_state"])
    if mt3_model_state_sha256(model) != payload.get("model_state_sha256"):
        raise MT3ProductionEvaluationError("checkpoint model hash mismatch")
    return model.eval()


def _evaluate_task_with_model(
    task_index: int,
    task: SourceLayoutTask,
    reference_tmax: float,
    *,
    model: torch.nn.Module,
    variant: MT3Variant,
    device: torch.device,
) -> MT3EvaluatedTask:
    sources = torch.from_numpy(task.sources).to(device=device, dtype=torch.float64)
    probe = compute_initial_probe(
        sources.unsqueeze(0),
        allow_cpu_unit_test=False,
    )
    condition = build_mt3_conditioning(
        probe,
        sources.unsqueeze(0),
        variant=variant,
    )
    with torch.no_grad():
        logits = model(condition)[0]
        projected = project_mt3_candidates(logits.unsqueeze(0), beta=8.0)
        binary = projected.binary[0]
    refined = select_and_refine(
        logits,
        binary,
        task,
        sources,
        scorer=independent_scipy64_tmax,
        steps=25,
    )
    best4 = min(score.binary_tmax for score in refined.candidate_scores)
    r25_binary = refined.binary_design.numpy().astype(np.float64, copy=False)
    r25 = independent_scipy64_tmax(r25_binary, task)
    row: dict[str, object] = {
        "task_index": task_index,
        "task_id": task.task_id,
        "candidate_solver": "independent_scipy_64",
        "reference_solver": "independent_scipy_64",
        "reference_tmax_scipy64": float(reference_tmax),
        "best4_tmax_scipy64": float(best4),
        "r25_tmax_scipy64": float(r25),
        "best4_relative_gap": (float(best4) - reference_tmax) / reference_tmax,
        "r25_relative_gap": (float(r25) - reference_tmax) / reference_tmax,
        "selected_head": refined.selected_head,
        "binary_cell_count": int(refined.binary_design.sum().item()),
        "refinement_updates": refined.total_refinement_updates,
        "validation_accessed": True,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }
    return MT3EvaluatedTask(
        row=row,
        candidate_binary_designs=(
            binary.detach().cpu().numpy().astype(np.float64, copy=False)
        ),
        refined_continuous_design=(
            refined.continuous_design.numpy().astype(np.float64, copy=False)
        ),
        refined_binary_design=r25_binary,
        refinement_trace=tuple(asdict(record) for record in refined.records),
    )


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise MT3ProductionEvaluationError("cannot write empty evaluation rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _atomic_npz(path: Path, evaluated: MT3EvaluatedTask) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        candidate_binary_designs=evaluated.candidate_binary_designs,
        refined_continuous_design=evaluated.refined_continuous_design,
        refined_binary_design=evaluated.refined_binary_design,
    )
    temporary.replace(path)


def run_checkpoint_evaluation(
    checkpoint: Path,
    *,
    variant: MT3Variant,
    tasks: tuple[SourceLayoutTask, ...],
    reference_peaks: dict[str, float],
    output_dir: Path,
    device: str | torch.device,
    task_evaluator: MT3TaskEvaluator | None = None,
) -> MT3CheckpointSummary:
    """Evaluate or resume one frozen checkpoint on development layouts only."""
    if len(tasks) != 32 or set(reference_peaks) != {task.task_id for task in tasks}:
        raise ValueError("MT3 development evaluation requires 32 paired layouts")
    completed = _checkpoint_completed_updates(checkpoint)
    target_device = torch.device(device)
    if task_evaluator is None:
        if target_device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("production MT3 development evaluation requires CUDA")
        model = _load_production_model(
            checkpoint,
            variant=variant,
            device=target_device,
        )

        def evaluate(
            index: int,
            task: SourceLayoutTask,
            reference: float,
            _checkpoint: Path,
            _variant: MT3Variant,
            _device: str | torch.device,
        ) -> MT3EvaluatedTask:
            return _evaluate_task_with_model(
                index,
                task,
                reference,
                model=model,
                variant=variant,
                device=target_device,
            )

    else:
        evaluate = task_evaluator

    metrics_path = output_dir / "validation_metrics.csv"
    rows = _read_csv(metrics_path)
    if any(
        int(row.get("task_index", -1)) != index
        or row.get("task_id") != tasks[index].task_id
        or not (output_dir / "tasks" / f"task_{index:02d}.npz").is_file()
        or not (output_dir / "tasks" / f"task_{index:02d}_trace.json").is_file()
        for index, row in enumerate(rows)
    ):
        raise MT3ProductionEvaluationError("stored MT3 evaluation rows are corrupted")
    for index, task in enumerate(tasks[len(rows) :], start=len(rows)):
        evaluated = evaluate(
            index,
            task,
            float(reference_peaks[task.task_id]),
            checkpoint,
            variant,
            target_device,
        )
        if evaluated.row.get("task_id") != task.task_id:
            raise MT3ProductionEvaluationError("evaluator returned the wrong task")
        _atomic_npz(output_dir / "tasks" / f"task_{index:02d}.npz", evaluated)
        _atomic_json(
            output_dir / "tasks" / f"task_{index:02d}_trace.json",
            {
                "schema_version": 1,
                "task_index": index,
                "task_id": task.task_id,
                "records": list(evaluated.refinement_trace),
                "validation_accessed": True,
                "test_id_accessed": False,
                "test_ood_accessed": False,
            },
        )
        rows.append(evaluated.row)
        _atomic_csv(metrics_path, rows)
        print(
            f"MT3_DEVELOPMENT_PROGRESS variant={variant} "
            f"updates={completed} tasks={index + 1}/32",
            flush=True,
        )
    summary = summarize_mt3_checkpoint_rows(
        rows,
        completed_updates=completed,
        variant=variant,
    )
    _atomic_json(output_dir / "checkpoint_summary.json", asdict(summary))
    return summary


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run_production_development_evaluation(
    *,
    training_root: Path,
    reference_root: Path,
    output_root: Path,
    variants: tuple[MT3Variant, ...] = ("FIELD_UNET", "SENS_UNET"),
    device: str | torch.device = "cuda",
) -> list[MT3CheckpointSummary]:
    """Evaluate every registered checkpoint without instantiating test splits."""
    tasks = validation_tasks()
    _, reference_peaks = _load_references(reference_root, tasks)
    summaries: list[MT3CheckpointSummary] = []
    for variant in variants:
        for checkpoint in production_checkpoint_paths(training_root, variant):
            completed = _checkpoint_completed_updates(checkpoint)
            summary = run_checkpoint_evaluation(
                checkpoint,
                variant=variant,
                tasks=tasks,
                reference_peaks=reference_peaks,
                output_dir=(
                    output_root / variant.lower() / f"checkpoint_{completed:06d}"
                ),
                device=device,
            )
            summaries.append(summary)
    _atomic_json(
        output_root / "checkpoint_summaries.json",
        [asdict(summary) for summary in summaries],
    )
    return summaries


MT3Verifier256 = Callable[
    [np.ndarray, SourceLayoutTask],
    IndependentTaskVerification,
]


def _default_verifier_256(
    design: np.ndarray,
    task: SourceLayoutTask,
) -> IndependentTaskVerification:
    return verify_binary_task(design, task, resolution=256)


def verify_selected_development_256(
    *,
    evaluation_root: Path,
    reference_root: Path,
    selected_updates: int,
    output_path: Path,
    verifier: MT3Verifier256 = _default_verifier_256,
) -> list[dict[str, object]]:
    """Verify selected FIELD, SENS, and gradient designs on independent 256 grids."""
    if selected_updates <= 0 or selected_updates % 500 != 0:
        raise ValueError("selected MT3 updates must be a positive multiple of 500")
    tasks = validation_tasks()
    rows: list[dict[str, object]] = []
    for index, task in enumerate(tasks):
        designs: tuple[tuple[str, np.ndarray], ...] = (
            (
                "REFERENCE",
                np.load(
                    reference_root
                    / "references"
                    / f"task_{index:02d}"
                    / "binary_design_64.npy",
                    allow_pickle=False,
                ),
            ),
            (
                "FIELD_UNET_BEST4_R25",
                _load_selected_binary(
                    evaluation_root,
                    "field_unet",
                    selected_updates,
                    index,
                ),
            ),
            (
                "SENS_UNET_BEST4_R25",
                _load_selected_binary(
                    evaluation_root,
                    "sens_unet",
                    selected_updates,
                    index,
                ),
            ),
        )
        for family, design in designs:
            result = verifier(np.asarray(design, dtype=np.float64), task)
            if (
                result.task_id != task.task_id
                or result.resolution != 256
                or abs(result.material_fraction - 0.25) > 1.0e-12
                or result.maximum_normalized_residual > 1.0e-10
            ):
                raise MT3ProductionEvaluationError(
                    "selected independent verification became invalid"
                )
            row: dict[str, object] = {
                "task_index": index,
                "task_id": task.task_id,
                "family": family,
                **asdict(result),
                "validation_accessed": True,
                "test_id_accessed": False,
                "test_ood_accessed": False,
            }
            rows.append(row)
        print(
            f"MT3_VERIFY_256_PROGRESS tasks={index + 1}/32",
            flush=True,
        )
    _atomic_csv(output_path, rows)
    return rows


def _load_selected_binary(
    evaluation_root: Path,
    variant: str,
    selected_updates: int,
    task_index: int,
) -> np.ndarray:
    path = (
        evaluation_root
        / variant
        / f"checkpoint_{selected_updates:06d}"
        / "tasks"
        / f"task_{task_index:02d}.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        design = np.asarray(payload["refined_binary_design"], dtype=np.float64)
    if (
        design.shape != (64, 64)
        or not np.isin(design, (0.0, 1.0)).all()
        or int(np.count_nonzero(design)) != 1024
    ):
        raise MT3ProductionEvaluationError("selected MT3 binary design is invalid")
    return design


def freeze_development_verdict(
    *,
    summaries: list[MT3CheckpointSummary],
    implementation_commit: str,
    frozen_artifacts: dict[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    """Select the primary checkpoint, apply the gate, and hash authorization."""
    selected = select_mt3_checkpoint(summaries)
    verdict = classify_mt3_development(
        median_gap=selected.median_r25_relative_gap,
        p90_gap=selected.p90_r25_relative_gap,
        worst_gap=selected.worst_r25_relative_gap,
        wins=selected.r25_win_count,
        valid_count=selected.task_count - selected.invalid_count,
        exact_budget_count=selected.exact_budget_count,
    )
    _atomic_json(output_dir / "selected_checkpoint.json", asdict(selected))
    _atomic_json(output_dir / "development_verdict.json", asdict(verdict))
    bundle = build_test_authorization_bundle(
        verdict=verdict,
        implementation_commit=implementation_commit,
        artifacts=frozen_artifacts,
    )
    _atomic_json(output_dir / "test_authorization.json", bundle)
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("evaluate", "verify-256", "verdict"),
        required=True,
    )
    parser.add_argument("--summaries", type=Path)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--artifact", type=Path, action="append")
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--selected-updates", type=int)
    parser.add_argument(
        "--variant",
        choices=("FIELD_UNET", "SENS_UNET"),
        action="append",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.phase == "evaluate":
        if arguments.training_root is None or arguments.reference_root is None:
            raise SystemExit("evaluate requires --training-root and --reference-root")
        run_production_development_evaluation(
            training_root=arguments.training_root.resolve(),
            reference_root=arguments.reference_root.resolve(),
            output_root=arguments.output.resolve(),
            variants=tuple(arguments.variant or ("FIELD_UNET", "SENS_UNET")),
        )
        return
    if arguments.phase == "verify-256":
        if arguments.reference_root is None or arguments.selected_updates is None:
            raise SystemExit(
                "verify-256 requires --reference-root and --selected-updates"
            )
        verify_selected_development_256(
            evaluation_root=arguments.output.resolve(),
            reference_root=arguments.reference_root.resolve(),
            selected_updates=arguments.selected_updates,
            output_path=arguments.output.resolve() / "selected_verified_256.csv",
        )
        return
    if (
        arguments.summaries is None
        or arguments.implementation_commit is None
        or not arguments.artifact
    ):
        raise SystemExit(
            "verdict requires --summaries, --implementation-commit, and --artifact"
        )
    rows = json.loads(arguments.summaries.read_text(encoding="utf-8"))
    summaries = [MT3CheckpointSummary(**row) for row in rows]
    artifacts = {path.name: path.resolve() for path in arguments.artifact}
    freeze_development_verdict(
        summaries=summaries,
        implementation_commit=arguments.implementation_commit,
        frozen_artifacts=artifacts,
        output_dir=arguments.output.resolve(),
    )


if __name__ == "__main__":
    main()
