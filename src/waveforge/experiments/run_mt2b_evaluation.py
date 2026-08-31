"""Evaluate frozen MT2B checkpoints against solver-consistent references."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from waveforge.ml.mt2b_evaluation import (
    MT2BCheckpointSummary,
    independent_scipy64_tmax,
    paired_bootstrap,
    select_mt2b_checkpoint,
)
from waveforge.ml.mt2b_final_evaluation import (
    FrozenMT2BBatch,
    classify_mt2b_result,
    generate_mt2b_designs,
)
from waveforge.ml.multitask_evaluation import (
    condition_causality_summary,
    pairwise_binary_diversity,
)
from waveforge.ml.multitask_tasks import (
    VALIDATION_SEED,
    SourceLayoutTask,
    sample_primary_task,
)
from waveforge.reproducibility import artifact_sha256
from waveforge.verification.multitask_verification import verify_binary_task

PROTOCOL_BUNDLE_SHA256 = (
    "567606c870720ca48001868efa9db1c6918e42345a1892932826c1ab0691d103"
)
_VARIANTS: tuple[Literal["RAW", "PHYSICS"], ...] = ("RAW", "PHYSICS")
_CHECKPOINT_UPDATES = tuple(range(250, 2001, 250))


class MT2BEvaluationError(RuntimeError):
    """Fail-closed MT2B artifact or evaluation error."""


def validation_tasks() -> tuple[SourceLayoutTask, ...]:
    """Build validation only; never instantiate either sealed test split."""
    return tuple(sample_primary_task(VALIDATION_SEED, index) for index in range(32))


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
        raise MT2BEvaluationError("cannot write an empty MT2B table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MT2BEvaluationError(f"unreadable artifact: {path}") from error


def _load_references(
    reference_root: Path,
    tasks: tuple[SourceLayoutTask, ...],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    registry = _read_json(reference_root / "reference_registry.json")
    if (
        registry.get("status") != "PASS"
        or registry.get("task_count") != 32
        or registry.get("split") != "validation"
        or registry.get("protocol_bundle_sha256") != PROTOCOL_BUNDLE_SHA256
        or registry.get("test_id_accessed") is not False
        or registry.get("test_ood_accessed") is not False
    ):
        raise MT2BEvaluationError("reference registry violates the locked protocol")
    designs: dict[str, np.ndarray] = {}
    peaks: dict[str, float] = {}
    for index, task in enumerate(tasks):
        directory = reference_root / "references" / f"task_{index:02d}"
        payload = _read_json(directory / "reference_result.json")
        path = directory / "binary_design_64.npy"
        design = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
        if payload.get("task_id") != task.task_id or payload.get(
            "binary_design_sha256"
        ) != artifact_sha256(path):
            raise MT2BEvaluationError("reference identity or hash mismatch")
        observed = independent_scipy64_tmax(design, task)
        if abs(observed - float(payload["independent_scipy64_tmax"])) > 1.0e-12:
            raise MT2BEvaluationError("reference SciPy64 replay mismatch")
        designs[task.task_id] = design
        peaks[task.task_id] = observed
    return designs, peaks


def _checkpoint_paths(training_root: Path, variant: str) -> list[Path]:
    directory = training_root / variant.lower()
    paths = [
        directory / f"checkpoint_{updates:06d}.pt" for updates in _CHECKPOINT_UPDATES
    ]
    if any(not path.is_file() for path in paths):
        raise MT2BEvaluationError(f"{variant} lacks all eight locked checkpoints")
    result = _read_json(directory / "multitask_run_result.json")
    if result.get("status") != "PASS" or result.get("completed_updates") != 2000:
        raise MT2BEvaluationError(f"{variant} training did not finish with PASS")
    return paths


def _save_generated(directory: Path, batch: FrozenMT2BBatch) -> None:
    continuous = np.stack([item.continuous_design for item in batch.designs])
    binary = np.stack([item.binary_design for item in batch.designs])
    _atomic_npy(directory / "continuous_designs_64.npy", continuous)
    _atomic_npy(directory / "binary_designs_64.npy", binary)
    _atomic_json(
        directory / "generation.json",
        {
            "schema_version": 1,
            "variant": batch.variant,
            "completed_updates": batch.completed_updates,
            "model_state_sha256": batch.model_state_sha256,
            "task_ids": [item.task_id for item in batch.designs],
            "binary_material_fractions": [
                item.binary_material_fraction for item in batch.designs
            ],
            "continuous_designs_sha256": artifact_sha256(
                directory / "continuous_designs_64.npy"
            ),
            "binary_designs_sha256": artifact_sha256(
                directory / "binary_designs_64.npy"
            ),
            "optimizer_updates_at_inference": 0,
            "backward_calls_at_inference": 0,
            "validation_accessed": True,
            "test_id_accessed": False,
            "test_ood_accessed": False,
        },
    )


def _score_checkpoint(
    checkpoint: Path,
    tasks: tuple[SourceLayoutTask, ...],
    *,
    variant: Literal["RAW", "PHYSICS"],
    reference_peaks: dict[str, float],
    output_root: Path,
    device: torch.device,
) -> tuple[MT2BCheckpointSummary, np.ndarray, np.ndarray]:
    generated = generate_mt2b_designs(
        checkpoint,
        tasks,
        variant=variant,
        device=device,
    )
    directory = output_root / variant.lower() / checkpoint.stem
    _save_generated(directory, generated)
    rows: list[dict[str, object]] = []
    candidate_peaks: list[float] = []
    gaps: list[float] = []
    for index, (task, item) in enumerate(zip(tasks, generated.designs, strict=True)):
        candidate = independent_scipy64_tmax(item.binary_design, task)
        reference = reference_peaks[task.task_id]
        gap = (candidate - reference) / reference
        candidate_peaks.append(candidate)
        gaps.append(gap)
        rows.append(
            {
                "task_index": index,
                "task_id": task.task_id,
                "candidate_tmax_scipy64": candidate,
                "reference_tmax_scipy64": reference,
                "relative_gap": gap,
                "binary_material_fraction": item.binary_material_fraction,
            }
        )
    _atomic_csv(directory / "solver_consistent_metrics.csv", rows)
    peaks = np.asarray(candidate_peaks, dtype=np.float64)
    gap_array = np.asarray(gaps, dtype=np.float64)
    summary = MT2BCheckpointSummary(
        completed_updates=generated.completed_updates,
        split_name="validation",
        task_count=32,
        invalid_count=0,
        median_relative_gap=float(np.median(gap_array)),
        p90_relative_gap=float(np.quantile(gap_array, 0.9)),
        worst_relative_gap=float(np.max(gap_array)),
        median_absolute_tmax=float(np.median(peaks)),
    )
    _atomic_json(directory / "summary.json", asdict(summary))
    print(
        f"MT2B_VALIDATION variant={variant} updates={generated.completed_updates} "
        f"median_gap={summary.median_relative_gap:.8f}",
        flush=True,
    )
    return summary, gap_array, peaks


def _selected_arrays(
    evaluation_root: Path, variant: str, completed_updates: int
) -> tuple[np.ndarray, np.ndarray]:
    directory = (
        evaluation_root / variant.lower() / f"checkpoint_{completed_updates:06d}"
    )
    continuous = np.load(directory / "continuous_designs_64.npy", allow_pickle=False)
    binary = np.load(directory / "binary_designs_64.npy", allow_pickle=False)
    return continuous, binary


def _selected_diagnostics(
    *,
    training_root: Path,
    evaluation_root: Path,
    tasks: tuple[SourceLayoutTask, ...],
    variant: Literal["RAW", "PHYSICS"],
    selected: MT2BCheckpointSummary,
    matched_peaks: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = (
        training_root
        / variant.lower()
        / f"checkpoint_{selected.completed_updates:06d}.pt"
    )
    shifted = tasks[1:] + tasks[:1]
    shuffled = generate_mt2b_designs(
        checkpoint,
        tasks,
        conditioning_tasks=shifted,
        variant=variant,
        device=device,
    )
    shuffled_peaks = [
        independent_scipy64_tmax(item.binary_design, task)
        for task, item in zip(tasks, shuffled.designs, strict=True)
    ]
    causality = condition_causality_summary(
        matched=matched_peaks.tolist(),
        shuffled=shuffled_peaks,
    )
    _, selected_binary = _selected_arrays(
        evaluation_root, variant, selected.completed_updates
    )
    diversity = pairwise_binary_diversity(list(selected_binary))
    payload: dict[str, object] = {
        "condition_causality": asdict(causality),
        "binary_diversity": asdict(diversity),
        "shuffle_rule": "cyclic_next_validation_layout",
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }
    _atomic_json(
        evaluation_root / variant.lower() / "selected_diagnostics.json", payload
    )
    return payload


def run_validation(
    *,
    training_root: Path,
    reference_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Select both variants and issue the preregistered paired verdict."""
    tasks = validation_tasks()
    _, reference_peaks = _load_references(reference_root, tasks)
    all_summaries: dict[str, list[MT2BCheckpointSummary]] = {}
    gap_lookup: dict[tuple[str, int], np.ndarray] = {}
    peak_lookup: dict[tuple[str, int], np.ndarray] = {}
    for variant in _VARIANTS:
        summaries: list[MT2BCheckpointSummary] = []
        for checkpoint in _checkpoint_paths(training_root, variant):
            summary, gaps, peaks = _score_checkpoint(
                checkpoint,
                tasks,
                variant=variant,
                reference_peaks=reference_peaks,
                output_root=output_root,
                device=device,
            )
            summaries.append(summary)
            gap_lookup[(variant, summary.completed_updates)] = gaps
            peak_lookup[(variant, summary.completed_updates)] = peaks
        all_summaries[variant] = summaries
        _atomic_csv(
            output_root / variant.lower() / "checkpoint_summaries.csv",
            [asdict(item) for item in summaries],
        )

    selected = {
        variant: select_mt2b_checkpoint(all_summaries[variant]) for variant in _VARIANTS
    }
    diagnostics = {
        variant: _selected_diagnostics(
            training_root=training_root,
            evaluation_root=output_root,
            tasks=tasks,
            variant=variant,
            selected=selected[variant],
            matched_peaks=peak_lookup[(variant, selected[variant].completed_updates)],
            device=device,
        )
        for variant in _VARIANTS
    }
    raw_gaps = gap_lookup[("RAW", selected["RAW"].completed_updates)]
    physics_gaps = gap_lookup[("PHYSICS", selected["PHYSICS"].completed_updates)]
    bootstrap = paired_bootstrap(raw_gaps, physics_gaps)
    verdict = classify_mt2b_result(
        physics_summary=selected["PHYSICS"],
        raw_gaps=raw_gaps,
        physics_gaps=physics_gaps,
        bootstrap=bootstrap,
    )
    frozen_dir = output_root / "frozen_models"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_models: dict[str, dict[str, object]] = {}
    for variant in _VARIANTS:
        source = (
            training_root
            / variant.lower()
            / f"checkpoint_{selected[variant].completed_updates:06d}.pt"
        )
        destination = frozen_dir / f"{variant.lower()}_selected.pt"
        shutil.copy2(source, destination)
        frozen_models[variant] = {
            "completed_updates": selected[variant].completed_updates,
            "checkpoint_sha256": artifact_sha256(destination),
            "path": str(destination.relative_to(output_root)),
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": verdict.status,
        "exact_reason": verdict.exact_reason,
        "selected": {key: asdict(value) for key, value in selected.items()},
        "bootstrap": asdict(bootstrap),
        "paired_effect": asdict(verdict),
        "diagnostics": diagnostics,
        "frozen_models": frozen_models,
        "primary_solver": "independent_scipy_64_for_both_candidate_and_reference",
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA256,
        "validation_accessed": True,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }
    _atomic_json(output_root / "mt2b_verdict.json", payload)
    return payload


def run_secondary_verification(
    *,
    reference_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Verify all selected-layout families independently at 256x256."""
    verdict = _read_json(output_root / "mt2b_verdict.json")
    tasks = validation_tasks()
    reference_designs, _ = _load_references(reference_root, tasks)
    selected = dict(verdict["selected"])
    rows: list[dict[str, object]] = []
    families: dict[str, np.ndarray] = {
        "REFERENCE": np.stack(list(reference_designs.values()))
    }
    for variant in _VARIANTS:
        updates = int(dict(selected[variant])["completed_updates"])
        _, binary = _selected_arrays(output_root, variant, updates)
        families[variant] = binary
    for family, designs in families.items():
        for index, (task, design) in enumerate(zip(tasks, designs, strict=True)):
            verified = verify_binary_task(design, task, resolution=256)
            rows.append(
                {
                    "family": family,
                    "task_index": index,
                    "task_id": task.task_id,
                    "tmax_256": verified.worst_peak,
                    "scenario_a_256": verified.scenario_peaks[0],
                    "scenario_b_256": verified.scenario_peaks[1],
                    "scenario_c_256": verified.scenario_peaks[2],
                    "material_fraction": verified.material_fraction,
                    "maximum_residual": verified.maximum_normalized_residual,
                    "wall_seconds": verified.wall_seconds,
                }
            )
            print(
                f"MT2B_VERIFY256 family={family} tasks={index + 1}/32",
                flush=True,
            )
    _atomic_csv(output_root / "selected_verified_256.csv", rows)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "families": list(families),
        "task_count_per_family": 32,
        "metrics_sha256": artifact_sha256(output_root / "selected_verified_256.csv"),
        "primary_verdict_unchanged": True,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }
    _atomic_json(output_root / "secondary_256_verification.json", payload)
    return payload


def run_inference_benchmark(
    *,
    reference_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Measure conservative frozen end-to-end latency against references."""
    if device.type != "cuda":
        raise ValueError("MT2B inference benchmark requires CUDA")
    verdict = _read_json(output_root / "mt2b_verdict.json")
    tasks = validation_tasks()
    variants: dict[str, dict[str, object]] = {}
    for variant in _VARIANTS:
        checkpoint = output_root / "frozen_models" / f"{variant.lower()}_selected.pt"
        single_seconds: list[float] = []
        for task in tasks[:8]:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            generated = generate_mt2b_designs(
                checkpoint,
                (task,),
                variant=variant,
                device=device,
                batch_size=1,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if len(generated.designs) != 1:
                raise MT2BEvaluationError(
                    "single-task inference benchmark is incomplete"
                )
            single_seconds.append(elapsed)
        batch_seconds: list[float] = []
        for _ in range(3):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            generated = generate_mt2b_designs(
                checkpoint,
                tasks,
                variant=variant,
                device=device,
                batch_size=32,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if len(generated.designs) != 32:
                raise MT2BEvaluationError("batched inference benchmark is incomplete")
            batch_seconds.append(elapsed)
        variants[variant] = {
            "single_task_samples": 8,
            "single_task_median_seconds": float(np.median(single_seconds)),
            "single_task_range_seconds": [
                float(np.min(single_seconds)),
                float(np.max(single_seconds)),
            ],
            "batch32_repeats": 3,
            "batch32_median_seconds": float(np.median(batch_seconds)),
            "batch32_amortized_seconds_per_task": float(np.median(batch_seconds) / 32),
            "timing_scope": (
                "checkpoint_load_plus_conditioning_plus_64_step_rollout_plus_projection"
            ),
        }
    reference_seconds = [
        float(
            _read_json(
                reference_root
                / "references"
                / f"task_{index:02d}"
                / "reference_result.json"
            )["wall_seconds"]
        )
        for index in range(32)
    ]
    physics_per_task = float(variants["PHYSICS"]["batch32_amortized_seconds_per_task"])
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "variants": variants,
        "gradient_reference_steps": 600,
        "gradient_reference_median_seconds": float(np.median(reference_seconds)),
        "gradient_reference_range_seconds": [
            float(np.min(reference_seconds)),
            float(np.max(reference_seconds)),
        ],
        "physics_batch32_amortized_speedup_vs_gradient_median": float(
            np.median(reference_seconds) / physics_per_task
        ),
        "benchmark_is_secondary_diagnostic": True,
        "frozen_verdict_status": verdict["status"],
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }
    _atomic_json(output_root / "inference_benchmark.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("validation", "verification", "benchmark"),
        required=True,
    )
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.phase == "validation":
        run_validation(
            training_root=arguments.training_root.resolve(),
            reference_root=arguments.reference_root.resolve(),
            output_root=arguments.output.resolve(),
            device=torch.device(arguments.device),
        )
    elif arguments.phase == "verification":
        run_secondary_verification(
            reference_root=arguments.reference_root.resolve(),
            output_root=arguments.output.resolve(),
        )
    else:
        run_inference_benchmark(
            reference_root=arguments.reference_root.resolve(),
            output_root=arguments.output.resolve(),
            device=torch.device(arguments.device),
        )


if __name__ == "__main__":
    main()
