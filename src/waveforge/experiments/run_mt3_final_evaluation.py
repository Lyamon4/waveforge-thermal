"""Run the frozen MT3 ID/OOD evaluation in resumable, hash-checked phases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from waveforge.design.batched_adam_baseline import optimize_adam_batched
from waveforge.design.mma_baseline import optimize_mma
from waveforge.ml.mt2b_evaluation import independent_scipy64_tmax
from waveforge.ml.mt3_conditioning import build_mt3_conditioning, compute_initial_probe
from waveforge.ml.mt3_final_evaluation import (
    classify_mt3_test_split,
    load_authorized_task_splits,
    median_gap_bootstrap,
    registered_baseline_jobs,
)
from waveforge.ml.mt3_refinement import select_and_refine_trajectory
from waveforge.ml.mt3_unet import project_mt3_candidates
from waveforge.ml.multitask_tasks import (
    FrozenTaskSplits,
    SourceLayoutTask,
    build_frozen_splits,
)
from waveforge.reproducibility import artifact_sha256, configure_cuda_reproducibility
from waveforge.verification.multitask_verification import verify_binary_task

_BUDGETS = (25, 50, 100, 200, 600)
_VARIANTS = ("FIELD_UNET", "SENS_UNET")


class MT3FinalRunError(RuntimeError):
    """Fail-closed final-test execution error."""


def exact_binary64(design: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate and return one exact 1024-cell strict-binary design."""
    array = np.asarray(design, dtype=np.float64)
    if array.shape != (64, 64) or not np.isin(array, (0.0, 1.0)).all():
        raise ValueError("design must be strict binary with shape [64,64]")
    if int(np.count_nonzero(array)) != 1024:
        raise ValueError("strict binary design must contain exactly 1024 cells")
    return array


def strong_single_reference(*, adam_tmax: float, mma_tmax: float) -> tuple[str, float]:
    """Apply the locked lower-Tmax rule with Adam as the exact-tie winner."""
    if not all(math.isfinite(value) and value > 0.0 for value in (adam_tmax, mma_tmax)):
        raise ValueError("baseline Tmax values must be finite and positive")
    if adam_tmax <= mma_tmax:
        return "ADAM_600", adam_tmax
    return "MMA_600", mma_tmax


def neural_equivalent_evaluations(*, refinement_updates: int) -> int:
    """Count one probe, four candidate scores, and one refinement chain."""
    if refinement_updates not in (25, 50):
        raise ValueError("refinement updates must be exactly 25 or 50")
    return 1 + 4 + refinement_updates


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MT3FinalRunError(f"unreadable artifact: {path}") from error
    if not isinstance(payload, dict):
        raise MT3FinalRunError(f"artifact must contain an object: {path}")
    return payload


def _valid_result(result_path: Path, array_path: Path, *, task_id: str) -> bool:
    if not result_path.is_file() and not array_path.is_file():
        return False
    if not result_path.is_file() or not array_path.is_file():
        raise MT3FinalRunError("partial completed artifact detected")
    payload = _read_json(result_path)
    if (
        payload.get("status") != "PASS"
        or payload.get("task_id") != task_id
        or payload.get("arrays_sha256") != artifact_sha256(array_path)
    ):
        raise MT3FinalRunError("stored completed artifact failed identity check")
    return True


def _open_splits(authorization: Path, expected_sha256: str) -> FrozenTaskSplits:
    return load_authorized_task_splits(
        authorization,
        expected_bundle_sha256=expected_sha256,
        split_factory=build_frozen_splits,
    )


def _split_tasks(
    splits: FrozenTaskSplits,
) -> tuple[tuple[str, tuple[SourceLayoutTask, ...]], ...]:
    return (("test_id", splits.test_id), ("test_ood", splits.test_ood))


def _write_task_manifest(output: Path, splits: FrozenTaskSplits) -> None:
    path = output / "opened_task_manifest.json"
    payload = {
        "schema_version": 1,
        "opened_once_for_frozen_final_evaluation": True,
        "test_id": [
            {
                "index": index,
                "task_id": task.task_id,
                "centers": [list(center) for center in task.centers],
            }
            for index, task in enumerate(splits.test_id)
        ],
        "test_ood": [
            {
                "index": index,
                "task_id": task.task_id,
                "centers": [list(center) for center in task.centers],
            }
            for index, task in enumerate(splits.test_ood)
        ],
    }
    if path.is_file():
        if _read_json(path) != payload:
            raise MT3FinalRunError("opened task manifest changed")
        return
    _atomic_json(path, payload)


def _load_model(
    checkpoint: Path, variant: str, device: torch.device
) -> torch.nn.Module:
    from waveforge.experiments.run_mt3_evaluation import _load_production_model

    return _load_production_model(checkpoint, variant=variant, device=device)  # type: ignore[arg-type]


def _cuda_seconds(action: Any, device: torch.device) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = action()
    torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def _run_neural_task(
    *,
    task: SourceLayoutTask,
    model: torch.nn.Module,
    variant: str,
    device: torch.device,
    destination: Path,
) -> dict[str, object]:
    arrays_path = destination / "designs.npz"
    result_path = destination / "result.json"
    if _valid_result(result_path, arrays_path, task_id=task.task_id):
        return _read_json(result_path)
    sources = torch.as_tensor(task.sources, dtype=torch.float64, device=device)

    probe, probe_seconds = _cuda_seconds(
        lambda: compute_initial_probe(sources.unsqueeze(0)), device
    )
    condition = build_mt3_conditioning(
        probe,
        sources.unsqueeze(0),
        variant=variant,  # type: ignore[arg-type]
    )

    def generate() -> tuple[torch.Tensor, Any]:
        with torch.no_grad():
            logits = model(condition)[0]
            projected = project_mt3_candidates(logits.unsqueeze(0), beta=8.0)
        return logits, projected

    generated, inference_seconds = _cuda_seconds(generate, device)
    logits, projected = generated
    trajectory, refinement_seconds = _cuda_seconds(
        lambda: select_and_refine_trajectory(
            logits,
            projected.binary[0],
            task,
            sources,
            scorer=independent_scipy64_tmax,
            steps=(25, 50),
        ),
        device,
    )
    scores = trajectory[50].candidate_scores
    selected_head = trajectory[50].selected_head
    binaries = projected.binary[0].detach().cpu().numpy().astype(np.float64)
    continuous = projected.designs[0].detach().cpu().numpy().astype(np.float64)
    r25_binary = exact_binary64(trajectory[25].binary_design.numpy())
    r50_binary = exact_binary64(trajectory[50].binary_design.numpy())
    r25_tmax = independent_scipy64_tmax(r25_binary, task)
    r50_tmax = independent_scipy64_tmax(r50_binary, task)
    _atomic_npz(
        arrays_path,
        candidate_logits=logits.detach().cpu().numpy(),
        candidate_continuous_designs=continuous,
        candidate_binary_designs=binaries,
        r25_continuous_design=trajectory[25].continuous_design.numpy(),
        r25_binary_design=r25_binary,
        r50_continuous_design=trajectory[50].continuous_design.numpy(),
        r50_binary_design=r50_binary,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "task_id": task.task_id,
        "variant": variant,
        "selected_head": selected_head,
        "head0_tmax_scipy64": scores[0].binary_tmax,
        "best4_tmax_scipy64": min(item.binary_tmax for item in scores),
        "candidate_tmax_scipy64": [item.binary_tmax for item in scores],
        "r25_tmax_scipy64": r25_tmax,
        "r50_tmax_scipy64": r50_tmax,
        "binary_cell_count": 1024,
        "probe_seconds": probe_seconds,
        "unet_forward_seconds": inference_seconds,
        "single_r50_chain_seconds": refinement_seconds,
        "r25_equivalent_evaluations": neural_equivalent_evaluations(
            refinement_updates=25
        ),
        "r50_equivalent_evaluations": neural_equivalent_evaluations(
            refinement_updates=50
        ),
        "arrays_sha256": artifact_sha256(arrays_path),
    }
    _atomic_json(result_path, payload)
    return payload


def run_neural_phase(
    *,
    splits: FrozenTaskSplits,
    training_root: Path,
    output: Path,
    device: torch.device,
) -> None:
    total = sum(len(tasks) for _, tasks in _split_tasks(splits)) * len(_VARIANTS)
    completed = 0
    for variant in _VARIANTS:
        checkpoint = training_root / variant.lower() / "checkpoint_004000.pt"
        model = _load_model(checkpoint, variant, device)
        for split, tasks in _split_tasks(splits):
            for index, task in enumerate(tasks):
                _run_neural_task(
                    task=task,
                    model=model,
                    variant=variant,
                    device=device,
                    destination=(
                        output
                        / "neural"
                        / variant.lower()
                        / split
                        / f"task_{index:02d}"
                    ),
                )
                completed += 1
                print(
                    f"MT3_FINAL_NEURAL {completed}/{total} {variant} {split} {index}",
                    flush=True,
                )


def _task_lookup(splits: FrozenTaskSplits) -> dict[tuple[str, int], SourceLayoutTask]:
    return {
        (split, index): task
        for split, tasks in _split_tasks(splits)
        for index, task in enumerate(tasks)
    }


def _baseline_destination(
    output: Path, method: str, split: str, task_index: int, start_index: int
) -> Path:
    return (
        output
        / "baselines"
        / method.lower()
        / split
        / f"task_{task_index:02d}"
        / f"start_{start_index}"
    )


def _write_adam_result(
    destination: Path,
    *,
    task: SourceLayoutTask,
    job: Any,
    result: Any,
    batch_seconds: float,
    batch_size: int,
) -> None:
    arrays_path = destination / "designs.npz"
    _atomic_npz(
        arrays_path,
        final_logits=result.final_logits,
        **{
            f"binary_{budget:03d}": exact_binary64(
                result.snapshots[budget].binary_design
            )
            for budget in _BUDGETS
        },
    )
    scores = {
        str(budget): independent_scipy64_tmax(
            exact_binary64(result.snapshots[budget].binary_design), task
        )
        for budget in _BUDGETS
    }
    _atomic_json(
        destination / "result.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "job_id": job.job_id,
            "method": "ADAM",
            "split": job.split,
            "task_index": job.task_index,
            "task_id": task.task_id,
            "start_index": job.start_index,
            "seed": job.seed,
            "completed_updates": 600,
            "snapshot_tmax_scipy64": scores,
            "batch_wall_seconds": batch_seconds,
            "allocated_task_wall_seconds": batch_seconds / batch_size,
            "arrays_sha256": artifact_sha256(arrays_path),
        },
    )


def _run_adam_jobs(
    *,
    splits: FrozenTaskSplits,
    output: Path,
    device: torch.device,
) -> None:
    lookup = _task_lookup(splits)
    jobs = [
        job
        for job in registered_baseline_jobs(
            id_task_ids=tuple(task.task_id for task in splits.test_id),
            ood_task_ids=tuple(task.task_id for task in splits.test_ood),
        )
        if job.method == "ADAM"
    ]
    groups: list[list[Any]] = []
    for split in ("test_id", "test_ood"):
        for start in (0, 1, 2, 3):
            selected = [
                job for job in jobs if job.split == split and job.start_index == start
            ]
            groups.extend(
                selected[index : index + 4] for index in range(0, len(selected), 4)
            )
    for group_index, group in enumerate(groups):
        destinations = [
            _baseline_destination(
                output, job.method, job.split, job.task_index, job.start_index
            )
            for job in group
        ]
        states = [
            _valid_result(
                destination / "result.json",
                destination / "designs.npz",
                task_id=job.task_id,
            )
            for destination, job in zip(destinations, group, strict=True)
        ]
        if all(states):
            print(f"MT3_FINAL_ADAM {group_index + 1}/{len(groups)} cached", flush=True)
            continue
        if any(states):
            raise MT3FinalRunError("partial completed Adam batch cannot be regrouped")
        tasks = tuple(lookup[(job.split, job.task_index)] for job in group)
        configure_cuda_reproducibility(group[0].seed)
        started = time.perf_counter()
        result = optimize_adam_batched(
            tasks,
            seeds=tuple(job.seed for job in group),
            total_updates=600,
            snapshot_updates=_BUDGETS,
            device=device,
        )
        batch_seconds = time.perf_counter() - started
        for destination, task, job, task_result in zip(
            destinations, tasks, group, result.tasks, strict=True
        ):
            _write_adam_result(
                destination,
                task=task,
                job=job,
                result=task_result,
                batch_seconds=batch_seconds,
                batch_size=len(group),
            )
        print(
            f"MT3_FINAL_ADAM {group_index + 1}/{len(groups)} "
            f"seconds={batch_seconds:.1f}",
            flush=True,
        )


def _run_mma_jobs(
    *,
    splits: FrozenTaskSplits,
    output: Path,
    device: torch.device,
) -> None:
    lookup = _task_lookup(splits)
    jobs = [
        job
        for job in registered_baseline_jobs(
            id_task_ids=tuple(task.task_id for task in splits.test_id),
            ood_task_ids=tuple(task.task_id for task in splits.test_ood),
        )
        if job.method == "MMA"
    ]
    for index, job in enumerate(jobs):
        task = lookup[(job.split, job.task_index)]
        destination = _baseline_destination(
            output, job.method, job.split, job.task_index, job.start_index
        )
        arrays_path = destination / "designs.npz"
        result_path = destination / "result.json"
        if _valid_result(result_path, arrays_path, task_id=task.task_id):
            print(f"MT3_FINAL_MMA {index + 1}/{len(jobs)} cached", flush=True)
            continue
        configure_cuda_reproducibility(job.seed)
        started = time.perf_counter()
        result = optimize_mma(
            task,
            evaluations=600,
            seed=job.seed,
            device=device,
            snapshot_evaluations=_BUDGETS,
        )
        wall_seconds = time.perf_counter() - started
        if (
            result.status != "PASS"
            or result.completed_evaluations != 600
            or set(result.snapshots) != set(_BUDGETS)
        ):
            raise MT3FinalRunError(f"MMA job failed: {job.job_id}")
        _atomic_npz(
            arrays_path,
            final_logits=result.final_logits,
            **{
                f"binary_{budget:03d}": exact_binary64(
                    result.snapshots[budget].binary_design
                )
                for budget in _BUDGETS
            },
        )
        scores = {
            str(budget): independent_scipy64_tmax(
                exact_binary64(result.snapshots[budget].binary_design), task
            )
            for budget in _BUDGETS
        }
        _atomic_json(
            result_path,
            {
                "schema_version": 1,
                "status": "PASS",
                "job_id": job.job_id,
                "method": "MMA",
                "split": job.split,
                "task_index": job.task_index,
                "task_id": task.task_id,
                "start_index": 0,
                "seed": job.seed,
                "completed_evaluations": result.completed_evaluations,
                "termination_codes": result.termination_codes,
                "snapshot_tmax_scipy64": scores,
                "wall_seconds": wall_seconds,
                "arrays_sha256": artifact_sha256(arrays_path),
            },
        )
        print(
            f"MT3_FINAL_MMA {index + 1}/{len(jobs)} seconds={wall_seconds:.1f}",
            flush=True,
        )


def run_baseline_phase(
    *, splits: FrozenTaskSplits, output: Path, device: torch.device, method: str
) -> None:
    if method == "adam":
        _run_adam_jobs(splits=splits, output=output, device=device)
    elif method == "mma":
        _run_mma_jobs(splits=splits, output=output, device=device)
    else:
        raise ValueError("baseline method must be adam or mma")


def _load_npz_design(path: Path, key: str) -> NDArray[np.float64]:
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload:
            raise MT3FinalRunError(f"missing design {key} in {path}")
        return exact_binary64(np.asarray(payload[key], dtype=np.float64))


def _verify_family(
    *, design: NDArray[np.float64], task: SourceLayoutTask
) -> dict[str, object]:
    return asdict(verify_binary_task(design, task, resolution=256))


def run_verification_phase(*, splits: FrozenTaskSplits, output: Path) -> None:
    verification_rows: list[dict[str, object]] = []
    split_gap_rows: dict[str, list[dict[str, object]]] = {
        "test_id": [],
        "test_ood": [],
    }
    multistart_rows: list[dict[str, object]] = []
    for split, tasks in _split_tasks(splits):
        for task_index, task in enumerate(tasks):
            neural_paths = {
                variant: output
                / "neural"
                / variant.lower()
                / split
                / f"task_{task_index:02d}"
                / "designs.npz"
                for variant in _VARIANTS
            }
            adam_path = (
                _baseline_destination(output, "ADAM", split, task_index, 0)
                / "designs.npz"
            )
            mma_path = (
                _baseline_destination(output, "MMA", split, task_index, 0)
                / "designs.npz"
            )
            sens_payload = _read_json(neural_paths["SENS_UNET"].parent / "result.json")
            selected_head = int(sens_payload["selected_head"])
            with np.load(neural_paths["SENS_UNET"], allow_pickle=False) as payload:
                sens_best4 = exact_binary64(
                    np.asarray(payload["candidate_binary_designs"][selected_head])
                )
            designs = {
                "SENS_UNET_BEST4_R25": _load_npz_design(
                    neural_paths["SENS_UNET"], "r25_binary_design"
                ),
                "FIELD_UNET_BEST4_R25": _load_npz_design(
                    neural_paths["FIELD_UNET"], "r25_binary_design"
                ),
                "SENS_UNET_BEST4": sens_best4,
                "SENS_UNET_BEST4_R50": _load_npz_design(
                    neural_paths["SENS_UNET"], "r50_binary_design"
                ),
                "ADAM_600": _load_npz_design(adam_path, "binary_600"),
                "MMA_600": _load_npz_design(mma_path, "binary_600"),
            }
            verified: dict[str, float] = {}
            for family, design in designs.items():
                row = _verify_family(design=design, task=task)
                verified[family] = float(row["worst_peak"])
                verification_rows.append(
                    {
                        "split": split,
                        "task_index": task_index,
                        "family": family,
                        **row,
                    }
                )
            winner, reference = strong_single_reference(
                adam_tmax=verified["ADAM_600"], mma_tmax=verified["MMA_600"]
            )
            candidate = verified["SENS_UNET_BEST4_R25"]
            split_gap_rows[split].append(
                {
                    "split": split,
                    "task_index": task_index,
                    "task_id": task.task_id,
                    "candidate_tmax_scipy256": candidate,
                    "adam600_tmax_scipy256": verified["ADAM_600"],
                    "mma600_tmax_scipy256": verified["MMA_600"],
                    "strong_single_family": winner,
                    "strong_single_tmax_scipy256": reference,
                    "primary_relative_gap": (candidate - reference) / reference,
                    "field_r25_tmax_scipy256": verified["FIELD_UNET_BEST4_R25"],
                    "sens_best4_tmax_scipy256": verified["SENS_UNET_BEST4"],
                    "sens_r50_tmax_scipy256": verified["SENS_UNET_BEST4_R50"],
                }
            )

            if task_index < 8:
                candidates: list[tuple[float, int, NDArray[np.float64]]] = []
                for start in (0, 1, 2, 3):
                    destination = _baseline_destination(
                        output, "ADAM", split, task_index, start
                    )
                    result = _read_json(destination / "result.json")
                    score64 = float(result["snapshot_tmax_scipy64"]["600"])
                    design = _load_npz_design(destination / "designs.npz", "binary_600")
                    candidates.append((score64, start, design))
                _, selected_start, selected_design = min(
                    candidates, key=lambda item: (item[0], item[1])
                )
                selected = _verify_family(design=selected_design, task=task)
                multistart_rows.append(
                    {
                        "split": split,
                        "task_index": task_index,
                        "task_id": task.task_id,
                        "selected_adam_start": selected_start,
                        "adam_multistart_tmax_scipy256": selected["worst_peak"],
                        "sens_r25_tmax_scipy256": candidate,
                        "relative_gap_to_adam_multistart": (
                            candidate - float(selected["worst_peak"])
                        )
                        / float(selected["worst_peak"]),
                    }
                )
            print(f"MT3_FINAL_VERIFY {split} {task_index + 1}/{len(tasks)}", flush=True)

    _atomic_csv(output / "verification" / "all_scipy256_rows.csv", verification_rows)
    _atomic_csv(output / "verification" / "adam_multistart_rows.csv", multistart_rows)
    verdicts: dict[str, object] = {}
    for split, rows in split_gap_rows.items():
        _atomic_csv(output / "verification" / f"{split}_primary_rows.csv", rows)
        gaps = np.asarray(
            [row["primary_relative_gap"] for row in rows], dtype=np.float64
        )
        bootstrap = median_gap_bootstrap(gaps, split=split)  # type: ignore[arg-type]
        verdict = classify_mt3_test_split(
            gaps=gaps,
            invalid_count=0,
            exact_budget_count=len(rows),
            equivalent_evaluation_speedup=600
            / neural_equivalent_evaluations(refinement_updates=25),
            split=split,  # type: ignore[arg-type]
            bootstrap=bootstrap,
        )
        verdicts[split] = {
            "bootstrap": asdict(bootstrap),
            "verdict": asdict(verdict),
        }
    _atomic_json(output / "verification" / "final_verdicts.json", verdicts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("neural", "adam", "mma", "verify"), required=True
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    splits = _open_splits(
        arguments.authorization.resolve(), arguments.authorization_sha256
    )
    output = arguments.output.resolve()
    _write_task_manifest(output, splits)
    if arguments.phase == "verify":
        run_verification_phase(splits=splits, output=output)
        return
    if not torch.cuda.is_available():
        raise SystemExit("MT3 final neural/baseline phases require CUDA")
    device = torch.device("cuda")
    if arguments.phase == "neural":
        run_neural_phase(
            splits=splits,
            training_root=arguments.training_root.resolve(),
            output=output,
            device=device,
        )
    else:
        run_baseline_phase(
            splits=splits,
            output=output,
            device=device,
            method=arguments.phase,
        )


if __name__ == "__main__":
    main()
