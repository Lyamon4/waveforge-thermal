"""Offline labels for raw-input generation and candidate selection."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .amortized import fresh_task_groups
from .amortized_policy import load_policy
from .baselines import digest, uniform_parallel
from .fast_channels import decode_one
from .locked_evaluation import ranking, write
from .ribbed_channels import RibbedGeometry
from .ribbed_evaluation import RibbedEvaluator
from .solver_free import raw_features
from .stage3_training import protected_geometries, task_from_dict, worker_init
from .stage5_physics import stage5_spec
from .stage5_training import condition_worker


def physical_labels(row):
    """Both grids and pump regimes retain their original admission semantics."""
    nominal = row["modes"]["128"]["nominal"]["silicon_peak_c"]
    power = row["modes"]["128"]["power"]["silicon_peak_c"]
    checks = [row["modes"][str(g)][m] for g in (64, 128) for m in ("nominal", "power")]
    reynolds = max(r["diagnostics"]["max_local_reynolds_estimate"] for r in checks)
    water = max(r["diagnostics"]["peak_open_water_temperature_c"] for r in checks)
    return np.array(
        ((nominal - 25) / 50, (power - 25) / 50, reynolds / 1200, (water - 25) / 35),
        np.float32,
    )


def label_candidates(task, q, output, *, known=None):
    evaluator = RibbedEvaluator(task, output)
    if known is not None:
        known_rows, known_masks = known
        for row, mask in zip(known_rows, known_masks, strict=True):
            evaluator.cache[row["binary_sha256"]] = row
            evaluator.designs[row["binary_sha256"]] = mask.copy()
    geometry = RibbedGeometry.from_task(evaluator.frame.task)
    binaries = [
        decode_one(geometry, parameters, h % 4) for h, parameters in enumerate(q)
    ]
    rows = [evaluator.evaluate(d) for d in binaries]
    uniform = evaluator.evaluate(uniform_parallel())
    if not uniform["valid"]:
        raise RuntimeError("task fails the unchanged reference admission")
    rows.append(uniform)
    best = min(range(9), key=lambda i: ranking(rows[i]))
    return (
        dict(
            metrics=np.stack([physical_labels(r) for r in rows]),
            valid=np.array([r["valid"] for r in rows]),
            peaks=np.array([r["peak_c"] for r in rows]),
            binaries=np.stack(binaries),
            best_binary=uniform_parallel() if best == 8 else binaries[best],
        ),
        rows,
        best,
    )


def prepare_dataset(
    root,
    teacher,
    registry,
    *,
    train_count=256,
    validation_count=24,
    seed=2026103011,
    previous=None,
):
    root, teacher = Path(root), Path(teacher)
    root.mkdir(parents=True, exist_ok=True)
    previous = Path(previous) if previous else None
    extra = []
    repo = Path(__file__).resolve().parents[4]
    opened = repo / "results/stage55_20261029/inputs/evaluation_tasks.json"
    extra.extend(task_from_dict(t) for t in json.loads(opened.read_text())["tasks"])
    if previous:
        for path in (
            previous / "run_001/tasks.json",
            previous / "fresh_audit/tasks.json",
        ):
            saved = json.loads(path.read_text())
            lists = saved.values() if isinstance(saved, dict) else [saved]
            for group in lists:
                if isinstance(group, list):
                    extra.extend(
                        task_from_dict(t)
                        for t in group
                        if isinstance(t, dict) and "chiplets" in t
                    )
    protected, hygiene = protected_geometries(registry, extra)
    config = dict(
        teacher_sha256=digest(teacher),
        seed=seed,
        train_tasks=train_count,
        validation_tasks=validation_count,
        physics=asdict(stage5_spec()),
        teacher_free=False,
        raw_channels=7,
        learned_proposals=8,
        analytic_fallback=True,
        online_physical_solves=0,
        quality_target=dict(
            median_gap_percent=0.5, p90_gap_percent=2.0, invalid_count=0
        ),
        geometry_hygiene=hygiene,
    )
    config = json.loads(json.dumps(config))
    if (root / "config.json").exists():
        if json.loads((root / "config.json").read_text()) != config:
            raise ValueError("output already has a different experiment")
        groups = {
            k: [task_from_dict(t) for t in v]
            for k, v in json.loads((root / "tasks.json").read_text()).items()
        }
    else:
        groups = fresh_task_groups(
            seed, protected, train_count=train_count, validation_count=validation_count
        )
        write(root / "config.json", config)
        write(
            root / "tasks.json",
            {k: [t.canonical() for t in v] for k, v in groups.items()},
        )
    return groups, config


def collect_dataset(root, teacher, groups, *, device="cuda", workers=2):
    worker_init()
    root = Path(root)
    data = root / "data"
    data.mkdir(exist_ok=True)
    model = load_policy(teacher, device)
    started = time.perf_counter()

    def collect(item):
        split, index, task = item
        path = data / f"{split}_{index:03d}"
        if path.with_suffix(".json").exists() and path.with_suffix(".npz").exists():
            meta = json.loads(path.with_suffix(".json").read_text())
            if meta["task_sha256"] != task.fingerprint():
                raise ValueError("cached task identity changed")
            return meta
        condition, counts = condition_worker(task)
        raw = raw_features(task)
        if not np.allclose(raw, condition[[0, 1, 2, 9, 10, 11, 12]], rtol=0, atol=1e-7):
            raise RuntimeError("raw inputs differ from task-only teacher channels")
        with torch.inference_mode():
            q = model(torch.from_numpy(condition[None]).to(device))[0].cpu().numpy()
        arrays, rows, best = label_candidates(task, q, root / "work" / path.stem)
        arrays.update(raw=raw, condition=condition, parameters=q)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path.with_suffix(".npz"))
        meta = dict(
            task_sha256=task.fingerprint(),
            reference=rows[best],
            reference_index=best,
            candidates=rows,
            conditioning=counts,
        )
        write(path.with_suffix(".json"), meta)
        return meta

    entries = [
        (split, i, t)
        for split in ("validation", "train")
        for i, t in enumerate(groups[split])
    ]
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(collect, item): item for item in entries}
        for future in as_completed(futures):
            future.result()
            completed += 1
            row = dict(
                phase="offline_teacher_labels",
                completed=completed,
                total=len(entries),
                elapsed_seconds=time.perf_counter() - started,
            )
            write(root / "progress.json", row)
            if completed % 8 == 0 or completed == len(entries):
                print(
                    f"LABELS {completed}/{len(entries)} "
                    f"elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    write(
        root / "data_complete.json",
        dict(
            tasks=completed,
            elapsed_seconds=time.perf_counter() - started,
            teacher_sha256=digest(teacher),
        ),
    )


def arrays(root, split, tasks):
    values = []
    for i, task in enumerate(tasks):
        path = Path(root) / "data" / f"{split}_{i:03d}"
        if (
            json.loads(path.with_suffix(".json").read_text())["task_sha256"]
            != task.fingerprint()
        ):
            raise ValueError("training entry identity mismatch")
        with np.load(path.with_suffix(".npz")) as saved:
            values.append({k: saved[k] for k in saved.files})
    return {k: np.stack([r[k] for r in values]) for k in values[0]}
