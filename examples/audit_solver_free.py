"""Freeze raw inference first, then independently audit its one chosen design."""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from waveforge.lab.mcl.adaptive_manifold import reflected_task
from waveforge.lab.mcl.amortized_policy import load_policy
from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.factorization import Factorization, factorization_backend
from waveforge.lab.mcl.locked_channels import canonical_frame
from waveforge.lab.mcl.locked_evaluation import ranking, write
from waveforge.lab.mcl.ribbed_evaluation import RibbedEvaluator
from waveforge.lab.mcl.solver_free import (
    create_session,
    load_solver_free,
    raw_features,
)
from waveforge.lab.mcl.solver_free_data import label_candidates
from waveforge.lab.mcl.solver_free_training import quality_summary
from waveforge.lab.mcl.stage3_training import (
    TrainingStream,
    protected_geometries,
    task_from_dict,
    worker_init,
)
from waveforge.lab.mcl.stage5_training import condition_worker

ROOT = Path(__file__).resolve().parents[1]


def audit_record(index, result, row, reference_row, oracle_row):
    choice = result["proposal_index"]
    record = dict(
        task_index=index,
        proposal_index=choice,
        valid=row["valid"],
        uniform_fallback=choice == 8,
        latency_ms=float(np.median(result["latency_samples_ms"])),
    )
    for mode in ("nominal", "power"):
        actual, reference, best = (
            r["modes"]["128"][mode]["silicon_peak_c"]
            for r in (row, reference_row, oracle_row)
        )
        record.update(
            {
                f"{mode}_c": actual,
                f"{mode}_reference_c": reference,
                f"{mode}_oracle_c": best,
                f"{mode}_gap_percent": 100 * (actual - reference) / (reference - 25),
                f"{mode}_selection_regret_percent": 100 * (actual - best) / (best - 25),
            }
        )
    return record


def sample_audit(args, sha):
    known = []
    for path in (
        args.run / "tasks.json",
        args.previous / "run_001/tasks.json",
        args.previous / "fresh_audit/tasks.json",
        ROOT / "results/stage55_20261029/inputs/evaluation_tasks.json",
        *args.exclude,
    ):
        saved = json.loads(path.read_text())
        groups = saved.values() if isinstance(saved, dict) else [saved]
        for group in groups:
            if isinstance(group, list):
                known.extend(
                    task_from_dict(t)
                    for t in group
                    if isinstance(t, dict) and "chiplets" in t
                )
    protected, hygiene = protected_geometries(args.registry, known)
    stream = TrainingStream(
        args.seed, protected + tuple(reflected_task(t) for t in protected)
    )
    tasks = []
    while len(tasks) < args.count:
        task = canonical_frame(stream.next()).task
        if not stream.protected.contains(task):
            tasks.append(task)
    write(
        args.output / "frozen_before_sampling.json",
        dict(
            model_sha256=sha,
            seed=args.seed,
            tasks=args.count,
            geometry_hygiene=hygiene,
            official_test_workloads_opened=False,
            used_for_training=False,
        ),
    )
    write(args.output / "tasks.json", dict(tasks=[t.canonical() for t in tasks]))
    return tasks


def main(args):
    worker_init()
    os.environ["WAVEFORGE_SPARSE_BACKEND"] = "cudss"
    args.output.mkdir(parents=True, exist_ok=args.resume)
    sha = digest(args.checkpoint)
    model = load_solver_free(args.checkpoint, args.device)
    source = load_policy(args.teacher, args.device)
    if args.resume:
        frozen = json.loads((args.output / "frozen_before_sampling.json").read_text())
        if (
            frozen["model_sha256"] != sha
            or frozen["seed"] != args.seed
            or frozen["tasks"] != args.count
        ):
            raise ValueError("resume must use exactly the same frozen audit")
        tasks = [
            task_from_dict(t)
            for t in json.loads((args.output / "tasks.json").read_text())["tasks"]
        ]
    else:
        tasks = sample_audit(args, sha)
    session = create_session(model, device=args.device, cuda_graphs=args.cuda_graphs)
    original = Factorization.__init__

    def forbidden(*unused, **kwargs):
        raise RuntimeError("physical solve entered the timed raw inference path")

    Factorization.__init__ = forbidden
    results = []
    try:
        for _ in range(5):
            session.infer(tasks[0])
        for task in tasks:
            repeats = [session.infer(task) for _ in range(3)]
            if any(
                not np.array_equal(r["binary"], repeats[0]["binary"]) for r in repeats
            ):
                raise RuntimeError("non-deterministic online result")
            selected = repeats[-1]
            selected["latency_samples_ms"] = [r["total_ms"] for r in repeats]
            results.append(selected)
    finally:
        Factorization.__init__ = original
    timing = [t for r in results for t in r["latency_samples_ms"]]
    if args.resume and not (args.output / "online_timing_original.json").exists():
        (args.output / "online_timing_original.json").write_bytes(
            (args.output / "online_timing.json").read_bytes()
        )
    write(
        args.output / "online_timing.json",
        dict(
            median_ms=float(np.median(timing)),
            p90_ms=float(np.quantile(timing, 0.9)),
            tasks=len(tasks),
            repeats_per_task=3,
            physical_factorizations_forbidden=True,
            online_physical_solves=0,
            scope="loaded raw-task to one binary design, no exports or physical checks",
        ),
    )

    def audit(index):
        task = tasks[index]
        folder = args.output / f"case_{index:03d}"
        if args.resume and (folder / "result.json").exists():
            saved = json.loads((folder / "result.json").read_text())
            if saved["frozen_model_sha256"] != sha:
                raise RuntimeError("cached audit model changed")
            with np.load(folder / "selected.npz") as selection:
                if not np.array_equal(selection["binary"], results[index]["binary"]):
                    raise RuntimeError("optimized inference changed the audited design")
            return audit_record(
                index,
                results[index],
                saved["selected"],
                saved["reference"],
                saved["oracle"],
            )
        folder.mkdir(exist_ok=True)
        condition, _ = condition_worker(task)
        raw = raw_features(task)
        with torch.inference_mode():
            teacher_q = (
                source(torch.from_numpy(condition[None]).to(args.device))[0]
                .cpu()
                .numpy()
            )
            student_q = results[index].get("_all_parameters")
            if student_q is None:
                student_q = (
                    model(torch.from_numpy(raw[None]).to(args.device))["parameters"][0]
                    .cpu()
                    .numpy()
                )
        teacher, teacher_rows, teacher_best = label_candidates(
            task, teacher_q, folder / "teacher"
        )
        teacher_masks = np.concatenate(
            (teacher["binaries"], teacher["best_binary"][None])
        )
        # The ninth known design must be the shared uniform, regardless of winner.
        from waveforge.lab.mcl.baselines import uniform_parallel

        teacher_masks[-1] = uniform_parallel()
        student, student_rows, _ = label_candidates(
            task, student_q, folder / "student", known=(teacher_rows, teacher_masks)
        )
        result = results[index]
        choice = 8 if result["uniform_fallback"] else result["proposal_index"]
        binary = uniform_parallel() if choice == 8 else student["binaries"][choice]
        if not np.array_equal(binary, result["binary"]):
            raise RuntimeError("auditor evaluated a different online design")
        row = student_rows[choice]
        oracle = min(range(9), key=lambda j: ranking(student_rows[j]))
        # Both reported finalists use the original independent SciPy path.
        verifier = RibbedEvaluator(
            task, folder / "independent_scipy", forward_backend="reference"
        )
        with factorization_backend("scipy"):
            checked = verifier.evaluate(binary)
            reference = verifier.evaluate(teacher_masks[teacher_best])
        for accelerated, independent in (
            (row, checked),
            (teacher_rows[teacher_best], reference),
        ):
            if accelerated["valid"] != independent["valid"]:
                raise RuntimeError("independent admission disagrees")
            for grid in ("64", "128"):
                for mode in ("nominal", "power"):
                    a = accelerated["modes"][grid][mode]["silicon_peak_c"]
                    b = independent["modes"][grid][mode]["silicon_peak_c"]
                    if abs(a - b) > 1e-8:
                        raise RuntimeError("independent temperature disagrees")
        row = checked
        np.savez_compressed(
            folder / "selected.npz",
            binary=binary,
            parameters=result["parameters"],
            head=result["head"],
            reference_binary=teacher_masks[teacher_best],
        )
        write(
            folder / "result.json",
            dict(
                selected=row,
                reference=reference,
                oracle=student_rows[oracle],
                proposal_index=choice,
                frozen_model_sha256=sha,
                online_physical_solves=0,
                independent_offline_physics=True,
                independent_factorization_backend="scipy_lu",
            ),
        )
        return audit_record(index, result, row, reference, student_rows[oracle])

    rows = [None] * len(tasks)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(audit, i): i for i in range(len(tasks))}
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            rows[index] = future.result()
            write(
                args.output / "progress.json",
                dict(
                    completed=completed,
                    total=len(tasks),
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
            print(f"FRESH AUDIT {completed}/{len(tasks)}", flush=True)
    with (args.output / "comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        tasks=len(tasks),
        model_sha256=sha,
        online_physical_solves=0,
        timing=json.loads((args.output / "online_timing.json").read_text()),
        uniform_fallbacks=sum(r["uniform_fallback"] for r in rows),
    )
    for mode in ("nominal", "power"):
        summary[mode] = quality_summary(
            np.array([r[f"{mode}_c"] for r in rows]),
            np.array([r["valid"] for r in rows]),
            np.array([r[f"{mode}_reference_c"] for r in rows]),
        )
        summary[mode]["median_selection_regret_percent"] = float(
            np.median([r[f"{mode}_selection_regret_percent"] for r in rows])
        )
    if digest(args.checkpoint) != sha:
        raise RuntimeError("frozen policy changed during fresh audit")
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument(
        "--teacher", type=Path, default=ROOT / "models/manifold_unet_amortized_r0.pt"
    )
    p.add_argument(
        "--previous", type=Path, default=ROOT.parent / "WaveForge_AMORTIZED_20261030"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--cuda-graphs", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--exclude", type=Path, nargs="*", default=[])
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--seed", type=int, default=202610303901)
    main(p.parse_args())
