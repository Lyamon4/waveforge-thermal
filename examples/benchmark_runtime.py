"""Measure complete requests in fresh processes; preserve every physical gate.

Example: python examples/benchmark_runtime.py --output D:/runtime --tasks 0 8 12
Optional --classical-tasks 0 also measures matched Adam600 and MMA600 with the
same accelerated solver. Every timed request includes startup, loading and I/O.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def measure(args):
    args.output.mkdir(parents=True, exist_ok=False)
    weights = ROOT / "models/manifold_unet_ribbed.pt"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    records = []
    configurations = (
        ("original_multi25", "multi25", "reference", "scipy", "numpy"),
        ("accelerated_multi25", "multi25", "fast", args.sparse_backend, "auto"),
        ("accelerated_r0", "r0", "fast", args.sparse_backend, "auto"),
    )
    for task in args.tasks:
        for repeat in range(args.repeats):
            # Alternate ordering so warm hardware does not always favor one path.
            ordered = configurations if repeat % 2 == 0 else configurations[::-1]
            paired = {}
            for name, mode, evaluation, sparse, residual in ordered:
                folder = args.output / f"task_{task:02d}_repeat_{repeat}_{name}"
                command = [
                    sys.executable,
                    str(ROOT / "examples/evaluate_pretrained_ribbed.py"),
                    "--task",
                    str(task),
                    "--device",
                    args.device,
                    "--mode",
                    mode,
                    "--sparse-backend",
                    sparse,
                    "--evaluation-backend",
                    evaluation,
                    "--workers",
                    str(args.workers if evaluation == "fast" else 1),
                    "--output",
                    str(folder),
                ]
                row = run(command, folder, sparse, residual)
                row.update(task=task, repeat=repeat, method=name)
                records.append(row)
                paired[name] = row
                write(args.output / "measurements.json", records)
            original = paired["original_multi25"]
            optimized = paired["accelerated_multi25"]
            if original["binary_sha256"] != optimized["binary_sha256"]:
                raise RuntimeError("optimization changed the selected binary design")
            if original["gate"] != optimized["gate"]:
                raise RuntimeError("optimization changed physical admission")
            for mode in ("nominal_c", "equal_power_c"):
                if abs(original[mode] - optimized[mode]) >= 1e-9:
                    raise RuntimeError("optimization changed temperature above 1e-9 K")
    for task in args.classical_tasks:
        for kind in ("Adam", "MMA"):
            folder = args.output / f"task_{task:02d}_{kind}600"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                kind,
                "--task",
                str(task),
                "--output",
                str(folder),
            ]
            row = run(command, folder, args.sparse_backend, "auto")
            row.update(task=task, repeat=0, method=kind + "600")
            records.append(row)
            write(args.output / "measurements.json", records)
    if hashlib.sha256(weights.read_bytes()).hexdigest() != digest:
        raise RuntimeError("frozen model file changed")
    speedups = [
        old["wall_seconds"] / new["wall_seconds"]
        for old in records
        if old["method"] == "original_multi25"
        for new in records
        if new["method"] == "accelerated_multi25"
        and (old["task"], old["repeat"]) == (new["task"], new["repeat"])
    ]
    report = dict(
        timing_scope=(
            "fresh process: imports, conditioning, weights, inference, all checks, I/O"
        ),
        hardware="one machine, sequential requests; CPU threads pinned to 1",
        checkpoint_sha256=digest,
        preserved_binary_temperature_and_gates=True,
        median_same_pipeline_acceleration=median(speedups),
        taskwise_speedups=speedups,
        median_wall_seconds={
            name: median(r["wall_seconds"] for r in records if r["method"] == name)
            for name in sorted({r["method"] for r in records})
        },
        rows=records,
        note=(
            "R0 omits refinement and has different quality; "
            "never combine its time with MultiR25 quality"
        ),
    )
    write(args.output / "summary.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2),
        flush=True,
    )


def run(command, folder, sparse, residual):
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        WAVEFORGE_SPARSE_BACKEND=sparse,
        WAVEFORGE_RESIDUAL_BACKEND=residual,
    )
    started = time.perf_counter()
    with (folder / "stdout.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"request failed; see {folder / 'stdout.log'}")
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    selected = result["selected"]
    row = dict(
        wall_seconds=elapsed,
        output=str(folder),
        binary_sha256=selected["binary_sha256"],
        gate=selected["gate"],
        nominal_c=selected["modes"]["128"]["nominal"]["silicon_peak_c"],
        equal_power_c=selected["modes"]["128"]["power"]["silicon_peak_c"],
        objective_queries=result["objective_evaluations"],
    )
    print(f"{folder.name}: {elapsed:.3f}s", flush=True)
    return row


def worker(args):
    from waveforge.lab.mcl.ribbed_evaluation import matched_reference
    from waveforge.lab.mcl.stage3_training import task_from_dict

    tasks = json.loads(
        (ROOT / "results/stage55_20261029/inputs/evaluation_tasks.json").read_text()
    )["tasks"]
    matched_reference(
        task_from_dict(tasks[args.task]),
        args.output,
        kind=args.worker,
        initial_starts=args.reference_starts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", type=int, default=[0, 8, 12])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--classical-tasks", nargs="*", type=int, default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", choices=(1, 2, 3), type=int, default=3)
    parser.add_argument("--sparse-backend", choices=("scipy", "cudss"), default="cudss")
    parser.add_argument("--worker", choices=("Adam", "MMA"))
    parser.add_argument("--reference-starts", choices=(4, 8), type=int, default=4)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repetitions required")
    if args.worker:
        worker(args)
    else:
        measure(args)
