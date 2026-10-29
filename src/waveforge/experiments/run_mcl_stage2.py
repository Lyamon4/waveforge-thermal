"""First eight OPEN development tasks, gated MCL controls; no training or test data."""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import torch

from waveforge.lab.contracts import Chiplet, ChipletTask
from waveforge.lab.mcl import problem_from_task, straight_channel
from waveforge.lab.mcl.baselines import (
    check_qualification,
    digest,
    evaluate_binary,
    optimize,
    uniform_parallel,
)
from waveforge.lab.mcl.contracts import stage12_spec


def write_summary(output, rows):
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "method",
                "accepted",
                "peak_rise_k",
                "diagnostic_final_peak_rise_k",
                "selected_step",
                "flow_ml_s",
                "max_local_re",
                "air_heat_fraction_max",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["task"], r["method"])))
    (output / "summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )


def summary_row(index, method, record, elapsed):
    return dict(
        task=index,
        method=method,
        accepted=record["accepted"],
        peak_rise_k=record["peak_rise_k"] if record["accepted"] else None,
        diagnostic_final_peak_rise_k=record["peak_rise_k"]
        if not record["accepted"]
        else None,
        selected_step=record.get("step", 0) if record["accepted"] else None,
        flow_ml_s=record["flow_ml_s"],
        max_local_re=record["diagnostics"]["max_local_reynolds_estimate"],
        air_heat_fraction_max=max(record["diagnostics"]["air_heat_fraction"]),
        elapsed_seconds=elapsed,
    )


def mark_completed(folder, manifest_sha):
    paths = [folder / "result.json", *sorted(folder.glob("*.npz"))]
    receipt = dict(
        manifest_sha256=manifest_sha, files={p.name: digest(p) for p in paths}
    )
    with (folder / "completion.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)


def completed_method(folder, manifest_sha):
    if not folder.exists():
        return None
    if not (folder / "completion.json").is_file():
        raise ValueError(
            f"partial method preserved; explicit restart required: {folder}"
        )
    receipt = json.loads((folder / "completion.json").read_text(encoding="utf-8"))
    if receipt["manifest_sha256"] != manifest_sha:
        raise ValueError(f"method provenance mismatch: {folder}")
    if "result.json" not in receipt["files"]:
        raise ValueError("completion receipt has no result checksum")
    for name, expected in receipt["files"].items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError(f"method checksum mismatch: {folder}/{name}")
    return json.loads((folder / "result.json").read_text(encoding="utf-8"))


def worker(problem, method, index, output, steps, manifest_sha):
    torch.set_num_threads(1)
    result = optimize(problem, method, index, output, steps=steps)
    mark_completed(output, manifest_sha)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=8, choices=range(1, 9))
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    torch.set_num_threads(1)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    root = Path(__file__).resolve().parents[3]
    qualification = check_qualification(args.qualification, root)
    if digest(args.selection) != qualification["selection_sha256"]:
        raise ValueError("selection is not the qualified OPEN development source")
    tasks = [
        ChipletTask(
            tuple(Chiplet(**c) for c in t["chiplets"]),
            tuple(tuple(p) for p in t["powers_w"]),
        )
        for t in json.loads(args.selection.read_text(encoding="utf-8"))["tasks"][
            : args.tasks
        ]
    ]
    if len(tasks) != args.tasks:
        raise ValueError("insufficient open tasks")
    spec = stage12_spec()
    files = [
        *sorted((root / "src/waveforge/lab/mcl").glob("*.py")),
        Path(__file__),
        root / "src/waveforge/design/parameterization.py",
        root / "src/waveforge/experiments/run_mcl_stage2.py",
    ]
    manifest = dict(
        stage="2",
        tasks=[t.fingerprint() for t in tasks],
        task_indices=list(range(args.tasks)),
        qualification_sha256=digest(args.qualification),
        selection_sha256=digest(args.selection),
        spec=asdict(spec),
        closed_test_accessed=False,
        workers=args.workers,
        steps=600,
        backend="CPU sparse LU; two local processes, not CUDA factorization",
        source_sha256={str(p.relative_to(root)): digest(p) for p in files},
    )
    if args.output.exists():
        stored = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if stored != json.loads(json.dumps(manifest)):
            raise ValueError("campaign provenance mismatch; preserve old output")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    manifest_sha = digest(args.output / "manifest.json")
    rows = []
    started = time.perf_counter()
    errors = []
    problems = [problem_from_task(t, spec) for t in tasks]
    for i, problem in enumerate(problems):
        for name, d in [
            ("Straight", straight_channel(spec)),
            ("Uniform-Parallel", uniform_parallel()),
        ]:
            then = time.perf_counter()
            folder = args.output / f"task_{i:02d}_{name}"
            previous = completed_method(folder, manifest_sha)
            if previous is not None:
                rows.append(summary_row(i, name, previous, previous["elapsed_seconds"]))
                continue
            record, result = evaluate_binary(problem, d)
            record["elapsed_seconds"] = time.perf_counter() - then
            folder.mkdir()
            (folder / "result.json").write_text(
                json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
            )
            import numpy as np

            np.savez_compressed(
                folder / "design.npz",
                binary=d,
                temperature=result.temperature,
                pressure=result.pressure,
            )
            rows.append(summary_row(i, name, record, time.perf_counter() - then))
            mark_completed(folder, manifest_sha)
            print(
                f"{i}/{name}: peak {record['peak_rise_k']:.3f} K, "
                f"accepted {record['accepted']}",
                flush=True,
            )
    write_summary(args.output, rows)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = []
        for i, problem in enumerate(problems):
            for method in ("Pixel-Adam", "Pixel-MMA", "Vascular-Adam", "Vascular-MMA"):
                folder = args.output / f"task_{i:02d}_{method}"
                previous = completed_method(folder, manifest_sha)
                if previous is not None:
                    rows.append(
                        summary_row(
                            i,
                            method,
                            previous["selected"] or previous["final"],
                            previous["elapsed_seconds"],
                        )
                    )
                else:
                    pending.append((problem, method, i, folder, 600, manifest_sha))
        futures = {pool.submit(worker, *job): (job[2], job[1]) for job in pending}
        for future in as_completed(futures):
            i, method = futures[future]
            try:
                result = future.result()
                record = result["selected"] or result["final"]
                rows.append(summary_row(i, method, record, result["elapsed_seconds"]))
                print(
                    f"COMPLETE {i}/{method}: peak {record['peak_rise_k']:.4f} K, "
                    f"accepted {record['accepted']}",
                    flush=True,
                )
            except Exception as exc:
                errors.append(dict(task=i, method=method, error=repr(exc)))
                print(f"FAILED {i}/{method}: {exc!r}", flush=True)
            write_summary(args.output, rows)
            (args.output / "status.json").write_text(
                json.dumps(
                    dict(
                        elapsed_seconds=time.perf_counter() - started,
                        completed=len(rows),
                        expected=args.tasks * 6,
                        errors=errors,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
    write_summary(args.output, rows)
    (args.output / "status.json").write_text(
        json.dumps(
            dict(
                completed=len(rows),
                expected=args.tasks * 6,
                errors=errors,
                finished=True,
                elapsed_seconds=time.perf_counter() - started,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
