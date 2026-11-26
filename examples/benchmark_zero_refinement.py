"""Complete cold requests: frozen R0 versus equally initialized 600-query controls."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from benchmark_runtime import ROOT, run, write


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    original_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    records = []
    candidate_count = None
    for repeat in range(args.repeats):
        folder = args.output / f"neural_{repeat}"
        command = [
            sys.executable,
            str(ROOT / "examples/evaluate_pretrained_ribbed.py"),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--task",
            str(args.task),
            "--device",
            "cuda",
            "--sparse-backend",
            "cudss",
            "--mode",
            "r0",
            "--output",
            str(folder),
        ]
        row = run(command, folder, "cudss", "auto")
        record = json.loads((folder / "result.json").read_text())
        count = record["initial_physics_scores"]
        candidate_count = count if candidate_count is None else candidate_count
        if count != candidate_count or row["objective_queries"] != 0:
            raise RuntimeError("neural candidate or refinement budget changed")
        row.update(method="R0", repeat=repeat)
        records.append(row)
    if candidate_count not in (4, 8):
        raise ValueError("matching reference supports four or eight initial starts")
    for kind in ("Adam", "MMA"):
        folder = args.output / (kind + "600")
        command = [
            sys.executable,
            str(ROOT / "examples/benchmark_runtime.py"),
            "--worker",
            kind,
            "--task",
            str(args.task),
            "--reference-starts",
            str(candidate_count),
            "--output",
            str(folder),
        ]
        row = run(command, folder, "cudss", "auto")
        row.update(method=kind + "600", repeat=0)
        records.append(row)
        write(args.output / "measurements.json", records)
    neural = [row for row in records if row["method"] == "R0"]
    if len({row["binary_sha256"] for row in neural}) != 1:
        raise RuntimeError("repeated neural requests produced different designs")
    seconds = statistics.median(row["wall_seconds"] for row in neural)
    comparison = [row for row in records if row["method"] != "R0"]
    quality = {}
    for key in ("nominal_c", "equal_power_c"):
        baseline = min(row[key] for row in comparison)
        quality[key] = 100 * (baseline - neural[0][key]) / (baseline - 25)
    report = dict(
        task=args.task,
        repeats=args.repeats,
        neural_candidate_count=candidate_count,
        classical_initial_candidates_per_method=candidate_count,
        neural_optimizer_queries=0,
        classical_optimizer_queries_per_method=600,
        scope=(
            "cold process, input preparation, model loading, "
            "every physical check and export"
        ),
        same_sparse_backend="cudss FP64 with unchanged compensated gates",
        neural_median_seconds=seconds,
        speedups={row["method"]: row["wall_seconds"] / seconds for row in comparison},
        gains_to_best_same_budget_control_percent=quality,
        rows=records,
        checkpoint_sha256=original_hash,
        limitations=(
            "one case; repeated cold neural requests; one run per classical method"
        ),
    )
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("frozen checkpoint changed")
    write(args.output / "summary.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    main(parser.parse_args())
