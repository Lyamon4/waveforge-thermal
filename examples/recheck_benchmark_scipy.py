"""Preserve warm algorithm timings; recheck every timed output with SciPy LU."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from waveforge.lab.mcl.factorization import factorization_backend
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.ribbed_evaluation import RibbedEvaluator
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init


def main(args):
    worker_init()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [task_from_dict(t) for t in json.loads(args.tasks.read_text())["tasks"]]
    rows = json.loads((args.benchmark / "measurements.json").read_text())
    summary = json.loads((args.benchmark / "summary.json").read_text())
    maximum = 0.0
    for row in rows:
        index = row["task_index"]
        with np.load(args.benchmark / f"task_{index:03d}/{row['method']}.npz") as data:
            binary = data["binary"]
        started = time.perf_counter()
        with factorization_backend("scipy"):
            evaluator = RibbedEvaluator(
                tasks[index],
                args.output / f"case_{index:03d}_{row['method']}",
                forward_backend="reference",
            )
            actual = evaluator.evaluate(binary)
        if actual["valid"] != row["selected"]["valid"]:
            raise RuntimeError("independent admission changed")
        for grid in ("64", "128"):
            for mode in ("nominal", "power"):
                error = abs(
                    actual["modes"][grid][mode]["silicon_peak_c"]
                    - row["selected"]["modes"][grid][mode]["silicon_peak_c"]
                )
                maximum = max(maximum, error)
        row["selected"] = actual
        row["independent_final_audit_seconds"] = time.perf_counter() - started
        print(f"SCIPY BENCH {index} {row['method']}", flush=True)
    if maximum > 1e-8:
        raise RuntimeError("temperature agreement failed")
    summary.update(
        independent_factorization_backend="scipy_lu",
        max_backend_temperature_difference_k=maximum,
    )
    for pair in summary["pairs"]:
        group = [r for r in rows if r["task_index"] == pair["task_index"]]
        neural = next(r for r in group if r["method"] == "solver_free")
        for mode in ("nominal", "power"):
            reference = min(
                r["selected"]["modes"]["128"][mode]["silicon_peak_c"]
                for r in group
                if r["method"] != "solver_free"
            )
            actual = neural["selected"]["modes"]["128"][mode]["silicon_peak_c"]
            pair[mode + "_gain_percent"] = 100 * (reference - actual) / (reference - 25)
    write(args.output / "measurements.json", rows)
    write(args.output / "summary.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
