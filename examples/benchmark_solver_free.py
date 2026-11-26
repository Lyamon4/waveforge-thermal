"""Warm complete algorithms on identical tasks, with final checks timed separately."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from waveforge.lab.mcl.baselines import digest, uniform_parallel
from waveforge.lab.mcl.factorization import Factorization, factorization_backend
from waveforge.lab.mcl.locked_evaluation import ranking, write
from waveforge.lab.mcl.ribbed_evaluation import (
    RibbedEvaluator,
    analytic_reference_starts,
    optimize_physical,
)
from waveforge.lab.mcl.solver_free import create_session, load_solver_free
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init


def classical(task, folder, kind):
    started = time.perf_counter()
    evaluator = RibbedEvaluator(task, folder)
    uniform = evaluator.evaluate(uniform_parallel())
    starts = [
        (q, h, evaluator.parameters(q, h)) for q, h in analytic_reference_starts(8)
    ]
    q, h, _ = min(starts, key=lambda x: ranking(x[2]))
    trajectory = optimize_physical(
        evaluator, q, h, kind=kind, steps=600, checkpoint_every=50
    )
    selected = min((uniform, trajectory["raw"]), key=ranking)
    seconds = time.perf_counter() - started
    binary = evaluator.designs[selected["binary_sha256"]].copy()
    # I/O and a common independent final audit are outside both algorithm clocks.
    np.savez_compressed(folder / "selected.npz", binary=binary)
    return dict(
        method=kind + "600",
        seconds=seconds,
        objective_queries=trajectory["objective_evaluations"],
        objective_physics_solves=trajectory["physical_objective_solves"],
        initial_candidates=8,
        selected=selected,
    ), binary


def main(args):
    worker_init()
    os.environ["WAVEFORGE_SPARSE_BACKEND"] = "cudss"
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [task_from_dict(t) for t in json.loads(args.tasks.read_text())["tasks"]]
    sha = digest(args.checkpoint)
    model = load_solver_free(args.checkpoint, "cuda")
    session = create_session(model, device="cuda", cuda_graphs=True)
    # Warm the actual solver, adjoint, optimizer imports and neural graph first.
    for _ in range(5):
        session.infer(tasks[args.indices[0]])
    warm = RibbedEvaluator(tasks[args.indices[0]], args.output / "warmup")
    q, h = analytic_reference_starts(8)[0]
    for kind in ("Adam", "MMA"):
        optimize_physical(warm, q, h, kind=kind, steps=2, checkpoint_every=2)
    records = []
    for position, index in enumerate(args.indices):
        task = tasks[index]
        folder = args.output / f"task_{index:03d}"
        folder.mkdir()
        original = Factorization.__init__

        def forbidden(*unused, **kwargs):
            raise RuntimeError("physical solve entered neural generation")

        Factorization.__init__ = forbidden
        try:
            samples = [session.infer(task) for _ in range(30)]
        finally:
            Factorization.__init__ = original
        if any(not np.array_equal(samples[0]["binary"], r["binary"]) for r in samples):
            raise RuntimeError("non-deterministic generation")
        neural = dict(
            method="solver_free",
            seconds=float(np.median([r["total_ms"] for r in samples])) / 1000,
            p90_seconds=float(np.quantile([r["total_ms"] for r in samples], 0.9))
            / 1000,
            objective_queries=0,
            online_physical_solves=0,
            initial_candidates=8,
        )
        designs = [(neural, samples[0]["binary"])]
        for kind in ("Adam", "MMA") if position % 2 == 0 else ("MMA", "Adam"):
            row, binary = classical(task, folder / kind, kind)
            designs.append((row, binary))
            write(folder / f"{kind}.json", row)
            print(f"BENCH task={index} {kind}600={row['seconds']:.3f}s", flush=True)
        verifier = RibbedEvaluator(
            task, folder / "independent_scipy", forward_backend="reference"
        )
        for row, binary in designs:
            started = time.perf_counter()
            with factorization_backend("scipy"):
                checked = verifier.evaluate(binary)
            row.update(
                task_index=index,
                selected=checked,
                independent_final_audit_seconds=time.perf_counter() - started,
            )
            np.savez_compressed(folder / (row["method"] + ".npz"), binary=binary)
            records.append(row)
        write(args.output / "measurements.json", records)
    pairs = []
    for index in args.indices:
        rows = [r for r in records if r["task_index"] == index]
        neural = next(r for r in rows if r["method"] == "solver_free")
        classical_rows = [r for r in rows if r["method"] != "solver_free"]
        pair = dict(
            task_index=index,
            neural_ms=neural["seconds"] * 1000,
            speedups={
                r["method"]: r["seconds"] / neural["seconds"] for r in classical_rows
            },
            all_methods_valid=all(r["selected"]["valid"] for r in rows),
        )
        for mode in ("nominal", "power"):
            reference = min(
                r["selected"]["modes"]["128"][mode]["silicon_peak_c"]
                for r in classical_rows
            )
            actual = neural["selected"]["modes"]["128"][mode]["silicon_peak_c"]
            pair[mode + "_gain_percent"] = 100 * (reference - actual) / (reference - 25)
        pairs.append(pair)
    if digest(args.checkpoint) != sha:
        raise RuntimeError("frozen weights changed")
    summary = dict(
        checkpoint_sha256=sha,
        tasks=len(args.indices),
        task_indices=args.indices,
        pairs=pairs,
        scope=(
            "warm complete algorithm: raw task to one binary; imports, "
            "one-time loading and final independent audit excluded for all methods"
        ),
        classical_budget=(
            "eight fixed starts plus shared Uniform, best start then "
            "one 600-query trajectory; binary incumbent every 50 queries"
        ),
        neural_budget=(
            "eight decoded neural proposals plus shared Uniform; "
            "learned selection; zero online physical solves"
        ),
        same_physics=True,
        physics_verification="same independent SciPy 64/128 nominal/equal-power path",
        limitations=(
            "selected-by-index fresh cases, one trajectory per classical "
            "method per case; not a 32-case quality claim"
        ),
    )
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    main(p.parse_args())
