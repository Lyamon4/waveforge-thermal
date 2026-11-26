"""Raw task to one binary design. No pressure/thermal solver is called."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from waveforge.lab.mcl.factorization import Factorization
from waveforge.lab.mcl.solver_free import create_session, load_solver_free
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    worker_init()

    # A physical solve is a hard error in this executable, including future edits.
    def forbidden(*unused, **kwargs):
        raise RuntimeError("physical factorization is forbidden in online inference")

    Factorization.__init__ = forbidden
    started = time.perf_counter()
    model = load_solver_free(args.checkpoint, args.device)
    session = create_session(model, device=args.device, cuda_graphs=args.cuda_graphs)
    source = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = (
        source["tasks"] if isinstance(source, dict) and "tasks" in source else source
    )
    if isinstance(tasks, dict) and "chiplets" in tasks:
        tasks = [tasks]
    task = task_from_dict(tasks[args.task])
    for _ in range(args.warmup):
        session.infer(task)
    warmed = time.perf_counter()
    results = [session.infer(task) for _ in range(args.repeats)]
    if len({hashlib.sha256(r["binary"].tobytes()).hexdigest() for r in results}) != 1:
        raise RuntimeError("repeated requests produced different designs")
    last = results[-1]
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "selected.npz",
        binary=last["binary"],
        fine_binary=np.repeat(np.repeat(last["binary"], 2, 0), 2, 1),
        parameters=last["parameters"],
        head=last["head"],
    )
    report = dict(
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        task=task.canonical(),
        proposal_index=last["proposal_index"],
        uniform_fallback=last["uniform_fallback"],
        physics_checked=False,
        online_physical_solves=0,
        candidate_count=8,
        cuda_graphs=args.cuda_graphs,
        decoded_designs=last.get(
            "decoded_proposals", 0 if last["uniform_fallback"] else 1
        ),
        median_ms=float(np.median([r["total_ms"] for r in results])),
        p90_ms=float(np.quantile([r["total_ms"] for r in results], 0.9)),
        repeats=args.repeats,
        warmup=args.warmup,
        startup_and_warmup_after_imports_seconds=warmed - started,
        scope=(
            "loaded service: raw task, generation, candidate rasterization "
            "and learned selection; physical solves and export excluded"
        ),
        timing_ms={
            k: [r[k] for r in results]
            for k in (
                "total_ms",
                "preparation_ms",
                "neural_ms",
                "decoder_ms",
                "selector_ms",
            )
            if k in last
        },
    )
    (args.output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("task", "timing_ms")},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "results/stage55_20261029/inputs/evaluation_tasks.json",
    )
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cuda-graphs", action="store_true")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        p.error("positive repetitions and nonnegative warmup required")
    main(args)
