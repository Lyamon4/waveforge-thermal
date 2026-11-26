"""Recheck exactly the frozen selected/reference masks with SciPy LU only."""

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from audit_solver_free import audit_record

from waveforge.lab.mcl.amortized_policy import load_policy
from waveforge.lab.mcl.baselines import digest, uniform_parallel
from waveforge.lab.mcl.factorization import factorization_backend
from waveforge.lab.mcl.fast_channels import decode_many
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.ribbed_channels import RibbedGeometry
from waveforge.lab.mcl.ribbed_evaluation import RibbedEvaluator
from waveforge.lab.mcl.solver_free_training import quality_summary
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init
from waveforge.lab.mcl.stage5_training import condition_worker


def main(args):
    worker_init()
    os.environ["WAVEFORGE_SPARSE_BACKEND"] = "cudss"
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [
        task_from_dict(t)
        for t in json.loads((args.audit / "tasks.json").read_text())["tasks"]
    ]
    previous = json.loads((args.audit / "summary.json").read_text())
    source = load_policy(args.teacher, "cuda")
    rows = []

    def check(index):
        task = tasks[index]
        folder = args.output / f"case_{index:03d}"
        folder.mkdir()
        saved = json.loads((args.audit / f"case_{index:03d}/result.json").read_text())
        with np.load(args.audit / f"case_{index:03d}/selected.npz") as data:
            selected = data["binary"]
        condition, _ = condition_worker(task)
        with torch.inference_mode():
            q = source(torch.from_numpy(condition[None]).cuda())[0].cpu().numpy()
        proposals = np.concatenate(
            (
                decode_many(RibbedGeometry.from_task(task), q, list(range(4)) * 2),
                uniform_parallel()[None],
            )
        )
        matched = [
            d
            for d in proposals
            if hashlib.sha256(d.tobytes()).hexdigest()
            == saved["reference"]["binary_sha256"]
        ]
        if not matched:
            raise RuntimeError(
                "cannot reproduce the exact previously selected reference"
            )
        reference_binary = matched[0]
        with factorization_backend("scipy"):
            evaluator = RibbedEvaluator(task, folder, forward_backend="reference")
            actual = evaluator.evaluate(selected)
            reference = evaluator.evaluate(reference_binary)
        errors = []
        for new, old in ((actual, saved["selected"]), (reference, saved["reference"])):
            if new["valid"] != old["valid"]:
                raise RuntimeError("SciPy admission differs")
            for grid in ("64", "128"):
                for mode in ("nominal", "power"):
                    errors.append(
                        abs(
                            new["modes"][grid][mode]["silicon_peak_c"]
                            - old["modes"][grid][mode]["silicon_peak_c"]
                        )
                    )
        if max(errors) > 1e-8:
            raise RuntimeError("SciPy differs beyond agreed numerical tolerance")
        write(
            folder / "result.json",
            dict(
                selected=actual,
                reference=reference,
                frozen_model_sha256=saved["frozen_model_sha256"],
                proposal_index=saved["proposal_index"],
                independent_factorization_backend="scipy_lu",
                max_backend_temperature_difference_k=max(errors),
            ),
        )
        np.savez_compressed(
            folder / "selected.npz", binary=selected, reference_binary=reference_binary
        )
        return audit_record(
            index,
            dict(proposal_index=saved["proposal_index"], latency_samples_ms=[0]),
            actual,
            reference,
            saved["oracle"],
        ), max(errors)

    values = [None] * len(tasks)
    maximum = 0.0
    with ThreadPoolExecutor(2) as pool:
        futures = {pool.submit(check, i): i for i in range(len(tasks))}
        for done, future in enumerate(as_completed(futures), 1):
            row, error = future.result()
            values[futures[future]] = row
            maximum = max(maximum, error)
            print(f"SCIPY ONLY {done}/{len(tasks)}", flush=True)
    with (args.audit / "comparison.csv").open(encoding="utf-8-sig") as handle:
        latency = {
            int(r["task_index"]): float(r["latency_ms"]) for r in csv.DictReader(handle)
        }
    rows = [dict(r, latency_ms=latency[r["task_index"]]) for r in values]
    with (args.output / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        tasks=len(tasks),
        model_sha256=previous["model_sha256"],
        timing=previous["timing"],
        independent_factorization_backend="scipy_lu",
        max_backend_temperature_difference_k=maximum,
        physics_unchanged=True,
        source_audit=str(args.audit.resolve()),
        source_teacher_sha256=digest(args.teacher),
        original_test_workloads_opened=False,
    )
    for mode in ("nominal", "power"):
        summary[mode] = quality_summary(
            np.array([r[mode + "_c"] for r in rows]),
            np.array([r["valid"] for r in rows]),
            np.array([r[mode + "_reference_c"] for r in rows]),
        )
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--teacher", type=Path, default=Path("models/manifold_unet_amortized_r0.pt")
    )
    main(p.parse_args())
