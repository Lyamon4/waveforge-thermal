"""Sealed Stage3.1 experiment; unchanged physics, no protected workload access."""

import argparse
import csv
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from waveforge.experiments.run_mcl_stage3 import METHODS, summary_row, verify_sources
from waveforge.lab.mcl.baselines import check_qualification, digest
from waveforge.lab.mcl.stage3_training import (
    NORMALIZATION,
    TrainingStream,
    condition_worker,
    protected_geometries,
    task_from_dict,
    worker_init,
    write_json,
)
from waveforge.lab.mcl.stage31_evaluation import CONTROL, R0, R25, evaluate_task
from waveforge.lab.mcl.stage31_training import CONFIG, train

ALL_METHODS = (*METHODS, CONTROL, R0, R25)


def write_comparison(output, rows):
    output = Path(output)
    write_json(output / "summary.json", rows)
    lookup = {(r["task"], r["method"]): r for r in rows}
    indices = sorted({r["task"] for r in rows})
    table = []
    lines = [
        "# Stage3.1 — OPEN development comparison",
        "",
        "Worst-case silicon temperature rise in K above inlet; lower is better.",
        "REJECTED: no admissible candidate, never treated as a winning low score.",
        "",
        "| Task | " + " | ".join(ALL_METHODS) + " |",
        "| --- | " + " | ".join("---:" for _ in ALL_METHODS) + " |",
    ]
    for i in indices:
        row = {"task": i}
        for method in ALL_METHODS:
            value = lookup.get((i, method))
            row[method] = value["peak_rise_k"] if value and value["accepted"] else None
        table.append(row)
        lines.append(
            f"| {i} | "
            + " | ".join(
                f"{row[m]:.6f}" if row[m] is not None else "REJECTED"
                for m in ALL_METHODS
            )
            + " |"
        )
    with (output / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", *ALL_METHODS])
        writer.writeheader()
        writer.writerows(table)
    metrics = {}
    for method in ALL_METHODS:
        if method == R25:
            continue
        gains = []
        for i in indices:
            base, new = lookup.get((i, method)), lookup.get((i, R25))
            if base and new and base["accepted"] and new["accepted"]:
                if base["peak_rise_k"] <= 0:
                    raise ValueError("nonpositive comparison denominator")
                gains.append(
                    100
                    * (base["peak_rise_k"] - new["peak_rise_k"])
                    / base["peak_rise_k"]
                )
        metrics[method] = dict(
            paired_tasks=len(gains),
            wins=sum(g > 0 for g in gains),
            reference_rejected=sum(
                not r["accepted"] for r in rows if r["method"] == method
            ),
            candidate_rejected=sum(
                not r["accepted"] for r in rows if r["method"] == R25
            ),
            reference_missing=sum((i, method) not in lookup for i in indices),
            candidate_missing=sum((i, R25) not in lookup for i in indices),
            median_paired_gain_percent=float(np.median(gains)) if gains else None,
            per_task_paired_gain_percent=gains,
        )
    write_json(output / "paired_metrics.json", metrics)
    lines += [
        "",
        "## Paired S31-R25 gains (positive = improvement)",
        "",
        "| Compared with | Valid pairs | Wins | Rejected base/R25 | Median gain % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method, metric in metrics.items():
        gain = metric["median_paired_gain_percent"]
        lines.append(
            f"| {method} | {metric['paired_tasks']} | {metric['wins']} | "
            f"{metric['reference_rejected']}/{metric['candidate_rejected']} | "
            + (f"{gain:.4f}" if gain is not None else "N/A")
            + " |"
        )
    lines += [
        "",
        "## Limits",
        "- STE is a biased gradient heuristic, not the derivative of the binary map.",
        "- The gradient-only score removes core priority; "
        "forward physics and decoder are unchanged.",
        "- Final model frozen before open-dev8 evaluation; "
        "no closed ID/OOD workload access.",
        "- R0 includes physical conditioning and four binary scoring solves, "
        "not zero compute.",
        "- All accepted outputs pass "
        "exact1024/connected/Re<500/residual/balance gates.",
        "- Reduced-order model, artificial additional sink "
        "and finite solid permeability remain.",
        "- Not a hardware result or a final held-out benchmark. "
        "No tuning after dev evaluation.",
    ]
    with (output / "SUMMARY_STAGE31.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(args):
    root = Path(__file__).resolve().parents[3]
    old = root / "artifacts/mcl_stage3_20261018/run_001"
    prior = json.loads((old / "manifest.json").read_text())
    receipt = json.loads((old / "completion.json").read_text())
    if not receipt.get("completed") or receipt.get("smoke_only"):
        raise ValueError("prior campaign incomplete")
    verify_sources(root, prior["source_sha256"])
    verify_sources(old, receipt["files"])
    qualification = root / "artifacts/mcl_stage12_20261017/run_002/qualification.json"
    check_qualification(qualification, root)
    if digest(qualification) != prior["qualification_sha256"]:
        raise ValueError("qualification seal changed")
    selection = Path(args.selection)
    if digest(selection) != prior["selection_sha256"]:
        raise ValueError("selection changed")
    selection_tasks = tuple(
        task_from_dict(t) for t in json.loads(selection.read_text())["tasks"]
    )
    tasks = selection_tasks[:8]
    if [t.fingerprint() for t in tasks] != prior["development_task_sha256"]:
        raise ValueError("development task order changed")
    protected, hygiene = protected_geometries(
        root / "artifacts/epyc_v2_20260923/splits", selection_tasks
    )
    if hygiene != prior["data_hygiene"]:
        raise ValueError("protected geometry seal changed")
    sources = dict(prior["source_sha256"])
    for p in (root / "src/waveforge/lab/mcl").glob("stage31*.py"):
        sources[str(p.relative_to(root))] = digest(p)
    for name in (
        "src/waveforge/experiments/run_mcl_stage31.py",
        "src/waveforge/lab/mcl/stage31_training.py",
    ):
        sources[name] = digest(root / name)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(CONFIG, steps=3 if args.smoke else 800, batch=1 if args.smoke else 4)
    manifest = dict(
        stage="3.1",
        created_utc=datetime.now(UTC).isoformat(),
        smoke_only=args.smoke,
        training=config,
        normalization=NORMALIZATION,
        source_sha256=sources,
        prior_manifest_sha256=digest(old / "manifest.json"),
        prior_completion_sha256=digest(old / "completion.json"),
        qualification_sha256=digest(qualification),
        selection_sha256=digest(selection),
        data_hygiene=hygiene,
        development_task_sha256=prior["development_task_sha256"],
        closed_test_accessed=False,
        workers=args.workers,
        backend="CPU PyTorch plus persistent qualified CPU sparse LU workers",
        prior_settings=prior["parameters"],
        gradient_override=(
            "remove smooth core priority ONLY in auxiliary score; "
            "beta_grad=beta_forward"
        ),
        forward=(
            "Stage3 core-consistent in phases0/1; unchanged binary Top1024 in phase2"
        ),
        training_boundaries_zero_based=[267, 533],
        training_alpha=[100, 250, 500],
        beta=[2, 4, 8],
        refinement_alpha=[150, 300, 500],
        branch_lr=[0.08, 0.05, 0.02],
        pixel_lr=[0.04, 0.03, 0.015],
        r25_boundaries_zero_based=[9, 17],
        control_boundaries_zero_based=[200, 400],
        r25_checkpoints=[0, 5, 10, 15, 20, 25],
        control_checkpoints=[0, 50, 100, 200, 400, 600],
        desaturation=(
            "first16/last4096 clamp to +/-atanh(.85); "
            "widths unchanged; ORIGINAL step0 retained"
        ),
        logit_l2=(
            "mean(raw branch increments**2) + mean(raw residual pixels**2), "
            "before decoder tanh"
        ),
        candidate_rules=(
            "same independent Stage1.2 SciPy and Stage2 admissibility gates"
        ),
        versions=dict(torch=torch.__version__, numpy=np.__version__),
    )
    write_json(output / "manifest.json", manifest)
    shutil.copyfile(old / "summary.json", output / "stage3_summary_original.json")
    shutil.copyfile(old / "comparison.csv", output / "stage3_comparison_original.csv")
    start = time.perf_counter()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=worker_init
        ) as pool:
            model, freeze = train(
                pool, output, protected, steps=config["steps"], batch=config["batch"]
            )
            verify_sources(root, sources)
            if digest(output / "model_frozen.pt") != freeze["model_sha256"]:
                raise RuntimeError("frozen model seal changed")
            if args.smoke:
                tasks = (TrainingStream(2026101809, protected).next(),)
                rows = []
            else:
                rows = json.loads((old / "summary.json").read_text())
            conditioning_start = time.perf_counter()
            conditioned = list(pool.map(condition_worker, tasks))
            with torch.inference_mode():
                parameters = (
                    model(torch.from_numpy(np.stack([c[0] for c in conditioned])))
                    .double()
                    .numpy()
                )
            write_json(
                output / "evaluation_conditioning.json",
                dict(
                    task_sha256=[t.fingerprint() for t in tasks],
                    diagnostics=[c[1] for c in conditioned],
                    conditioning_and_inference_seconds=time.perf_counter()
                    - conditioning_start,
                ),
            )
            print(
                "Model frozen; independent binary evaluation and matched control.",
                flush=True,
            )
            jobs = {
                pool.submit(
                    evaluate_task, t, p, output / f"task_{i:02d}", smoke=args.smoke
                ): i
                for i, (t, p) in enumerate(zip(tasks, parameters, strict=True))
            }
            for future in as_completed(jobs):
                index = jobs[future]
                reports = future.result()
                rows.extend(summary_row(index, r) for r in reports)
                print(
                    f"evaluation task {index}: "
                    + ", ".join(
                        r["method"]
                        + "="
                        + (
                            f"{r['selected']['peak_rise_k']:.3f}K"
                            if r["selected"]
                            else "REJECTED"
                        )
                        for r in reports
                    ),
                    flush=True,
                )
                write_json(output / "evaluation_progress.json", rows, exclusive=False)
        verify_sources(root, sources)
        verify_sources(old, receipt["files"])
        if digest(output / "model_frozen.pt") != freeze["model_sha256"]:
            raise RuntimeError("model changed during evaluation")
        write_comparison(output, rows)
        files = {
            str(p.relative_to(output)): digest(p)
            for p in output.rglob("*")
            if p.is_file()
            and p.name not in ("progress.json", "evaluation_progress.json")
        }
        write_json(
            output / "completion.json",
            dict(
                completed=True,
                smoke_only=args.smoke,
                elapsed_seconds=time.perf_counter() - start,
                closed_test_accessed=False,
                original_artifacts_unchanged=True,
                files=files,
            ),
        )
        print(f"Completed: {output}", flush=True)
    except BaseException as error:
        write_json(
            output / "failure.json",
            dict(
                type=type(error).__name__,
                error=str(error),
                elapsed_seconds=time.perf_counter() - start,
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--selection",
        default="artifacts/inputs/development_tasks.json",
    )
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
