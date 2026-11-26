"""Train once on disjoint tasks, freeze, evaluate dev8; never open closed workloads."""

import argparse
import csv
import json
import os
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
import torch

from waveforge.lab.mcl.baselines import check_qualification, digest
from waveforge.lab.mcl.stage3_evaluation import evaluate_task
from waveforge.lab.mcl.stage3_training import (
    NORMALIZATION,
    TrainingStream,
    condition_worker,
    protected_geometries,
    task_from_dict,
    train_model,
    worker_init,
    write_json,
)

METHODS = (
    "Straight",
    "Uniform-Parallel",
    "Pixel-Adam",
    "Pixel-MMA",
    "Vascular-Adam",
    "Vascular-MMA",
    "Vascular-Adam600-Halo",
    "Vascular-UNet-R0",
    "Vascular-UNet-R25",
)


def verify_sources(root, hashes):
    for name, expected in hashes.items():
        if digest(root / name) != expected:
            raise ValueError(f"source changed: {name}")


def summary_row(index, result):
    selected = result["selected"]
    return {
        "task": index,
        "method": result["method"],
        "accepted": selected is not None,
        "peak_rise_k": selected["peak_rise_k"] if selected else None,
        "diagnostic_final_peak_rise_k": None,
        "selected_step": selected["step"] if selected else None,
        "flow_ml_s": selected["flow_ml_s"] if selected else None,
        "max_local_re": selected["diagnostics"]["max_local_reynolds_estimate"]
        if selected
        else None,
        "air_heat_fraction_max": max(selected["diagnostics"]["air_heat_fraction"])
        if selected
        else None,
        "elapsed_seconds": result["elapsed_seconds"],
    }


def write_summary(output, rows, task_count):
    rows = sorted(rows, key=lambda row: (row["task"], row["method"]))
    write_json(output / "summary.json", rows)
    with (output / "summary.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(row["task"], row["method"]): row for row in rows}
    table = []
    lines = [
        "# MCL Stage 3: OPEN development comparison",
        "",
        "Worst-scenario silicon temperature rise (K); LOWER is better. "
        "REJECTED means no admissible selected candidate, not a missing low score.",
        "All accepted candidates: exactly 1024 cells, all cells port-connected, "
        "Re<500 and unchanged Stage 1.2 numerical/conservation gates.",
        "",
        "| Task | " + " | ".join(METHODS) + " |",
        "| --- | " + " | ".join("---:" for _ in METHODS) + " |",
    ]
    for index in range(task_count):
        row = {"task": index}
        for method in METHODS:
            result = lookup.get((index, method))
            row[method] = (
                result["peak_rise_k"] if result and result["accepted"] else None
            )
        table.append(row)
        cells = [
            f"{row[method]:.3f}" if row[method] is not None else "REJECTED"
            for method in METHODS
        ]
        lines.append(f"| {index} | " + " | ".join(cells) + " |")
    with (output / "comparison.csv").open(
        "x", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", *METHODS])
        writer.writeheader()
        writer.writerows(table)
    metrics = {}
    for method in METHODS:
        valid = [row for row in rows if row["method"] == method and row["accepted"]]
        pairs = [
            (
                row["peak_rise_k"],
                lookup[(row["task"], "Uniform-Parallel")]["peak_rise_k"],
            )
            for row in valid
            if (row["task"], "Uniform-Parallel") in lookup
        ]
        metrics[method] = {
            "admissible_tasks": len(valid),
            "mean_peak_rise_k_on_valid": float(
                np.mean([r["peak_rise_k"] for r in valid])
            )
            if valid
            else None,
            "wins_vs_uniform": sum(a < b for a, b in pairs),
            "paired_tasks_vs_uniform": len(pairs),
        }
    lines += [
        "",
        "Original Stage 2 CSV is preserved verbatim as stage2_summary_original.csv.",
        "Stage 2 Adam/MMA retain their original 600-step/evaluation budgets. "
        "Adam600-Halo uses the original single start, optimizer and continuation, "
        "but the same halo/core wrapper as R25.",
        "R0 compares all four frozen-network heads by independent SciPy solves; "
        "R25 refines only the best admissible R0 head, retaining its step-0 incumbent.",
        "",
        "## Limits",
        "- Halo bounds positive residuals, but is not a universal topology proof. "
        "The unchanged strict gate guarantees connectivity of ACCEPTED results.",
        "- Continuous density has core=1 and mass=1024, not exactly 1024 binary cells. "
        "A continuous-to-binary gap remains; core membership is nondifferentiable.",
        "- R0 includes baseline physical conditioning and four binary evaluations; "
        "it means zero per-task gradient updates, not zero solver calls.",
        "- Development-only, reduced-order simulation. No closed ID/OOD workloads "
        "opened, no hardware validation, no claim of benchmark generalization.",
        "- Face velocities are superficial model velocities in m/s; qx/qy are m3/s.",
    ]
    (output / "SUMMARY_STAGE3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "metrics.json", metrics)


def run(args):
    root = Path(__file__).resolve().parents[3]
    output = Path(args.output).resolve()
    qualification = root / "artifacts/mcl_stage12_20261017/run_002/qualification.json"
    stage2 = root / "artifacts/mcl_stage2_20261017/run_001"
    qualified = check_qualification(qualification, root)
    prior = json.loads((stage2 / "manifest.json").read_text(encoding="utf-8"))
    verify_sources(root, prior["source_sha256"])
    if digest(qualification) != prior["qualification_sha256"]:
        raise ValueError("Stage 2 qualification seal mismatch")
    selection = Path(args.selection).resolve()
    if digest(selection) != prior["selection_sha256"]:
        raise ValueError("Stage 2 selection source mismatch")
    selection_data = json.loads(selection.read_text(encoding="utf-8"))
    selection_tasks = tuple(task_from_dict(row) for row in selection_data["tasks"])
    dev = tuple(selection_tasks[i] for i in prior["task_indices"])
    if [task.fingerprint() for task in dev] != prior["tasks"]:
        raise ValueError("development task identity/order changed")
    protected, hygiene = protected_geometries(
        root / "artifacts/epyc_v2_20260923/splits", selection_tasks
    )
    sources = {
        **qualified["source_sha256"],
        **prior["source_sha256"],
        **{
            str(p.relative_to(root)): digest(p)
            for p in (root / "src/waveforge/lab/mcl").glob("*.py")
        },
    }
    for relative in (
        "src/waveforge/experiments/run_mcl_stage3.py",
        "src/waveforge/ml/mt3_unet.py",
        "src/waveforge/lab/epyc_tasks.py",
        "src/waveforge/lab/contracts.py",
    ):
        sources[relative] = digest(root / relative)
    config = {
        "steps": 2 if args.smoke else 400,
        "batch": 2 if args.smoke else 4,
        "learning_rate": 0.0003,
        "softmin_temperature": 0.05,
        "mean_head_weight": 0.1,
        "diversity_weight": 0.01,
        "gradient_clip": 1.0,
        "train_seed": 2026101801,
        "model_seed": 2026101802,
    }
    manifest = {
        "stage": "3",
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke_only": args.smoke,
        "training": config,
        "normalization": NORMALIZATION,
        "data_hygiene": hygiene,
        "development_task_sha256": prior["tasks"],
        "source_sha256": sources,
        "qualification_sha256": digest(qualification),
        "stage2_manifest_sha256": digest(stage2 / "manifest.json"),
        "stage2_summary_sha256": digest(stage2 / "summary.csv"),
        "selection_sha256": digest(selection),
        "closed_test_accessed": False,
        "workers": args.workers,
        "backend": "CPU PyTorch network and geometry; persistent CPU sparse LU workers",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "parameters": {
            "anchors": [
                [13, 25, 38, 50],
                [10, 16, 32, 49],
                [13, 31, 46, 52],
                [12, 28, 35, 51],
            ],
            "halo_radius": 1.45,
            "halo_slope": 6.0,
            "pixel_u_bound": "tanh",
            "core_radius": 0.75,
            "projection_bisections": 80,
            "training_phases": "floor(3*zero_based_step/steps): beta 2,4,8; "
            "alpha 100,250,500; binary penalty 0,.01,.02; TV .001",
            "r25": {
                "steps": 25,
                "lr": 0.05,
                "beta": 8.0,
                "alpha": 500.0,
                "checkpoints": [0, 5, 10, 15, 20, 25],
            },
            "control": {
                "steps": 600,
                "lr": 0.05,
                "phases": "Stage2 unchanged",
                "checkpoints": [0, 50, 100, 200, 400, 600],
            },
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    shutil.copyfile(stage2 / "summary.csv", output / "stage2_summary_original.csv")
    start = time.perf_counter()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=worker_init
        ) as pool:
            model, training = train_model(pool, output, protected, config)
            verify_sources(root, sources)
            freeze_sha = digest(output / "model_frozen.pt")
            if freeze_sha != training["model_sha256"]:
                raise RuntimeError("model freeze seal mismatch")
            print("Model frozen. Starting independent binary evaluation.", flush=True)
            if args.smoke:
                # Smoke evaluation is on a fresh TRAIN-distribution task, not dev8.
                smoke_stream = TrainingStream(2026101809, protected)
                tasks = (smoke_stream.next(),)
                rows = []
            else:
                tasks = dev
                rows = json.loads((stage2 / "summary.json").read_text(encoding="utf-8"))
            conditioning = list(pool.map(condition_worker, tasks))
            with torch.inference_mode():
                parameters = (
                    model(torch.from_numpy(np.stack([r[0] for r in conditioning])))
                    .double()
                    .numpy()
                )
            write_json(
                output / "evaluation_conditioning.json",
                [
                    dict(task_sha256=t.fingerprint(), diagnostics=c[1])
                    for t, c in zip(tasks, conditioning, strict=True)
                ],
            )
            jobs = {
                pool.submit(
                    evaluate_task,
                    task,
                    params,
                    output / f"task_{index:02d}",
                    r25_steps=1 if args.smoke else 25,
                    control_steps=2 if args.smoke else 600,
                ): index
                for index, (task, params) in enumerate(
                    zip(tasks, parameters, strict=True)
                )
            }
            for future in as_completed(jobs):
                index = jobs[future]
                reports = future.result()
                rows.extend(summary_row(index, result) for result in reports)
                print(
                    f"evaluation task {index}: "
                    + ", ".join(
                        f"{r['method']}="
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
        if digest(output / "model_frozen.pt") != freeze_sha:
            raise RuntimeError("frozen model changed during evaluation")
        if digest(stage2 / "summary.csv") != manifest["stage2_summary_sha256"]:
            raise RuntimeError("Stage 2 results changed")
        write_summary(output, rows, len(tasks))
        receipts = {
            str(p.relative_to(output)): digest(p)
            for p in output.rglob("*")
            if p.is_file()
            and p.name not in ("progress.json", "evaluation_progress.json")
        }
        write_json(
            output / "completion.json",
            {
                "completed": True,
                "smoke_only": args.smoke,
                "closed_test_accessed": False,
                "frozen_sources_unchanged": True,
                "frozen_model_unchanged": True,
                "elapsed_seconds": time.perf_counter() - start,
                "files": receipts,
            },
        )
        print(f"Completed: {output}", flush=True)
    except BaseException as error:
        write_json(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--selection",
        default=(
            "artifacts/inputs/development_tasks.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
