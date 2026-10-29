"""Sequential CPU R0/R25 timing and controlled frozen-start ablations."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from waveforge.experiments.run_mcl_stage4 import write_json, write_tables
from waveforge.lab.mcl.baselines import initial_parameters
from waveforge.lab.mcl.contracts import problem_from_task, stage12_spec
from waveforge.lab.mcl.stage3 import VascularUNet
from waveforge.lab.mcl.stage3_evaluation import record_candidate, select_valid
from waveforge.lab.mcl.stage3_training import condition_worker, worker_init
from waveforge.lab.mcl.stage4_inventory import load_inventory, seal_files, verify_seal
from waveforge.lab.mcl.stage4_refinement import optimize_variant
from waveforge.lab.mcl.stage31_evaluation import optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-control-task", type=int, default=0, choices=range(8))
    args = parser.parse_args()
    worker_init()
    root = Path(__file__).resolve().parents[3]
    out = Path(args.output).resolve()
    tasks, entries, seals = load_inventory(
        root,
        Path("artifacts/inputs/development_tasks.json"),
    )
    run = root / "artifacts/mcl_stage3_20261018/run_002"
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        dict(
            inputs=seals,
            source_sha256=seal_files(
                [Path(__file__), root / "src/waveforge/lab/mcl/stage4_refinement.py"]
            ),
            closed_test_accessed=False,
            training=False,
            execution="sequential one PyTorch CPU thread",
            measured_control_task=args.timing_control_task,
            limits=(
                "single-run timings, no confidence interval; "
                "training was on different execution resources"
            ),
        ),
    )
    model = VascularUNet()
    model.load_state_dict(
        torch.load(run / "model_frozen.pt", map_location="cpu", weights_only=True)[
            "model"
        ]
    )
    model.eval()
    model.requires_grad_(False)
    # Warm up neural kernels only, never a closed/evaluation physics task.
    with torch.inference_mode():
        model(torch.zeros(1, 13, 64, 64))
    rows = []
    timings = []
    for index, task in enumerate(tasks):
        folder = out / f"task_{index:02d}"
        folder.mkdir()
        timing_start = time.perf_counter()
        condition, condition_report = condition_worker(task)
        conditioning_seconds = time.perf_counter() - timing_start
        infer_start = time.perf_counter()
        with torch.inference_mode():
            pred = model(torch.tensor(condition[None]))[0].double()
        inference_seconds = time.perf_counter() - infer_start
        scoring_start = time.perf_counter()
        (folder / "R0_timing").mkdir()
        heads = [
            record_candidate(
                problem_from_task(task, stage12_spec()),
                p,
                folder / "R0_timing",
                0,
                head=h,
            )
            for h, p in enumerate(pred)
        ]
        r0time = time.perf_counter() - timing_start
        head_seconds = time.perf_counter() - scoring_start
        original = next(
            e
            for e in entries
            if e["task"] == index and e["method"] == "Vascular-UNet-S31-R0"
        )
        with np.load(original["path"], allow_pickle=False) as saved:
            initial = saved["parameters"].copy()
        selected = select_valid(heads)
        timing = dict(
            task=index,
            conditioning_seconds=conditioning_seconds,
            inference_seconds=inference_seconds,
            head_scoring_seconds=head_seconds,
            r0_end_to_end_seconds=r0time,
            condition_report=condition_report,
            cpu_selected_matches_frozen=selected is not None
            and selected["binary_sha256"] == original["binary_sha256"],
            condition_channels=13,
            condition_forward_calls=2,
            condition_forward_rhs=4,
            condition_adjoint_calls=6,
            condition_adjoint_rhs=12,
            head_forward_calls=8,
            head_forward_rhs=16,
        )
        write_json(
            folder / "R0_timing/result.json",
            dict(**timing, heads=heads, selected=selected),
        )
        for variant in ("A", "B", "C"):
            result = optimize_variant(task, initial, folder / f"R25_{variant}", variant)
            if result["checkpoints"][0]["binary_sha256"] != original["binary_sha256"]:
                raise RuntimeError("ablation changed original R0 start")
            peak = result["selected"]["peak_rise_k"]
            baseline = original["original_peak_rise_k"]
            row = dict(
                task=index,
                variant=variant,
                peak_rise_k=peak,
                initial_peak_rise_k=baseline,
                gain_vs_r0_percent=100 * (baseline - peak) / baseline,
                selected_step=result["selected"]["step"],
                refinement_seconds=result["elapsed_seconds"],
                end_to_end_seconds=r0time + result["elapsed_seconds"],
                objective_evaluations=result["objective_evaluations"],
                binary_evaluations=result["binary_evaluations"],
                solver_counts_objective=result["solver_counts_objective"],
            )
            if variant == "A":
                old = next(
                    e
                    for e in entries
                    if e["task"] == index and e["method"] == "Vascular-UNet-S31-R25"
                )
                row["replays_old_selected_binary"] = (
                    result["selected"]["binary_sha256"] == old["binary_sha256"]
                )
                row["replay_peak_delta_k"] = peak - old["original_peak_rise_k"]
                if abs(row["replay_peak_delta_k"]) > 1e-8:
                    raise RuntimeError("A replay mismatch")
            rows.append(row)
            print(
                f"task{index:02d} R25-{variant}: "
                f"gain={row['gain_vs_r0_percent']:.6f}% "
                f"time={row['refinement_seconds']:.2f}s",
                flush=True,
            )
        timings.append(timing)
    # A fresh control is timed ONLY, never replaces a frozen comparison design.
    i = args.timing_control_task
    control = optimize(
        tasks[i],
        initial_parameters("vascular", 0),
        out / f"Adam600_timing_task{i:02d}",
        budget=600,
    )
    write_tables(out, rows)
    training = json.loads((run / "freeze.json").read_text(encoding="utf-8"))[
        "elapsed_seconds"
    ]
    control_seconds = control["elapsed_seconds"]
    break_even = []
    for variant in ("R0", "A", "B", "C"):
        duration = (
            np.median([t["r0_end_to_end_seconds"] for t in timings])
            if variant == "R0"
            else np.median(
                [r["end_to_end_seconds"] for r in rows if r["variant"] == variant]
            )
        )
        saving = control_seconds - duration
        break_even.append(
            dict(
                variant=variant,
                median_seconds=float(duration),
                speed_ratio=control_seconds / float(duration),
                observed_hardware_cost_amortization_tasks=math.ceil(training / saving)
                if saving > 0
                else None,
                caveat=(
                    "runtime-only estimate, not equal-quality "
                    "or same-hardware training break-even"
                ),
            )
        )
    write_json(
        out / "timing.json",
        dict(
            tasks=timings,
            control_seconds=control_seconds,
            control_task=i,
            control_objectives=600,
            control_binary_evaluations=control["binary_evaluations"],
            training_seconds_observed=training,
            break_even=break_even,
            cpu_selected_matches_frozen=all(
                t["cpu_selected_matches_frozen"] for t in timings
            ),
        ),
    )
    verify_seal(seals)
    write_json(
        out / "completion.json",
        dict(completed=True, inputs_unchanged=True, closed_test_accessed=False),
    )


if __name__ == "__main__":
    main()
