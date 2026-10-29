"""Strict S31 binary comparison; original R0 incumbent survives desaturation."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import initial_parameters
from .contracts import problem_from_task, stage12_spec
from .differentiable import MCLSolver
from .stage3_evaluation import record_candidate, select_valid
from .stage3_training import linear_report, worker_init, write_json
from .stage31 import density, desaturate, refinement_phase

R0 = "Vascular-UNet-S31-R0"
R25 = "Vascular-UNet-S31-R25"
CONTROL = "Vascular-Adam600-Decoupled"


def optimize(task, initial, output, *, budget, steps=None):
    steps = budget if steps is None else steps
    if budget not in (25, 600) or not 1 <= steps <= budget:
        raise ValueError("invalid optimizer step count")
    worker_init()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    problem = problem_from_task(task, stage12_spec())
    solver = MCLSolver([problem])
    original = torch.tensor(initial, dtype=torch.float64)
    rows = [record_candidate(problem, original, output, 0)]
    prepared = desaturate(original)
    branches = prepared[:20].clone().requires_grad_(True)
    pixels = prepared[20:].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([{"params": [branches]}, {"params": [pixels]}])
    checkpoints = (
        {5, 10, 15, 20, 25} if budget == 25 else {50, 100, 200, 400, 600}
    ) | {steps}
    max_residual = 0.0
    with (output / "history.jsonl").open("x", encoding="utf-8") as history:
        for step in range(steps):
            phase = refinement_phase(step, budget)
            optimizer.param_groups[0]["lr"] = phase.branch_lr
            optimizer.param_groups[1]["lr"] = phase.pixel_lr
            optimizer.zero_grad(set_to_none=True)
            parameters = torch.cat((branches, pixels))
            design, _ = density(parameters[None], beta=phase.beta, hard=phase.hard)
            result = solver(design)
            active = result.temperature[0, :, 0, solver.chip[0]].flatten() / 100
            thermal = (
                torch.logsumexp(phase.alpha * active, 0) - np.log(active.numel())
            ) / phase.alpha
            d = design[0]
            tv = (d[1:] - d[:-1]).abs().mean() + (d[:, 1:] - d[:, :-1]).abs().mean()
            loss = (
                thermal
                + 0.001 * tv
                + (0.0, 0.01, 0.02)[phase.level] * (d * (1 - d)).mean()
            )
            loss.backward()
            if not torch.isfinite(loss) or not all(
                torch.isfinite(v.grad).all() for v in (branches, pixels)
            ):
                raise RuntimeError("nonfinite refinement gradient/loss")
            max_residual = max(
                max_residual, linear_report(solver)["max_linear_residual"]
            )
            optimizer.step()
            with torch.no_grad():
                branches.clamp_(-6, 6)
                pixels.clamp_(-12, 12)
            row = dict(
                step=step + 1,
                phase=phase.level,
                hard_forward=phase.hard,
                beta=phase.beta,
                alpha=phase.alpha,
                branch_lr=phase.branch_lr,
                pixel_lr=phase.pixel_lr,
                loss=loss.item(),
                forward_peak_rise_k=result.scenario_peaks.detach().max().item(),
                max_linear_residual=max_residual,
            )
            history.write(json.dumps(row, allow_nan=False) + "\n")
            if step + 1 in checkpoints:
                rows.append(
                    record_candidate(
                        problem, torch.cat((branches, pixels)), output, step + 1
                    )
                )
            if (step + 1) % 50 == 0:
                history.flush()
                write_json(output / "progress.json", row, exclusive=False)
    selected = select_valid(rows)
    if selected:
        shutil.copyfile(output / selected["file"], output / "selected.npz")
    report = dict(
        method=R25 if budget == 25 else CONTROL,
        steps=steps,
        budget=budget,
        checkpoints=rows,
        selected=selected,
        original_step0_retained=True,
        initial_desaturation_max_change=float((prepared - original).abs().max()),
        objective_evaluations=steps,
        binary_evaluations=len(rows),
        max_linear_residual=max_residual,
        elapsed_seconds=time.perf_counter() - start,
    )
    write_json(output / "result.json", report)
    return report


def evaluate_task(task, parameters, output, *, smoke=False):
    worker_init()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    problem = problem_from_task(task, stage12_spec())
    folder = output / R0
    folder.mkdir()
    start = time.perf_counter()
    rows = [
        record_candidate(
            problem, torch.tensor(p, dtype=torch.float64), folder, 0, head=h
        )
        for h, p in enumerate(parameters)
    ]
    selected = select_valid(rows)
    r0 = dict(
        method=R0,
        checkpoints=rows,
        selected=selected,
        elapsed_seconds=time.perf_counter() - start,
    )
    if selected:
        shutil.copyfile(folder / selected["file"], folder / "selected.npz")
    write_json(folder / "result.json", r0)
    if selected:
        r25 = optimize(
            task,
            parameters[selected["head"]],
            output / R25,
            budget=25,
            steps=2 if smoke else 25,
        )
        if r25["checkpoints"][0]["binary_sha256"] != selected["binary_sha256"]:
            raise RuntimeError("R25 step 0 differs from selected R0")
        if r25["selected"]["peak_rise_k"] > selected["peak_rise_k"] + 1e-10:
            raise RuntimeError("best-valid R25 regressed below its R0 incumbent")
        r25["source_head"] = selected["head"]
        r25["initial_r0_sha256"] = selected["binary_sha256"]
        write_json(output / R25 / "result.json", r25, exclusive=False)
    else:
        (output / R25).mkdir()
        r25 = dict(
            method=R25,
            selected=None,
            steps=0,
            checkpoints=[],
            elapsed_seconds=0.0,
            reason="No admissible R0 head; no fallback or topology repair",
        )
        write_json(output / R25 / "result.json", r25)
    control = optimize(
        task,
        initial_parameters("vascular", 0),
        output / CONTROL,
        budget=600,
        steps=2 if smoke else 600,
    )
    return [r0, r25, control]
