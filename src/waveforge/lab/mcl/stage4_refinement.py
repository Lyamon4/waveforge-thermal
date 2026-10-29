"""Frozen R0-start ablation, original operator; no training or test access."""

import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .contracts import problem_from_task, stage12_spec
from .differentiable import MCLSolver
from .stage3 import continuous
from .stage3_evaluation import record_candidate, select_valid
from .stage3_training import linear_report, physical_loss, worker_init, write_json
from .stage31 import Phase, density, desaturate, refinement_phase


def prepare(parameters, variant):
    if variant not in ("A", "B", "C"):
        raise ValueError("unknown refinement variant")
    return desaturate(parameters) if variant == "A" else parameters.detach().clone()


def schedule(variant, step):
    if variant not in ("A", "B", "C") or not 0 <= step < 25:
        raise ValueError("invalid variant/step")
    if variant != "C":
        return refinement_phase(step, 25)
    lr = 0.015 + (0.003 - 0.015) * step / 24
    return Phase(8.0, 500.0, lr, lr, False, 2)


def count_solves(rows):
    return {
        f"{p}_{kind}": sum(
            (1 if kind == "calls" else r["rhs"]) for r in rows if r["phase"] == p
        )
        for p in ("forward", "adjoint")
        for kind in ("calls", "rhs")
    }


def optimize_variant(task, initial, output, variant):
    worker_init()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    problem = problem_from_task(task, stage12_spec())
    solver = MCLSolver([problem])
    original = torch.tensor(initial, dtype=torch.float64)
    rows = [record_candidate(problem, original, output, 0)]
    prepared = prepare(original, variant)
    branches = prepared[:20].clone().requires_grad_(True)
    pixels = prepared[20:].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([{"params": [branches]}, {"params": [pixels]}])
    totals = count_solves([])
    history = []
    max_residual = 0.0
    for step in range(25):
        phase = schedule(variant, step)
        optimizer.param_groups[0]["lr"] = phase.branch_lr
        optimizer.param_groups[1]["lr"] = phase.pixel_lr
        optimizer.zero_grad(set_to_none=True)
        parameters = torch.cat((branches, pixels))
        if variant == "C":
            design = continuous(parameters[None], 8.0)[0]
            loss, result = physical_loss(design, solver, 2)
        else:
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
            raise RuntimeError("nonfinite refinement")
        counts = count_solves(solver.linear_diagnostics)
        for k in totals:
            totals[k] += counts[k]
        max_residual = max(max_residual, linear_report(solver)["max_linear_residual"])
        optimizer.step()
        with torch.no_grad():
            branches.clamp_(-6, 6)
            pixels.clamp_(-12, 12)
        history.append(
            dict(
                step=step + 1,
                loss=loss.item(),
                beta=phase.beta,
                alpha=phase.alpha,
                hard=phase.hard,
                branch_lr=phase.branch_lr,
                pixel_lr=phase.pixel_lr,
            )
        )
        if (step + 1) % 5 == 0:
            rows.append(
                record_candidate(
                    problem, torch.cat((branches, pixels)), output, step + 1
                )
            )
    selected = select_valid(rows)
    if selected is not None:
        shutil.copyfile(output / selected["file"], output / "selected.npz")
    report = dict(
        variant=variant,
        checkpoints=rows,
        selected=selected,
        history=history,
        initial_desaturation_max_change=float((prepared - original).abs().max()),
        solver_counts_objective=totals,
        binary_evaluations=len(rows),
        max_linear_residual=max_residual,
        elapsed_seconds=time.perf_counter() - start,
        objective_evaluations=25,
        original_step0_retained=True,
    )
    write_json(output / "result.json", report)
    return report
