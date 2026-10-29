"""Stage5.1 matched optimizers: original schedules, shared loss, two-grid admission."""

import shutil
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from waveforge.experiments.audit_mcl_stage5_admission import record as evaluate_binary

from .contracts import problem_from_task
from .high_fidelity_eval import refine_problem
from .stage3_training import linear_report, worker_init, write_json
from .stage4_refinement import count_solves
from .stage5_decoder import binary, continuous, density, initial_parameters
from .stage5_physics import Stage5Solver, stage5_spec
from .stage31 import refinement_phase
from .stage51 import admissible_both, choose, physical_loss


def record(problem, parameters, output, step, suffix=""):
    start = time.perf_counter()
    d = binary(parameters[None])[0].detach().cpu().numpy()
    coarse, result = evaluate_binary(d, problem)
    fine_problem, fine_design = refine_problem(problem, d, 2)
    fine, _ = evaluate_binary(fine_design, fine_problem)
    filename = f"checkpoint_{step:04d}{suffix}.npz"
    np.savez_compressed(
        output / filename,
        parameters=parameters.detach().cpu().numpy(),
        binary=d,
        temperature=result.temperature,
        pressure=result.pressure,
        qx=result.qx,
        qy=result.qy,
    )
    return dict(
        coarse=coarse,
        fine=fine,
        step=step,
        file=filename,
        elapsed_seconds=time.perf_counter() - start,
    )


def optimize(task, method, output, *, steps=600, initial=None):
    """Same decoder and step0 Uniform fallback. MMA budget counts callbacks."""
    worker_init()
    if method not in ("Manifold-Adam", "Manifold-MMA", "Manifold-UNet-R25-C"):
        raise ValueError("unknown method")
    refine = method.endswith("R25-C")
    if refine and (steps != 25 or initial is None):
        raise ValueError("R25-C requires neural R0 and25 steps")
    if not 1 <= steps <= 600:
        raise ValueError("budget outside1..600")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    problem = problem_from_task(task, stage5_spec())
    solver = Stage5Solver([problem])
    p = torch.tensor(
        initial_parameters() if initial is None else initial,
        dtype=torch.float64,
        requires_grad=True,
    )
    checkpoints = [record(problem, p, output, 0)]
    history = []
    totals = count_solves([])
    stops = []
    wanted = {0, steps} | ({5, 10, 15, 20} if refine else {50, 100, 200, 300, 400, 500})

    def calculate(v, index):
        level = min(2, index * 3 // steps)
        if refine:
            design = continuous(v[None], 8.0)[0]
            loss, result = physical_loss(design, solver, 2)
        else:
            phase = refinement_phase(min(599, level * 200), 600)
            design, _ = density(v[None], beta=phase.beta, hard=phase.hard)
            loss, result = physical_loss(design[0], solver, level)
        gradient = torch.autograd.grad(loss, v)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("nonfinite optimizer gradient")
        for k, val in count_solves(solver.linear_diagnostics).items():
            totals[k] += val
        report = linear_report(solver)
        history.append(
            dict(
                evaluation=len(history) + 1,
                loss=loss.item(),
                phase=level,
                peak_rise_k=result.scenario_peaks.max().item(),
                **report,
            )
        )
        if len(history) % 25 == 0:
            write_json(
                output / "progress.json",
                dict(
                    method=method,
                    evaluations=len(history),
                    elapsed_seconds=time.perf_counter() - start,
                ),
                exclusive=False,
            )
        return loss.item(), gradient.detach()

    if method != "Manifold-MMA":
        opt = torch.optim.Adam([p])
        for index in range(steps):
            lr = (
                (0.015 - 0.012 * index / 24)
                if refine
                else (0.05, 0.03, 0.015)[min(2, index * 3 // steps)]
            )
            opt.param_groups[0]["lr"] = lr
            opt.zero_grad(set_to_none=True)
            _, p.grad = calculate(p, index)
            opt.step()
            with torch.no_grad():
                p[:25].clamp_(-6, 6)
                p[25:].clamp_(-12, 12)
            if index + 1 in wanted:
                checkpoints.append(record(problem, p, output, index + 1))
    else:
        import nlopt

        current = p.detach().numpy().copy()
        bound = np.full(4121, 12.0)
        bound[:25] = 6.0
        boundaries = sorted({0, steps // 3, 2 * steps // 3, steps})
        for first, last in pairwise(boundaries):

            def callback(x, g, first=first):
                v = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                loss, grad = calculate(v, first)
                if g.size:
                    g[:] = grad.numpy()
                if len(history) in wanted and len(history) != steps:
                    checkpoints.append(
                        record(problem, v, output, len(history), "_evaluated")
                    )
                return loss

            opt = nlopt.opt(nlopt.LD_MMA, 4121)
            opt.set_lower_bounds(-bound)
            opt.set_upper_bounds(bound)
            opt.set_min_objective(callback)
            opt.set_maxeval(last - first)
            current = opt.optimize(current)
            stops.append(
                dict(code=opt.last_optimize_result(), evaluations=len(history))
            )
            if len(history) != last:
                break
        p = torch.tensor(current, dtype=torch.float64)
        checkpoints.append(record(problem, p, output, len(history), "_returned"))
    selected = choose(checkpoints)
    shutil.copyfile(output / selected["file"], output / "selected.npz")
    report = dict(
        method=method,
        requested_evaluations=steps,
        evaluations=len(history),
        terminated_early=len(history) != steps,
        optimizer_stops=stops,
        selected=selected,
        selected_admissible_both=admissible_both(selected),
        checkpoints=checkpoints,
        history=history,
        solver_counts=totals,
        independent_binary_evaluations=2 * len(checkpoints),
        selection_forward_calls=4 * len(checkpoints),
        selection_forward_rhs=8 * len(checkpoints),
        elapsed_seconds=time.perf_counter() - start,
        exact_hard_gradient=False,
        old_results_modified=False,
    )
    write_json(output / "result.json", report)
    return report


def neural_task(task, parameters, output):
    worker_init()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    problem = problem_from_task(task, stage5_spec())
    heads = [
        record(problem, torch.tensor(p, dtype=torch.float64), output, i)
        for i, p in enumerate(parameters)
    ]
    best = choose(heads)
    shutil.copyfile(output / best["file"], output / "selected.npz")
    r0 = dict(
        method="Manifold-UNet-R0",
        selected=best,
        heads=heads,
        selected_admissible_both=admissible_both(best),
        independent_binary_evaluations=8,
        selection_forward_calls=16,
        selection_forward_rhs=32,
        elapsed_seconds=time.perf_counter() - start,
    )
    write_json(output / "result.json", r0)
    r25 = optimize(
        task,
        "Manifold-UNet-R25-C",
        output / "R25-C",
        steps=25,
        initial=parameters[best["step"]],
    )
    return r0, r25
