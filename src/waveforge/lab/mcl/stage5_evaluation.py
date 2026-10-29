"""Matched optimizers and independent binary evaluation in approved Stage5."""

import math
import shutil
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from .baselines import uniform_parallel
from .contracts import problem_from_task
from .stage3_training import linear_report, physical_loss, worker_init, write_json
from .stage4_refinement import count_solves
from .stage5_decoder import binary, continuous, density, initial_parameters
from .stage5_physics import Stage5Solver, stage5_spec
from .stage5_preflight import solve_profile
from .stage31 import refinement_phase
from .vascular import connectivity

PUMP_W = 0.11090112476467685
REGIMES = ("nominal", "flow15", "flow16.7", "power")


def pressure_for(nominal_flow, regime):
    if not np.isfinite(nominal_flow) or nominal_flow <= 0:
        raise ValueError("positive nominal flow required")
    if regime == "nominal":
        return 10000.0
    if regime in ("flow15", "flow16.7"):
        target = 15.0 if regime == "flow15" else 16.7
        return 10000 * target / nominal_flow
    if regime == "power":
        return math.sqrt(PUMP_W * 10000 / (nominal_flow * 1e-6))
    raise ValueError("unknown operating regime")


def evaluate(design, problem, *, pressure=None):
    d = np.asarray(design)
    if d.shape != (64, 64) or not np.all((d == 0) | (d == 1)) or d.sum() != 1024:
        raise ValueError("binary 1024-cell design required")
    result = solve_profile(d, problem, k_tim=40.0, pressure_drop_pa=pressure)
    diag = dict(result.diagnostics)
    # This reused forward evaluator cannot certify its separate training adjoint.
    # The run-level qualification.json supplies that certificate, not this flag.
    diag.pop("training_operator_qualified", None)
    residual = max(
        diag["pressure_residual"],
        *diag["thermal_residual"],
        diag["compensated_pressure_residual"],
        *diag["compensated_thermal_residual"],
    )
    conservation = max(diag["mass_relative_error"], *diag["energy_relative_error"])
    geom = connectivity(d)
    peak = float(result.scenario_peaks.max())
    accepted = (
        residual < 1e-12 and conservation < 1e-9 and geom["all_cells_port_connected"]
    )
    return dict(
        peak_rise_k=peak,
        silicon_peak_c=peak + 25,
        silicon_le95=peak + 25 <= 95,
        flow_ml_s=result.flow_ml_s,
        pressure_pa=pressure or problem.spec.pressure_drop_pa,
        accepted=bool(accepted),
        max_linear_residual=residual,
        max_balance_error=conservation,
        geometry=geom,
        diagnostics=diag,
    ), result


def with_fallback(raw, uniform):
    if not uniform["accepted"]:
        raise RuntimeError("common fallback failed numerical/geometric gate")
    use = not raw["accepted"] or raw["peak_rise_k"] >= uniform["peak_rise_k"]
    system = uniform if use else raw
    return dict(
        raw_peak_rise_k=raw["peak_rise_k"],
        raw_accepted=raw["accepted"],
        system_peak_rise_k=system["peak_rise_k"],
        fallback_used=bool(use),
        raw_gain_percent=100
        * (uniform["peak_rise_k"] - raw["peak_rise_k"])
        / uniform["peak_rise_k"],
        system_gain_percent=100
        * (uniform["peak_rise_k"] - system["peak_rise_k"])
        / uniform["peak_rise_k"],
    )


def record(problem, parameters, output, step, suffix=""):
    d = binary(parameters[None])[0].cpu().numpy()
    row, result = evaluate(d, problem)
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
    return dict(row, step=step, file=filename)


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
    uniform, _ = evaluate(uniform_parallel(), problem)
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
    valid = [r for r in checkpoints if r["accepted"]]
    if not valid:
        raise RuntimeError("no valid raw candidate")
    selected = min(valid, key=lambda r: (r["peak_rise_k"], r["step"]))
    shutil.copyfile(output / selected["file"], output / "selected.npz")
    report = dict(
        method=method,
        requested_evaluations=steps,
        evaluations=len(history),
        terminated_early=len(history) != steps,
        optimizer_stops=stops,
        selected=selected,
        uniform=uniform,
        fallback=with_fallback(selected, uniform),
        checkpoints=checkpoints,
        history=history,
        solver_counts=totals,
        independent_binary_evaluations=1 + len(checkpoints),
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
    valid = [r for r in heads if r["accepted"]]
    if not valid:
        raise RuntimeError("all neural heads invalid")
    best = min(valid, key=lambda r: (r["peak_rise_k"], r["step"]))
    shutil.copyfile(output / best["file"], output / "selected.npz")
    uniform, _ = evaluate(uniform_parallel(), problem)
    r0 = dict(
        method="Manifold-UNet-R0",
        selected=best,
        heads=heads,
        fallback=with_fallback(best, uniform),
        independent_binary_evaluations=5,
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


def audit_regimes(task, paths, output):
    """Frozen nominal-selected raw geometries; reselect shared fallback per regime."""
    worker_init()
    problem = problem_from_task(task, stage5_spec())
    designs = {"Uniform-Parallel": uniform_parallel()}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            designs[name] = data["binary"].copy()
    rows = []
    nominal = {name: evaluate(d, problem)[0] for name, d in designs.items()}
    for regime in REGIMES:
        records = {}
        for name, d in designs.items():
            dp = pressure_for(nominal[name]["flow_ml_s"], regime)
            rec = (
                nominal[name]
                if regime == "nominal"
                else evaluate(d, problem, pressure=dp)[0]
            )
            records[name] = dict(rec, method=name, regime=regime)
        for row in records.values():
            row["fallback"] = with_fallback(row, records["Uniform-Parallel"])
            rows.append(row)
    write_json(output, rows)
    return rows
