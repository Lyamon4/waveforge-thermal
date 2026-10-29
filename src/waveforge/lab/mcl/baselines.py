"""Matched Stage2 controls, strict binary eligibility and reproducible histories."""

import hashlib
import json
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from waveforge.design.parameterization import filter_logits, project_volume

from .candidate import candidate_status
from .contracts import stage12_spec
from .differentiable import MCLSolver
from .reference import solve_reference
from .vascular import binary_topk, connectivity
from .vascular_stage12 import limited_waypoints, reserve_core, smooth_scores


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_qualification(path, root):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("stage") != "1.2" or not report.get("verdict", {}).get(
        "baseline_launch_allowed"
    ):
        raise ValueError("Stage1.2 qualification is not green")
    if report["spec"] != json.loads(json.dumps(asdict(stage12_spec()))):
        raise ValueError("qualification physical profile mismatch")
    if not report.get("source_sha256"):
        raise ValueError("qualification lacks source provenance")
    for name, expected in report["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"qualification stale source: {name}")
    return report


def uniform_parallel():
    d = np.zeros((64, 64), dtype=bool)
    for y in (7, 19, 31, 43, 55):
        d[:, y : y + 2] = True
    d[:4, 7:57] = d[60:, 7:57] = True
    d[4:60, 33] = True
    d[28:36, 34] = True
    assert d.sum() == 1024
    return d


def initial_parameters(family, task_index):
    if family == "pixel":
        return np.random.default_rng(2026101700 + task_index).normal(0, 0.1, 4096)
    if family != "vascular":
        raise ValueError("unknown design family")
    raw = np.zeros((4, 4))
    for b, target in enumerate((8.0, 24.0, 39.0, 55.0)):
        previous = 31.5
        for k in range(4):
            y = np.clip(
                target, max(6.0, 31.5 - 16 * (4 - k)), min(57.0, 31.5 + 16 * (4 - k))
            )
            delta = np.clip((y - previous) / 16, -0.995, 0.995)
            raw[b, k] = np.arctanh(delta)
            previous = np.clip(
                previous + 16 * delta,
                max(6.0, 31.5 - 16 * (4 - k)),
                min(57.0, 31.5 + 16 * (4 - k)),
            )
    return np.concatenate([raw.ravel(), np.zeros(4 + 4096)])


def score_field(values, family):
    if family == "pixel":
        return filter_logits(values.reshape(64, 64)), None
    if family != "vascular" or values.shape != (4116,):
        raise ValueError("vascular vector has 16 increments,4 widths,4096 pixels")
    waypoints = limited_waypoints(values[:16].reshape(1, 4, 4))
    widths = 1 + 3 * values[16:20].sigmoid().reshape(1, 4)
    pixel = filter_logits(values[20:].reshape(64, 64))[None]
    score, distance = smooth_scores(waypoints, widths, pixel)
    return score[0], distance.amin(1) <= 0.75


def continuous_design(values, family, *, beta):
    score, _ = score_field(values, family)
    return project_volume(score, beta=beta, target=0.25)[0]


def binary_design(values, family):
    with torch.no_grad():
        score, core = score_field(values, family)
        binary = (
            binary_topk(score[None])
            if core is None
            else reserve_core(score[None], core)[0]
        )
    return binary[0].cpu().numpy()


def objective(values, family, solver, index):
    level = 0 if index < 200 else 1 if index < 400 else 2
    beta, alpha = (2.0, 4.0, 8.0)[level], (100.0, 250.0, 500.0)[level]
    design = continuous_design(values, family, beta=beta)
    result = solver(design[None])
    active = result.temperature[0, :, 0, solver.chip[0]].flatten() / 100
    thermal = (torch.logsumexp(alpha * active, 0) - np.log(active.numel())) / alpha
    tv = (design[1:] - design[:-1]).abs().mean() + (
        design[:, 1:] - design[:, :-1]
    ).abs().mean()
    loss = (
        thermal + 0.001 * tv + (0.0, 0.01, 0.02)[level] * (design * (1 - design)).mean()
    )
    return loss, result


def evaluate_binary(problem, design):
    d = np.asarray(design)
    if d.shape != (64, 64) or not np.all((d == 0) | (d == 1)) or d.sum() != 1024:
        raise ValueError(
            "independent evaluation requires exactly1024 binary channel cells"
        )
    result = solve_reference(d, problem)
    status = candidate_status(result.diagnostics)
    geom = connectivity(d)
    eligible = status["accepted"] and geom["all_cells_port_connected"]
    reasons = []
    if status["reynolds_contract_violated"]:
        reasons.append("local_reynolds_ge_500")
    if not status["numerically_accepted"]:
        reasons.append("numerical_residual_gate")
    if not status["conservation_accepted"]:
        reasons.append("mass_or_heat_balance")
    if not geom["all_cells_port_connected"]:
        reasons.append("disconnected_selected_cells")
    record = dict(
        peak_rise_k=float(result.scenario_peaks.max()),
        scenario_peaks_k=result.scenario_peaks.tolist(),
        flow_ml_s=result.flow_ml_s,
        diagnostics=result.diagnostics,
        physics_status=status,
        geometry=geom,
        accepted=bool(eligible),
        rejection_reasons=reasons,
        binary_sha256=hashlib.sha256(d.astype(np.uint8).tobytes()).hexdigest(),
    )
    return record, result


def best_checkpoint(rows):
    eligible = [r for r in rows if r["accepted"]]
    return (
        min(eligible, key=lambda r: (r["peak_rise_k"], r["step"])) if eligible else None
    )


def optimize(problem, method, task_index, output, *, steps=600):
    """One method; files immutable except explicitly named progress.json."""
    if method not in ("Pixel-Adam", "Pixel-MMA", "Vascular-Adam", "Vascular-MMA"):
        raise ValueError("unknown method")
    if not isinstance(steps, int) or steps < 1 or steps > 600:
        raise ValueError("steps must be integer1..600")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    family = method.split("-")[0].lower()
    start = time.perf_counter()
    solver = MCLSolver([problem])
    current = initial_parameters(family, task_index)
    values = torch.tensor(current, dtype=torch.float64, requires_grad=True)
    bound = np.full_like(current, 12.0)
    if family == "vascular":
        bound[:20] = 6.0
    history = []
    checkpoints = []
    nfev = 0
    forward_rhs = adjoint_rhs = 0
    wanted = {0, 50, 100, 200, 400, steps}
    terminated_early = False
    optimizer_stops = []

    def checkpoint(v, step, suffix=""):
        binary = binary_design(v, family)
        record, result = evaluate_binary(problem, binary)
        record["step"] = step
        record["elapsed_seconds"] = time.perf_counter() - start
        checkpoints.append(record)
        np.savez_compressed(
            output / f"checkpoint_{step:04d}{suffix}.npz",
            parameters=v.detach().numpy(),
            binary=binary,
            temperature=result.temperature,
            pressure=result.pressure,
            qx=result.qx,
            qy=result.qy,
        )
        return record

    def calculate(v, index):
        nonlocal nfev, forward_rhs, adjoint_rhs
        loss, result = objective(v, family, solver, index)
        loss.backward()
        derivative = v.grad.detach().numpy().copy()
        if not torch.isfinite(loss) or not np.isfinite(derivative).all():
            raise RuntimeError("nonfinite loss/gradient")
        nfev += 1
        forward_rhs += 1 + len(problem.power_w)
        adjoint_rhs += 1 + len(problem.power_w)
        history.append(
            dict(
                evaluation=nfev,
                loss=float(loss.detach()),
                continuous_peak_rise_k=float(result.scenario_peaks.detach().max()),
                max_linear_residual=max(
                    max(r["relative_residual"], r["compensated_relative_residual"])
                    for r in solver.linear_diagnostics
                ),
            )
        )
        if nfev % 25 == 0 or nfev == steps:
            progress = dict(
                method=method,
                task_index=task_index,
                evaluations=nfev,
                steps=steps,
                elapsed_seconds=time.perf_counter() - start,
                last=history[-1],
            )
            (output / "progress.json").write_text(
                json.dumps(progress, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(
                f"{task_index}/{method}: {nfev}/{steps}, "
                f"{progress['elapsed_seconds']:.1f}s",
                flush=True,
            )
        return float(loss.detach()), derivative

    checkpoint(values, 0)
    if method.endswith("Adam"):
        optimizer = torch.optim.Adam([values], lr=0.05)
        lo = torch.tensor(-bound)
        hi = torch.tensor(bound)
        for index in range(steps):
            optimizer.zero_grad(set_to_none=True)
            calculate(values, index)
            optimizer.step()
            with torch.no_grad():
                values.clamp_(lo, hi)
            if index + 1 in wanted:
                checkpoint(values, index + 1)
    else:
        import nlopt

        boundaries = sorted({0, min(200, steps), min(400, steps), steps})
        for first, last in pairwise(boundaries):

            def callback(x, g, first=first):
                v = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                loss, grad = calculate(v, first)
                if g.size:
                    g[:] = grad
                if nfev in wanted and nfev != steps:
                    checkpoint(v, nfev)
                return loss

            opt = nlopt.opt(nlopt.LD_MMA, len(current))
            opt.set_lower_bounds(-bound)
            opt.set_upper_bounds(bound)
            opt.set_min_objective(callback)
            opt.set_maxeval(last - first)
            current = opt.optimize(current)
            optimizer_stops.append(
                dict(
                    phase_start=first, code=opt.last_optimize_result(), evaluations=nfev
                )
            )
            if nfev != last:
                terminated_early = True
                break
        values = torch.tensor(current, dtype=torch.float64)
        checkpoint(values, nfev, suffix="_final")
    selected = best_checkpoint(checkpoints)
    report = dict(
        method=method,
        task_index=task_index,
        completed=True,
        terminated_early=terminated_early,
        requested_evaluations=steps,
        optimizer_stops=optimizer_stops,
        steps=nfev,
        history=history,
        checkpoints=checkpoints,
        selected=selected,
        final=checkpoints[-1],
        elapsed_seconds=time.perf_counter() - start,
        physical_validation=False,
        counts=dict(
            objective_evaluations=nfev,
            optimizer_updates=nfev if method.endswith("Adam") else None,
            forward_rhs=forward_rhs,
            adjoint_rhs=adjoint_rhs,
            independent_rhs=len(checkpoints) * (1 + len(problem.power_w)),
        ),
    )
    report["update_count_note"] = (
        "Adam steps counted; NLopt exposes function evaluations, "
        "not accepted MMA updates."
    )
    (output / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report
