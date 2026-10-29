"""Independent strict binary evaluation, best-valid selection and matched control."""

import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import evaluate_binary, initial_parameters
from .contracts import problem_from_task, stage12_spec
from .differentiable import MCLSolver
from .stage3 import binary, continuous
from .stage3_training import linear_report, physical_loss, worker_init, write_json


def select_valid(rows):
    accepted = [row for row in rows if row["accepted"]]
    return (
        min(
            accepted,
            key=lambda row: (
                row["peak_rise_k"],
                row.get("step", 0),
                row.get("head", 0),
            ),
        )
        if accepted
        else None
    )


def save_fields(path, parameters, design, result, spec):
    with Path(path).open("xb") as handle:
        np.savez_compressed(
            handle,
            parameters=parameters,
            binary=np.asarray(design, dtype=bool),
            temperature=result.temperature,
            pressure=result.pressure,
            qx=result.qx,
            qy=result.qy,
            qin=result.qin,
            qout=result.qout,
            velocity_x_m_s=result.qx / (spec.dy * spec.dz[3]),
            velocity_y_m_s=result.qy / (spec.dx * spec.dz[3]),
            temperature_units="K above inlet",
            pressure_units="Pa",
            flux_units="m3/s",
            velocity_units="m/s",
            layout="x-first; face velocities; reduced-order superficial velocities",
        )


def record_candidate(problem, parameters, output, step, *, head=None):
    design = binary(parameters[None])[0].numpy()
    record, result = evaluate_binary(problem, design)
    record["step"] = step
    if head is not None:
        record["head"] = head
    record["file"] = (
        f"head_{head}.npz" if head is not None else f"checkpoint_{step:04d}.npz"
    )
    save_fields(
        output / record["file"],
        parameters.detach().numpy(),
        design,
        result,
        problem.spec,
    )
    return record


def optimize_parameters(task, initial, output, *, steps, checkpoints, method):
    worker_init()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    problem = problem_from_task(task, stage12_spec())
    solver = MCLSolver([problem])
    parameters = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([parameters], lr=0.05)
    rows = [record_candidate(problem, parameters, output, 0)]
    max_residual = 0.0
    with (output / "history.jsonl").open("x", encoding="utf-8") as history:
        import json

        for step in range(steps):
            level = min(2, step // 200) if method == "Vascular-Adam600-Halo" else 2
            optimizer.zero_grad(set_to_none=True)
            design = continuous(parameters[None], (2.0, 4.0, 8.0)[level])[0]
            loss, result = physical_loss(design, solver, level)
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(parameters.grad).all():
                raise RuntimeError("nonfinite refinement/control loss or gradient")
            max_residual = max(
                max_residual, linear_report(solver)["max_linear_residual"]
            )
            optimizer.step()
            with torch.no_grad():
                parameters[:20].clamp_(-6, 6)
                parameters[20:].clamp_(-12, 12)
            row = dict(
                step=step + 1,
                loss=loss.item(),
                continuous_peak_rise_k=result.scenario_peaks.detach().max().item(),
                max_linear_residual=max_residual,
            )
            history.write(json.dumps(row, allow_nan=False) + "\n")
            if step + 1 in checkpoints:
                rows.append(record_candidate(problem, parameters, output, step + 1))
            if (step + 1) % 50 == 0:
                write_json(output / "progress.json", row, exclusive=False)
                history.flush()
    selected = select_valid(rows)
    if selected:
        shutil.copyfile(output / selected["file"], output / "selected.npz")
    report = dict(
        method=method,
        steps=steps,
        checkpoints=rows,
        selected=selected,
        max_linear_residual=max_residual,
        elapsed_seconds=time.perf_counter() - start,
    )
    write_json(output / "result.json", report)
    return report


def evaluate_task(task, parameters, output, *, r25_steps=25, control_steps=600):
    worker_init()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    problem = problem_from_task(task, stage12_spec())
    start = time.perf_counter()
    r0_folder = output / "Vascular-UNet-R0"
    r0_folder.mkdir()
    r0_rows = [
        record_candidate(
            problem, torch.tensor(p, dtype=torch.float64), r0_folder, 0, head=h
        )
        for h, p in enumerate(parameters)
    ]
    selected = select_valid(r0_rows)
    r0 = dict(
        method="Vascular-UNet-R0",
        checkpoints=r0_rows,
        selected=selected,
        elapsed_seconds=time.perf_counter() - start,
    )
    if selected:
        shutil.copyfile(r0_folder / selected["file"], r0_folder / "selected.npz")
    write_json(r0_folder / "result.json", r0)
    if selected:
        head = selected["head"]
        r25 = optimize_parameters(
            task,
            parameters[head],
            output / "Vascular-UNet-R25",
            steps=r25_steps,
            checkpoints=set(range(5, r25_steps + 1, 5)) | {r25_steps},
            method="Vascular-UNet-R25",
        )
        r25["source_head"] = head
        r25["initial_r0_sha256"] = selected["binary_sha256"]
        # Independent step 0 must reproduce the exact R0 candidate.
        if r25["checkpoints"][0]["binary_sha256"] != selected["binary_sha256"]:
            raise RuntimeError("R25 step 0 differs from selected R0")
        if r25["selected"]["peak_rise_k"] > selected["peak_rise_k"] + 1e-10:
            raise RuntimeError("best-valid R25 regressed below its R0 incumbent")
        write_json(output / "Vascular-UNet-R25" / "result.json", r25, exclusive=False)
    else:
        (output / "Vascular-UNet-R25").mkdir()
        r25 = dict(
            method="Vascular-UNet-R25",
            selected=None,
            checkpoints=[],
            steps=0,
            reason="No admissible R0 head; no fallback or topology repair",
            elapsed_seconds=0.0,
        )
        write_json(output / "Vascular-UNet-R25" / "result.json", r25)
    control = optimize_parameters(
        task,
        initial_parameters("vascular", 0),
        output / "Vascular-Adam600-Halo",
        steps=control_steps,
        checkpoints={
            n for n in (50, 100, 200, 400, control_steps) if n <= control_steps
        },
        method="Vascular-Adam600-Halo",
    )
    return [r0, r25, control]
