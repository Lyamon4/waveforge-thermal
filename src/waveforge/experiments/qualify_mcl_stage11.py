"""Stage 1.1 qualification; preserve all failed claims rather than repairing data."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
import torch

from waveforge.experiments.qualify_mcl_stage1 import (
    TASK0_SHA256,
    _numpy_objective,
    _sha,
    _torch_objective,
)
from waveforge.lab.contracts import Chiplet, ChipletTask
from waveforge.lab.mcl import (
    MCLSolver,
    problem_from_task,
    solve_reference,
    straight_channel,
)
from waveforge.lab.mcl.complex_step import complex_step_check
from waveforge.lab.mcl.contracts import stage11_spec
from waveforge.lab.mcl.linear import NumericalQualificationError
from waveforge.lab.mcl.vascular import binary_topk, connectivity, vascular_scores


def verdict(*, numerical, gradients, connectivity, hydraulic):
    qualified = bool(numerical and gradients and connectivity and hydraulic)
    return {
        "stage11_qualified": qualified,
        "numerically_qualified": bool(numerical and gradients),
        "decoder_qualification_passed": bool(connectivity),
        "sampled_hydraulic_gate_passed": bool(hydraulic),
        "physically_validated": False,
        "baseline_launch_allowed": qualified,
        "training_launched": False,
    }


def reference_records(result, label):
    """No diagnostic forward solve is exempt from the original residual gate."""
    values = [
        result.diagnostics["pressure_residual"],
        *result.diagnostics["thermal_residual"],
    ]
    compensated = [
        result.diagnostics.get("compensated_pressure_residual", values[0]),
        *result.diagnostics.get("compensated_thermal_residual", values[1:]),
    ]
    if len(values) != len(compensated):
        raise ValueError("residual diagnostics do not cover the same RHS")
    return [
        {
            "label": label,
            "system": "pressure" if i == 0 else "temperature",
            "rhs": None if i == 0 else i - 1,
            "relative_residual": float(value),
            "compensated_relative_residual": float(compensated[i]),
            "accepted": bool(
                all(np.isfinite(v) and 0 <= v < 1e-12 for v in (value, compensated[i]))
            ),
            "residual_system": "original_unscaled",
        }
        for i, value in enumerate(values)
    ]


def record_residuals(records):
    return [
        v
        for r in records
        for v in (
            r["relative_residual"],
            r.get("compensated_relative_residual", r["relative_residual"]),
        )
    ]


def check_geometry(root, output):
    """Replay old parameters and fixed prospective stress stream, without repairs."""
    old_path = root / "artifacts/mcl_stage1_20261017/connectivity_counterexample.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261018)
    fixtures = [("previous_counterexample", old["waypoints_y"], old["widths"])]
    unequal = [4.0, 1, 1, 1, 1, 1]
    fixtures.append(
        ("extreme_alternating", [[4, 59, 4, 59], [59, 4, 59, 4]] * 3, unequal)
    )
    for i in range(128):
        fixtures.append((f"extreme_{i}", rng.choice([4.0, 59.0], (6, 4)), unequal))
    for i in range(256):
        fixtures.append(
            (f"uniform_{i}", rng.uniform(4, 59, (6, 4)), rng.uniform(1, 4, 6))
        )
    records, examples = [], {}
    largest_core = None
    for name, y, w in fixtures:
        yt = torch.tensor(np.asarray(y)[None], dtype=torch.float64)
        wt = torch.tensor(np.asarray(w)[None], dtype=torch.float64)
        with torch.no_grad():
            scores, ds = vascular_scores(yt, wt)
            core = ds.amin(1) <= 0.8
            # Permitted residual depresses core, raises off-core: adverse ranking.
            for mode in ("zero", "adverse"):
                residual = (
                    torch.zeros_like(scores)
                    if mode == "zero"
                    else torch.where(core, -20.0, 20.0)
                )
                scored = scores + 0.35 * residual.tanh()
                binary = binary_topk(scored)[0].numpy()
                record = connectivity(binary, ds[0].numpy())
                record.update(name=name, residual=mode)
                records.append(record)
                if (
                    largest_core is None
                    or record["core_cells"] > largest_core["core_cells"]
                ):
                    largest_core = dict(
                        record,
                        waypoints=np.asarray(y).tolist(),
                        widths=np.asarray(w).tolist(),
                    )
                failure = (
                    not record["all_cells_port_connected"]
                    or record["omitted_core_cells"] > 0
                )
                kind = (
                    "disconnected_ports"
                    if not record["inlet_outlet_connected"]
                    else "islands"
                    if not record["all_cells_port_connected"]
                    else "omitted_core"
                )
                if failure and kind not in examples:
                    example = dict(
                        record,
                        waypoints=np.asarray(y).tolist(),
                        widths=np.asarray(w).tolist(),
                    )
                    examples[kind] = example
                    np.savez_compressed(
                        output / f"geometry_{kind}.npz",
                        binary=binary,
                        scores=scored[0].numpy(),
                        distances=ds[0].numpy(),
                        waypoints=np.asarray(y),
                        widths=np.asarray(w),
                        pixel=residual[0].numpy(),
                    )
    max_core = max(records, key=lambda r: r["core_cells"])
    return {
        "seed": 20261018,
        "fixture_count": len(fixtures),
        "evaluations": len(records),
        "previous_fixture_sha256": _sha(old_path),
        "records": records,
        "max_core_cells": max_core["core_cells"],
        "max_core_fixture": max_core["name"],
        "largest_core_parameters": largest_core,
        "failures_ports": sum(not r["inlet_outlet_connected"] for r in records),
        "failures_all_selected_connected": sum(
            not r["all_cells_port_connected"] for r in records
        ),
        "failures_all_core_retained": sum(r["omitted_core_cells"] > 0 for r in records),
        "counterexamples": examples,
        "passed": all(
            r["all_cells_port_connected"] and r["omitted_core_cells"] == 0
            for r in records
        ),
        "finite_stress_test_not_universal_proof": True,
    }


def run(
    selection,
    output,
    *,
    spec=None,
    geometry_checker=check_geometry,
    stage="1.1",
    extra_sources=(),
):
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[3]
    payload = json.loads(selection.read_text(encoding="utf-8"))["tasks"][0]
    task = ChipletTask(
        tuple(Chiplet(**c) for c in payload["chiplets"]),
        tuple(tuple(p) for p in payload["powers_w"]),
    )
    if task.fingerprint() != TASK0_SHA256:
        raise ValueError("development task 0 fingerprint mismatch")
    spec = stage11_spec() if spec is None else spec
    problem = problem_from_task(task, spec)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "stage": stage,
        "task_sha256": task.fingerprint(),
        "task": task.canonical(),
        "selection_sha256": _sha(selection),
        "closed_test_accessed": False,
        "spec": asdict(spec),
        "source_sha256": {},
        "software": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "backend": "CPU sparse LU; Torch coefficient assembly and coupled adjoint",
        "thresholds": {
            "unscaled_relative_residual_lt": 1e-12,
            "gradient_relative_error_lt": 1e-6,
            "temperature_max_error_k_le": 1e-8,
            "mass_relative_error_lt": 1e-10,
            "energy_relative_error_lt": 1e-10,
            "sampled_local_re_lt": 500,
        },
    }
    for path in [
        *sorted((root / "src/waveforge/lab/mcl").glob("*.py")),
        Path(__file__),
        root / "src/waveforge/experiments/qualify_mcl_stage1.py",
        root / "src/waveforge/experiments/qualify_mcl_stage11.py",
        *extra_sources,
    ]:
        report["source_sha256"][path.relative_to(root).as_posix()] = _sha(path)
    print(
        f"Stage {stage}: isolated physical profile, original tolerances, task 0 only",
        flush=True,
    )
    straight = straight_channel(spec)
    neck = straight.copy()
    neck[32, 24:40] = 0
    neck[32, 31] = 1
    neck[10:25, 23] = (
        1  # 15 removed cells restored, still ONE connected 1024-cell domain
    )
    assert neck.sum() == 1024 and connectivity(neck)["all_cells_port_connected"]
    designs = {
        "straight": straight,
        "gray_diagnostic": np.full((64, 64), 0.25),
        "blocked_diagnostic": np.zeros((64, 64)),
        "one_cell_neck": neck,
    }
    controls, residuals, differences, mass, heat = {}, [], [], [], []
    rejected = False
    for name, design in designs.items():
        reference = solve_reference(design, problem)
        solver = MCLSolver([problem])
        error = None
        difference = None
        try:
            result = solver(torch.tensor(design[None], dtype=torch.float64))
            difference = float(
                np.max(abs(result.temperature.numpy()[0] - reference.temperature))
            )
            differences.append(difference)
        except NumericalQualificationError as exc:
            error = str(exc)
            rejected = True
        dg = reference.diagnostics
        residuals.extend(record_residuals(reference_records(reference, name)))
        residuals.extend(record_residuals(solver.linear_diagnostics))
        mass.append(dg["mass_relative_error"])
        heat.extend(dg["energy_relative_error"])
        controls[name] = {
            "peak_rise_k": float(reference.scenario_peaks.max()),
            "flow_ml_s": reference.flow_ml_s,
            "diagnostics": dg,
            "torch_rejection": error,
            "torch_records": solver.linear_diagnostics,
            "max_temperature_difference_k": difference,
        }
        np.savez_compressed(
            output / f"{name}.npz",
            design=design,
            pressure=reference.pressure,
            temperature=reference.temperature,
            qx=reference.qx,
            qy=reference.qy,
            qin=reference.qin,
            qout=reference.qout,
        )
        print(
            f"{name}: Q={reference.flow_ml_s:.6f} mL/s, "
            f"local Re={dg['max_local_reynolds_estimate']:.2f}, "
            f"residual={max(dg['thermal_residual']):.3e}",
            flush=True,
        )
    report["controls"] = controls
    old_probes = root / "artifacts/mcl_stage1_20261017/run_004/gradient_check.npz"
    with np.load(old_probes, allow_pickle=False) as saved:
        design = saved["design"].copy()
        names = [k for k in saved.files if k not in ("design", "gradient")]
        directions = np.stack([saved[k] for k in names])
    # Record rejected residuals but finish the derivative experiment. This is not
    # tolerance relaxation: rejected solves KEEP the numerical gate false below.
    solver = MCLSolver([problem], diagnostic_only=True)
    d = torch.tensor(design[None], dtype=torch.float64, requires_grad=True)
    gradient_report = {
        "saved_directions_sha256": _sha(old_probes),
        "direction_names": names,
    }
    gradient_report["diagnostic_only_continuation"] = True
    gradient_ok = False
    try:
        result = solver(d)
        gt = torch.autograd.grad(
            _torch_objective(result.temperature, problem.chip_mask),
            d,
            retain_graph=True,
        )[0].numpy()[0]
        gp = torch.autograd.grad(result.flow_ml_s.sum(), d)[0].numpy()[0]
        checks = complex_step_check(design, problem, directions, gt, gp)
        gradient_report.update(checks)
        gradient_ok = (
            max(checks["max_thermal_relative_error"], checks["max_flow_relative_error"])
            < 1e-6
        )
        residuals.append(checks["max_complex_solve_relative_residual"])
        ref = solve_reference(design, problem)
        differences.append(
            float(np.max(abs(result.temperature.detach().numpy()[0] - ref.temperature)))
        )
        mass.append(ref.diagnostics["mass_relative_error"])
        heat.extend(ref.diagnostics["energy_relative_error"])
        residuals.extend(record_residuals(reference_records(ref, "gradient_base")))
        fd = []
        for name, direction in zip(names, directions, strict=True):
            adjoint = float((gt * direction).sum())
            for epsilon in (1e-4, 1e-6):
                plus = solve_reference(design + epsilon * direction, problem)
                minus = solve_reference(design - epsilon * direction, problem)
                diagnostic_records = []
                for side, solved in (("plus", plus), ("minus", minus)):
                    diagnostic_records.extend(
                        reference_records(solved, f"{name}/{epsilon}/{side}")
                    )
                    mass.append(solved.diagnostics["mass_relative_error"])
                    heat.extend(solved.diagnostics["energy_relative_error"])
                residuals.extend(record_residuals(diagnostic_records))
                central = (
                    _numpy_objective(plus.temperature, problem.chip_mask)
                    - _numpy_objective(minus.temperature, problem.chip_mask)
                ) / (2 * epsilon)
                fd.append(
                    {
                        "name": name,
                        "epsilon": epsilon,
                        "adjoint": adjoint,
                        "central": central,
                        "linear_records": diagnostic_records,
                        "relative_error": abs(central - adjoint)
                        / max(abs(central), abs(adjoint), 1e-30),
                    }
                )
        gradient_report["additional_finite_differences"] = fd
        np.savez_compressed(
            output / "gradient_check.npz",
            design=design,
            thermal_gradient=gt,
            flow_gradient=gp,
            directions=directions,
        )
    except NumericalQualificationError as exc:
        gradient_report["numerical_rejection"] = str(exc)
        rejected = True
    except ValueError as exc:
        # Invalid/non-smooth complex-step probe is a failed check, not skipped success.
        gradient_report["probe_rejection"] = str(exc)
    gradient_report["torch_records"] = solver.linear_diagnostics
    if any(not r["accepted"] for r in solver.linear_diagnostics):
        rejected = True
    residuals.extend(record_residuals(solver.linear_diagnostics))
    report["gradient_check"] = gradient_report
    print(f"Complex-step check passed: {gradient_ok}", flush=True)
    geometry = geometry_checker(root, output)
    report["geometry_check"] = geometry
    numerical = (
        not rejected
        and all(
            np.isfinite(v) and v >= 0 for v in [*residuals, *differences, *mass, *heat]
        )
        and len(differences) >= len(designs)
        and max(residuals) < 1e-12
        and max(differences) <= 1e-8
        and max(mass) < 1e-10
        and max(heat) < 1e-10
    )
    hydraulic = all(
        c["diagnostics"]["max_local_reynolds_estimate"] < 500 for c in controls.values()
    )
    report.update(
        max_linear_residual=max(residuals),
        max_temperature_difference_k=max(differences, default=None),
        max_mass_relative_error=max(mass),
        max_energy_relative_error=max(heat),
        verdict=verdict(
            numerical=numerical,
            gradients=gradient_ok,
            connectivity=geometry["passed"],
            hydraulic=hydraulic,
        ),
        total_seconds=time.perf_counter() - started,
    )
    if stage != "1.1":
        report["verdict"][f"stage{stage.replace('.', '')}_qualified"] = report[
            "verdict"
        ].pop("stage11_qualified")
    with (output / "qualification.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report["verdict"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run(args.selection, args.output)
    if not report["verdict"]["stage11_qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
