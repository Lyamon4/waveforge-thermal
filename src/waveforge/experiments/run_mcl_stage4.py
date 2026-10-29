"""Frozen OPEN-dev physical audit. Separate artifacts; no training/test access."""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from waveforge.lab.mcl.contracts import problem_from_task, stage12_spec
from waveforge.lab.mcl.high_fidelity_eval import (
    hydraulic_targets,
    refine_problem,
    solve_audit,
)
from waveforge.lab.mcl.stage3_training import worker_init
from waveforge.lab.mcl.stage4_inventory import (
    METHODS,
    load_inventory,
    seal_files,
    verify_seal,
)
from waveforge.lab.mcl.stage4_metrics import paired_gain, rank_agreement


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def numerical_summary(diag, minimum_temperature):
    linear = [
        diag["pressure_residual"],
        diag["compensated_pressure_residual"],
        *diag["thermal_residual"],
        *diag["compensated_thermal_residual"],
    ]
    balances = [diag["mass_relative_error"], *diag["energy_relative_error"]]
    transport = diag.get("transport_equation_residual", 0.0)
    fixed_point = max(
        diag.get("alpha_fixed_point_residual", 0.0),
        diag.get("thermal_iteration_change", 0.0),
    )
    accepted = (
        all(np.isfinite(v) and 0 <= v < 1e-12 for v in linear)
        and all(
            np.isfinite(v) and 0 <= v < 1e-10
            for v in [*balances, transport, fixed_point]
        )
        and diag.get("transport_converged", True)
        and diag.get("nonlinear_converged", True)
        and minimum_temperature >= -1e-9
    )
    return dict(
        numerical_ok=bool(accepted),
        max_linear_system_residual=max(linear),
        max_transport_equation_residual=transport,
        max_nonlinear_fixed_point_residual=fixed_point,
        max_residual=max(*linear, transport, fixed_point),
        max_balance_error=max(balances),
    )


def reynolds_summary(diags):
    original = max(a["max_local_reynolds_estimate"] for a in diags)
    local = max(a.get("local_cell_re_variable_mu", 0.0) for a in diags)
    return dict(
        max_local_re=original,
        max_cell_re_variable_mu=local,
        within_original_re_gate=original < 500,
        original_re_basis="maximum face/cell speed; fixed reference viscosity",
        local_mu_re_basis="cell-averaged speed; local viscosity; separate diagnostic",
    )


def evaluate_entry(entry, task, controls):
    worker_init()
    start = time.perf_counter()
    p = problem_from_task(task, stage12_spec())
    with np.load(entry["path"], allow_pickle=False) as saved:
        d = saved["binary"]
    if controls.get("grid", 64) == 128:
        p, d = refine_problem(p, d, 2)
    r = solve_audit(
        d,
        p,
        h_air=controls["h_air"],
        pressure_drop_pa=controls["pressure_pa"],
        scheme=controls.get("scheme", "upwind"),
    )
    diag = r.diagnostics
    numerical = numerical_summary(diag, float(r.temperature.min()))
    return dict(
        task=entry["task"],
        method=entry["method"],
        **controls,
        peak_rise_k=float(r.scenario_peaks.max()),
        silicon_max_c=diag["peak_silicon_temperature_c"],
        water_max_c=diag["peak_open_water_temperature_c"],
        flow_ml_s=r.flow_ml_s,
        pump_w=diag["pump_hydraulic_power_w"],
        max_local_re=diag["max_local_reynolds_estimate"],
        **numerical,
        within_original_re_gate=diag["max_local_reynolds_estimate"] < 500,
        silicon_below_95c=diag["peak_silicon_temperature_c"] <= 95,
        water_below_60c=diag["peak_open_water_temperature_c"] < 60,
        water_heat_fraction_min=min(diag["water_heat_fraction"]),
        water_heat_fraction_max=max(diag["water_heat_fraction"]),
        flowing_open_fraction=diag["flowing_open_fraction"],
        wall_face_flux_fraction=diag["wall_touching_internal_abs_face_flux_fraction"],
        wall_outlet_flow_fraction=diag["wall_outlet_flow_fraction"],
        elapsed_seconds=time.perf_counter() - start,
        scenario_peaks_k=r.scenario_peaks.tolist(),
        diagnostics=diag,
    )


def write_tables(folder, rows):
    write_json(folder / "results.json", rows)
    flat = [
        {k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows
    ]
    fields = list(dict.fromkeys(k for r in flat for k in r))
    with (folder / "results.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def evaluate_nonlinear(entry, task, controls):
    from waveforge.lab.mcl.stage4_nonlinear import solve_nonlinear

    worker_init()
    start = time.perf_counter()
    p = problem_from_task(task, stage12_spec())
    with np.load(entry["path"], allow_pickle=False) as saved:
        d = saved["binary"]
    r = solve_nonlinear(
        d,
        p,
        h_air=controls["h_air"],
        c_f=controls["c_f"],
        variable_viscosity=controls["variable_viscosity"],
        resistance_ratio=controls["resistance_ratio"],
    )
    diags = r["scenario_diagnostics"]
    t = r["temperature"]
    qualified = [numerical_summary(a, float(t[i].min())) for i, a in enumerate(diags)]
    numerical = {
        key: max(a[key] for a in qualified)
        for key in qualified[0]
        if key != "numerical_ok"
    }
    numerical["numerical_ok"] = all(a["numerical_ok"] for a in qualified)
    return dict(
        task=entry["task"],
        method=entry["method"],
        **controls,
        peak_rise_k=float(t[:, 0, p.chip_mask].max()),
        silicon_max_c=float(t[:, 0, p.chip_mask].max() + 25),
        water_max_c=float(t[:, 3, d == 1].max() + 25),
        scenario_flow_ml_s=r["scenario_flow_ml_s"],
        **numerical,
        nonlinear_converged=r["converged"],
        max_iterations=max(a["nonlinear_iterations"] for a in diags),
        **reynolds_summary(diags),
        viscosity_range_valid=all(
            a["viscosity_temperature_range_25_100c"] for a in diags
        ),
        flowing_open_fraction=min(a["flowing_open_fraction"] for a in diags),
        wall_face_flux_fraction=max(
            a["wall_touching_internal_abs_face_flux_fraction"] for a in diags
        ),
        wall_outlet_flow_fraction=max(a["wall_outlet_flow_fraction"] for a in diags),
        elapsed_seconds=time.perf_counter() - start,
        diagnostics=diags,
    )


def run_jobs(folder, jobs, tasks, workers, evaluator=evaluate_entry):
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    errors = []
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        futures = {
            pool.submit(evaluator, e, tasks[e["task"]], c): (e, c) for e, c in jobs
        }
        for i, future in enumerate(as_completed(futures), 1):
            e, c = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(
                    task=e["task"],
                    method=e["method"],
                    **c,
                    numerical_ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                errors.append(row)
            rows.append(row)
            write_json(folder / f"case_{i:04d}.json", row)
            if i % 16 == 0 or i == len(jobs):
                print(
                    f"{folder.name}: {i}/{len(jobs)}; exceptions={len(errors)}",
                    flush=True,
                )
    rows.sort(
        key=lambda r: (r.get("mode", ""), r.get("h_air", 0), r["task"], r["method"])
    )
    write_tables(folder, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase", choices=("linear", "grid", "nonlinear"), default="linear"
    )
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    root = Path(__file__).resolve().parents[3]
    selection = Path(
        "artifacts/inputs/development_tasks.json"
    )
    tasks, entries, seals = load_inventory(root, selection)
    out = Path(args.output).resolve()
    if args.phase == "linear":
        out.mkdir(parents=True, exist_ok=False)
        write_json(
            out / "manifest.json",
            dict(
                created_utc=datetime.now(UTC).isoformat(),
                inputs=seals,
                entries=entries,
                task_hashes=[t.fingerprint() for t in tasks],
                closed_test_accessed=False,
                training=False,
                workers=args.workers,
                source_sha256=seal_files(
                    [
                        Path(__file__),
                        *sorted((root / "src/waveforge/lab/mcl").glob("*stage4*.py")),
                        root / "src/waveforge/lab/mcl/high_fidelity_eval.py",
                    ]
                ),
                interpretation=(
                    "sensitivity of reduced-order model, not high-fidelity CFD"
                ),
                pumping_power_references=[
                    "Uniform-Parallel at2500Pa",
                    "Uniform-Parallel at15ml/s",
                ],
            ),
        )
        active = [e for e in entries if e["source_accepted"]]
        jobs = [
            (
                e,
                dict(
                    mode=f"air_{h}",
                    h_air=h,
                    pressure_pa=2500.0,
                    grid=64,
                    scheme="upwind",
                ),
            )
            for h in (0.0, 10.0, 25.0, 50.0, 100.0, 500.0)
            for e in active
        ]
        rows = run_jobs(out / "P0_1", jobs, tasks, args.workers)
        originals = [r for r in rows if r["h_air"] == 500]
        for r in originals:
            source = next(
                e
                for e in active
                if e["task"] == r["task"] and e["method"] == r["method"]
            )
            if (
                not r["numerical_ok"]
                or abs(r["peak_rise_k"] - source["original_peak_rise_k"]) > 1e-8
            ):
                raise RuntimeError(
                    f"original operator replay mismatch: {r['task']} {r['method']}"
                )
        metrics = []
        for h in (0.0, 10.0, 25.0, 50.0, 100.0, 500.0):
            subset = [r for r in rows if r["h_air"] == h]
            metrics.append(
                dict(
                    h_air=h,
                    paired=[
                        paired_gain(subset, m, b)
                        for m in METHODS[2:]
                        for b in METHODS[:2]
                    ],
                    rank_vs_500=rank_agreement(originals, subset),
                )
            )
        write_json(out / "P0_1" / "metrics.json", metrics)
        jobs = []
        for e in active:
            uniform = next(
                x
                for x in active
                if x["task"] == e["task"] and x["method"] == "Uniform-Parallel"
            )
            up = hydraulic_targets(uniform["original_flow_ml_s"], 2500.0, 15.0, 1.0)
            targets = [
                ("iso_delta_p", 2500.0),
                (
                    "iso_flow_15",
                    hydraulic_targets(e["original_flow_ml_s"], 2500, 15, 1)[
                        "iso_flow_pressure_pa"
                    ],
                ),
            ]
            for label, power in [
                ("iso_power_2500", 2500 * uniform["original_flow_ml_s"] * 1e-6),
                ("iso_power_15", up["iso_flow_power_w"]),
            ]:
                targets.append(
                    (
                        label,
                        hydraulic_targets(e["original_flow_ml_s"], 2500, 15, power)[
                            "iso_power_pressure_pa"
                        ],
                    )
                )
            for mode, dp in targets:
                jobs.append(
                    (
                        e,
                        dict(
                            mode=mode,
                            h_air=10.0,
                            pressure_pa=dp,
                            grid=64,
                            scheme="upwind",
                        ),
                    )
                )
        rows2 = run_jobs(out / "P0_2", jobs, tasks, args.workers)
        write_json(
            out / "P0_2" / "metrics.json",
            [
                dict(
                    mode=mode,
                    paired=[
                        paired_gain([r for r in rows2 if r["mode"] == mode], m, b)
                        for m in METHODS[2:]
                        for b in METHODS[:2]
                    ],
                )
                for mode in sorted({r["mode"] for r in rows2})
            ],
        )
    else:
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        verify_seal(previous["inputs"])
        write_json(
            out / f"{args.phase}_manifest.json",
            dict(
                inputs=seals,
                source_sha256=seal_files(
                    [
                        Path(__file__),
                        *sorted((root / "src/waveforge/lab/mcl").glob("*stage4*.py")),
                        root / "src/waveforge/lab/mcl/high_fidelity_eval.py",
                    ]
                ),
                closed_test_accessed=False,
                created_utc=datetime.now(UTC).isoformat(),
                workers=args.workers,
            ),
        )
        active = [e for e in entries if e["source_accepted"]]
        if args.phase == "grid":
            jobs = [
                (
                    e,
                    dict(
                        mode=f"{grid}_{scheme}",
                        h_air=10.0,
                        pressure_pa=2500.0,
                        grid=grid,
                        scheme=scheme,
                    ),
                )
                for grid, scheme in [(64, "vanleer"), (128, "upwind"), (128, "vanleer")]
                for e in active
            ]
            rows = run_jobs(out / "P0_4", jobs, tasks, args.workers)
            original = json.loads(
                (out / "P0_1/results.json").read_text(encoding="utf-8")
            )
            original = [dict(r, mode="64_upwind") for r in original if r["h_air"] == 10]
            combined = original + rows
        else:
            jobs = [
                (
                    e,
                    dict(
                        mode=mode,
                        h_air=10.0,
                        pressure_pa=2500.0,
                        grid=64,
                        c_f=cf,
                        variable_viscosity=vm,
                        resistance_ratio=1e7,
                    ),
                )
                for mode, cf, vm in [
                    ("wall_contrast_only", 0.0, False),
                    ("nonlinear_mu", 0.55, True),
                ]
                for e in active
            ]
            rows = run_jobs(out / "P0_3", jobs, tasks, args.workers, evaluate_nonlinear)
            original = json.loads(
                (out / "P0_1/results.json").read_text(encoding="utf-8")
            )
            original = [
                dict(r, mode="original_linear") for r in original if r["h_air"] == 10
            ]
            combined = original + rows
        folder = out / ("P0_4" if args.phase == "grid" else "P0_3")
        write_json(
            folder / "metrics.json",
            [
                dict(
                    mode=mode,
                    paired=[
                        paired_gain([r for r in combined if r["mode"] == mode], m, b)
                        for m in METHODS[2:]
                        for b in METHODS[:2]
                    ],
                    rank_vs_original=rank_agreement(
                        original, [r for r in combined if r["mode"] == mode]
                    ),
                )
                for mode in sorted({r["mode"] for r in combined})
            ],
        )
    verify_seal(seals)
    write_json(
        out / f"{args.phase}_completion.json",
        dict(completed=True, inputs_unchanged=True, closed_test_accessed=False),
    )
    print(f"Completed {args.phase}: {out}", flush=True)


if __name__ == "__main__":
    main()
