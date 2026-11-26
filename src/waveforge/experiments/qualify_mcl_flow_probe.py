"""Independent saved-field checks and same-objective pressure-only timing."""

import argparse
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from waveforge.lab.hydraulic_probe import (
    FlowResult,
    HydraulicProbe,
    refined_spec,
    replicated_design,
)
from waveforge.lab.mcl.contracts import problem_from_task
from waveforge.lab.mcl.high_fidelity_eval import refine_problem
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init
from waveforge.lab.mcl.stage4_inventory import seal_files, verify_seal
from waveforge.lab.mcl.stage5_evaluation import PUMP_W
from waveforge.lab.mcl.stage5_physics import Stage5Solver, stage5_spec
from waveforge.lab.mcl.stage51 import reynolds_samples

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(a, b):
    return float(
        np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(np.max(np.abs(b)), 1e-30)
    )


def verified_reference_seal(source):
    source = Path(source).resolve()
    receipt = read(source / "completion.json")
    if not receipt.get("completed"):
        raise ValueError("reference campaign incomplete")
    paths = {
        str((source / name).resolve()): sha for name, sha in receipt["files"].items()
    }
    if any(not Path(p).is_relative_to(source) for p in paths):
        raise ValueError("reference file outside its campaign")
    paths.update(seal_files([source / "completion.json"]))
    verify_seal(paths)
    return paths


def compare_locked_flow(result, locked, spec):
    if len(result.pressure) != 1:
        raise ValueError("one locked case required")
    errors = dict(
        flow_relative_error=abs(result.flow_ml_s.item() / locked["flow_ml_s"] - 1),
        re_relative_error=abs(
            reynolds_samples(result, spec).max().item()
            / locked["diagnostics"]["max_local_reynolds_estimate"]
            - 1
        ),
    )
    mass = abs((result.qin.sum() - result.qout.sum()).item()) / result.qin.sum().item()
    if max(errors.values()) >= 1e-11 or not all(np.isfinite(list(errors.values()))):
        raise ValueError("locked flow or Reynolds metric differs")
    measured_rejection = reynolds_samples(result, spec).max().item() >= 1200
    if measured_rejection != ("local_re" in locked["admission"]["rejection_reasons"]):
        raise ValueError("locked hydraulic admission differs")
    if not np.isfinite(mass) or mass >= 1e-10:
        raise ValueError("mass conservation gate failed")
    return dict(**errors, mass_relative_error=mass, hydraulic_admission_matches=True)


def system_counts(records, pressure_size):
    counts = {
        f"{system}_{phase}": dict(systems=0, rhs=0)
        for system in ("pressure", "thermal")
        for phase in ("forward", "adjoint")
    }
    for r in records:
        if r["size"] not in (pressure_size, 4 * pressure_size) or r["phase"] not in (
            "forward",
            "adjoint",
        ):
            raise ValueError("unexpected benchmark system")
        key = (
            ("pressure" if r["size"] == pressure_size else "thermal") + "_" + r["phase"]
        )
        counts[key]["systems"] += 1
        counts[key]["rhs"] += r["rhs"]
    return counts


def main(args):
    SOURCE = Path(args.source).resolve()
    OUT = Path(args.output).resolve()
    selection = Path(args.selection).resolve()
    worker_init()
    reference_seal = verified_reference_seal(SOURCE)
    reference_manifest = read(SOURCE / "manifest.json")
    verify_seal({str(selection): reference_manifest["source_sha256"][str(selection)]})
    selected_tasks = read(selection)["tasks"][:8]
    if [task_from_dict(t).fingerprint() for t in selected_tasks] != reference_manifest[
        "development_task_sha256"
    ]:
        raise ValueError("task source differs from locked development tasks")
    paths = [
        ROOT / "src/waveforge/lab/hydraulic_probe.py",
        ROOT / "tests/test_mcl_flow_probe.py",
        Path(__file__).resolve(),
        ROOT / "tests/test_mcl_probe_qualification.py",
        selection,
        *list((ROOT / "src/waveforge/lab/mcl").glob("*.py")),
        SOURCE / "manifest.json",
        SOURCE / "completion.json",
    ]
    protocol = ROOT / "src/waveforge/experiments/qualify_mcl_flow_probe.py"
    if protocol.is_file():
        paths.append(protocol)
    source_seal = dict(reference_seal, **seal_files(paths))
    OUT.mkdir(exist_ok=False)
    for p in paths:
        if p.is_relative_to(ROOT):
            target = OUT / "SOURCE_SNAPSHOT" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    rows = read(SOURCE / "STAGE5_2_AUDIT/results.json")
    records = []
    spec = stage5_spec()
    fine = refined_spec(spec, 2)
    probe = HydraulicProbe(fine)
    for i in range(8):
        saved = SOURCE / f"STAGE5_2_AUDIT/dev_{i:02d}_128_Manifold-UNet-R25-C.npz"
        coarse = SOURCE / f"dev_{i:02d}/Manifold-UNet-R0/R25-C/selected.npz"
        with np.load(coarse) as data:
            d = torch.tensor(data["binary"][None], dtype=torch.float64)
        result = probe(replicated_design(d, 2))
        with np.load(saved) as data:
            errors = {
                k: relative(getattr(result, k)[0].detach().numpy(), data[k])
                for k in ("pressure", "qx", "qy")
            }
        expected = next(
            r
            for r in rows
            if r["task"] == i
            and r["grid"] == 128
            and r["regime"] == "nominal"
            and r["method"] == "Manifold-UNet-R25-C"
        )
        re = reynolds_samples(result, fine).max().item()
        errors["flow"] = abs(result.flow_ml_s.item() / expected["flow_ml_s"] - 1)
        errors["re"] = abs(
            re / expected["diagnostics"]["max_local_reynolds_estimate"] - 1
        )
        locked_check = compare_locked_flow(result, expected, fine)
        mass = locked_check["mass_relative_error"]
        assert max(errors.values()) < 1e-11, (i, errors)
        assert mass < 1e-10
        fixed = result.at_pump_power(PUMP_W)
        expected_power = next(
            r
            for r in rows
            if r["task"] == i
            and r["grid"] == 128
            and r["regime"] == "power"
            and r["method"] == "Manifold-UNet-R25-C"
        )
        compare_locked_flow(fixed, expected_power, fine)
        records.append(
            dict(
                task=i,
                relative_errors=errors,
                mass_relative_error=mass,
                max_linear_residual=max(
                    max(r["relative_residual"], r["compensated_relative_residual"])
                    for r in probe.linear_diagnostics
                ),
                nominal_re=re,
                power_re=reynolds_samples(fixed, fine).max().item(),
            )
        )
    task = task_from_dict(selected_tasks[4])
    with np.load(SOURCE / "dev_04/Manifold-UNet-R0/checkpoint_0002.npz") as data:
        coarse = data["binary"].copy()
    full_problem, binary = refine_problem(problem_from_task(task, spec), coarse, 2)
    raw = probe(torch.tensor(binary[None], dtype=torch.float64))
    cold_head = read(SOURCE / "dev_04/Manifold-UNet-R0/result.json")["heads"][2]
    cold_fine = compare_locked_flow(raw, cold_head["fine"], fine)
    coarse_flow = HydraulicProbe(spec)(torch.tensor(coarse[None], dtype=torch.float64))
    cold_coarse = compare_locked_flow(coarse_flow, cold_head["coarse"], spec)
    full = Stage5Solver([full_problem])

    def objective(result):
        if not isinstance(result, FlowResult):
            result = FlowResult(
                result.pressure,
                result.qx,
                result.qy,
                result.qin,
                result.qout,
                torch.full((1,), fine.pressure_drop_pa, dtype=torch.float64),
            )
        power = result.at_pump_power(PUMP_W)
        return (
            power.flow_ml_s.log().sum()
            + reynolds_samples(power, fine).square().mean() / 1e6
        )

    def measured(solver):
        x = torch.tensor(binary[None], dtype=torch.float64, requires_grad=True)
        tick = time.perf_counter()
        value = objective(solver(x))
        g = torch.autograd.grad(value, x)[0]
        return (
            time.perf_counter() - tick,
            g.detach().numpy(),
            value.item(),
            [dict(r) for r in solver.linear_diagnostics],
        )

    measured(probe)
    measured(full)
    pressure_times = []
    coupled_times = []
    gradient_errors = []
    benchmark_counts = []
    for _ in range(3):
        a, ga, va, da = measured(probe)
        b, gb, vb, db = measured(full)
        error = relative(ga, gb)
        assert error < 1e-9, error
        assert abs(va - vb) < 1e-10
        assert all(r["size"] == 128 * 128 and r["rhs"] == 1 for r in da)
        assert any(r["size"] == 4 * 128 * 128 for r in db)
        probe_counts = system_counts(da, 128 * 128)
        full_counts = system_counts(db, 128 * 128)
        assert probe_counts["thermal_forward"]["systems"] == 0
        assert probe_counts["thermal_adjoint"]["systems"] == 0
        assert full_counts["thermal_adjoint"]["systems"] == 0
        benchmark_counts.append(dict(probe=probe_counts, full=full_counts))
        pressure_times.append(a)
        coupled_times.append(b)
        gradient_errors.append(error)
    benchmark = dict(
        pressure_only_seconds=pressure_times,
        coupled_seconds=coupled_times,
        median_pressure_only_seconds=statistics.median(pressure_times),
        median_coupled_seconds=statistics.median(coupled_times),
        measured_speed_ratio=statistics.median(coupled_times)
        / statistics.median(pressure_times),
        density_gradient_relative_errors=gradient_errors,
        identical_flow_objective=True,
        system_counts_per_repeat=benchmark_counts,
        no_thermal_adjoint_requested=True,
        thermal_forward_skipped_by_probe=True,
        device="CPU sparse LU, one Torch/BLAS thread",
        warmups_each=1,
        repeats=3,
        end_to_end_optimization_speedup_claim=False,
    )
    verify_seal(source_seal)
    result = dict(
        passed=True,
        physics_changed=False,
        admission_thresholds_changed=False,
        optimized_new_design=False,
        network_retrained=False,
        closed_test_accessed=False,
        saved_field_records=records,
        rejected_cold_head_re=reynolds_samples(raw, fine).max().item(),
        cold_head_coarse_check=cold_coarse,
        cold_head_fine_check=cold_fine,
        benchmark=benchmark,
        source_sha256=source_seal,
        note=(
            "Qualifies continuous-density derivatives and a cheaper constraint "
            "evaluator; no discrete-STE or cooling-quality guarantee."
        ),
    )
    (OUT / "qualification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    shutil.copyfile(ROOT / "scratch/fine_flow_probe_tests.log", OUT / "tests.log")
    (OUT / "completion.json").write_text(
        json.dumps(
            dict(
                completed=True,
                files={
                    str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in OUT.rglob("*")
                    if p.is_file()
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(
                passed=True,
                maximum_saved_field_relative_error=max(
                    max(r["relative_errors"].values()) for r in records
                ),
                benchmark=benchmark,
                output=str(OUT),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
