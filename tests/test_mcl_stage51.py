"""Stage5.1: flow gradients and strict two-grid selection, without new physics."""

import importlib.util

import numpy as np
import pytest
import torch


def module():
    name = "waveforge.lab.mcl.stage51"
    assert importlib.util.find_spec(name) is not None, "Stage5.1 implementation missing"
    return __import__(name, fromlist=["*"])


def row(peak=50, re=800):
    return dict(
        accepted=True,
        peak_rise_k=peak,
        silicon_peak_c=peak + 25,
        diagnostics=dict(
            max_local_reynolds_estimate=re,
            peak_open_water_temperature_c=40,
            pressure_residual=1e-14,
            compensated_pressure_residual=1e-14,
            thermal_residual=[1e-14],
            compensated_thermal_residual=[1e-14],
            mass_relative_error=1e-13,
            energy_relative_error=[1e-13],
        ),
    )


def test_selection_rejects_fine_grid_failure_and_keeps_r0():
    m = module()
    r0 = dict(coarse=row(52), fine=row(53), step=0)
    failed_refinement = dict(coarse=row(45), fine=row(46, 1200), step=25)
    assert m.choose([r0, failed_refinement]) is r0
    bad_coarse = dict(coarse=row(45, 1201), fine=row(46), step=10)
    assert m.choose([bad_coarse, r0]) is r0
    assert not m.admissible_both(failed_refinement)


def test_selection_no_valid_is_explicit_diagnostic_not_a_win():
    m = module()
    bad = dict(coarse=row(30), fine=row(31, 1400), step=0)
    assert m.choose([bad]) is bad
    assert not m.admissible_both(m.choose([bad]))
    with pytest.raises(ValueError):
        m.choose([])


def test_flow_samples_match_independent_scipy_including_cell_speed():
    m = module()
    from waveforge.lab.mcl.contracts import MCLProblem, straight_channel
    from waveforge.lab.mcl.stage5_physics import Stage5Solver, stage5_spec
    from waveforge.lab.mcl.stage5_preflight import solve_profile

    s = stage5_spec(nx=8, ny=8)
    mask = np.ones((8, 8), bool)
    p = MCLProblem(s, mask, np.full((3, 8, 8), 10 / 64))
    d = straight_channel(s)
    result = Stage5Solver([p])(torch.tensor(d[None], dtype=torch.float64))
    expected = solve_profile(d, p, k_tim=40).diagnostics["max_local_reynolds_estimate"]
    assert m.reynolds_samples(result, s).max().item() == pytest.approx(
        expected, rel=1e-11
    )


def test_flow_penalty_adjoint_matches_four_finite_difference_directions():
    m = module()
    # Diagnostic high-flow fixture exercises the active penalty, not a production mode.
    from dataclasses import replace

    from waveforge.lab.mcl.contracts import MCLProblem
    from waveforge.lab.mcl.stage5_physics import Stage5Solver, stage5_spec

    s = replace(stage5_spec(nx=8, ny=8), pressure_drop_pa=30000.0)
    mask = np.ones((8, 8), bool)
    p = MCLProblem(s, mask, np.full((3, 8, 8), 10 / 64))
    solver = Stage5Solver([p])
    rng = np.random.default_rng(2026102801)
    x = torch.tensor(rng.uniform(0.996, 0.999, (8, 8)), requires_grad=True)

    def f(v):
        return m.flow_penalty(solver(v[None]), s)

    value = f(x)
    assert value > 0, "must exercise an active flow penalty"
    grad = torch.autograd.grad(value, x)[0]
    errors = []
    for _ in range(4):
        v = torch.tensor(rng.normal(size=(8, 8)))
        v /= v.norm()
        h = 1e-6
        fd = (f(x.detach() + h * v) - f(x.detach() - h * v)) / (2 * h)
        ad = (grad * v).sum()
        errors.append(
            abs((fd - ad).item()) / max(abs(fd.item()), abs(ad.item()), 1e-12)
        )
    assert max(errors) < 1e-6, errors


def test_warm_start_does_not_change_source_weights(tmp_path):
    m = module()
    from waveforge.lab.mcl.baselines import digest
    from waveforge.lab.mcl.stage5_decoder import ManifoldUNet

    torch.manual_seed(51)
    original = ManifoldUNet()
    path = tmp_path / "old.pt"
    torch.save(dict(model=original.state_dict()), path)
    sha = digest(path)
    loaded = m.load_model(path, "cpu")
    for a, b in zip(original.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        next(loaded.parameters()).add_(1)
    assert digest(path) == sha


def test_optimizer_records_both_grids_and_accounts_selection(tmp_path):
    import json
    from pathlib import Path

    from waveforge.lab.mcl.stage3_training import task_from_dict

    name = "waveforge.lab.mcl.stage51_evaluation"
    assert importlib.util.find_spec(name) is not None, "new optimizer missing"
    from waveforge.lab.mcl.stage51_evaluation import optimize

    task = task_from_dict(
        json.loads(
            Path(
                "artifacts/inputs/development_tasks.json"
            ).read_text()
        )["tasks"][0]
    )
    report = optimize(task, "Manifold-Adam", tmp_path / "optimizer", steps=3)
    assert report["evaluations"] == 3
    assert report["independent_binary_evaluations"] == 2 * len(report["checkpoints"])
    assert report["solver_counts"]["forward_calls"] == 6
    for rec in report["checkpoints"]:
        assert set(("coarse", "fine")) <= rec.keys()
        assert np.isfinite(rec["fine"]["peak_rise_k"])
    assert (tmp_path / "optimizer" / "selected.npz").exists()


def test_finetune_preserves_old_file_and_records_new_tasks(tmp_path):
    import json
    from concurrent.futures import ProcessPoolExecutor

    from waveforge.lab.mcl.baselines import digest
    from waveforge.lab.mcl.stage3_training import worker_init
    from waveforge.lab.mcl.stage5_decoder import ManifoldUNet

    name = "waveforge.lab.mcl.stage51_training"
    assert importlib.util.find_spec(name) is not None, "fine tuning missing"
    from waveforge.lab.mcl.stage51_training import train

    source = tmp_path / "old.pt"
    torch.save(dict(model=ManifoldUNet().state_dict()), source)
    sha = digest(source)
    out = tmp_path / "new"
    out.mkdir()
    with ProcessPoolExecutor(max_workers=1, initializer=worker_init) as pool:
        _, report = train(
            pool,
            out,
            (),
            initial_checkpoint=source,
            excluded_hashes={"never-sampled"},
            steps=3,
            batch=1,
        )
    assert digest(source) == sha
    assert report["unique_training_tasks"] == 3
    assert report["optimizer_updates"] == 3
    assert report["source_model_sha256"] == sha
    assert report["objective_solver_counts"]["forward_calls"] == 24
    assert len((out / "training_tasks.jsonl").read_text().splitlines()) == 3
    assert json.loads((out / "freeze.json").read_text())["model_sha256"] != sha


def test_audit_cannot_resurrect_diagnostic_that_failed_joint_selection():
    from waveforge.lab.mcl.stage5_admission import admission, system

    m = module()
    assert hasattr(m, "apply_selection_gate"), "audit selection gate missing"
    uniform = dict(
        row(55), method="Uniform-Parallel", task=0, regime="nominal", grid=64
    )
    diagnostic = dict(
        row(40), method="Manifold-UNet-R0", task=0, regime="nominal", grid=64
    )
    for r in (uniform, diagnostic):
        r["admission"] = admission(r)
        r["comparison_available"] = True
        r["system"] = system(r, uniform)
    assert diagnostic["system"]["system_gain_percent"] > 0
    m.apply_selection_gate([uniform, diagnostic], {"Manifold-UNet-R0": False})
    assert diagnostic["grid_admission"]["admissible"]
    assert not diagnostic["admission"]["admissible"]
    assert diagnostic["system"]["fallback_used"]
    assert diagnostic["system"]["system_gain_percent"] == 0
    assert diagnostic["system"]["raw_gain_percent"] is None


def test_short_classical_budget_cannot_enter_matched_comparison():
    m = module()
    assert hasattr(m, "require_complete_budget"), "matched budget gate missing"
    with pytest.raises(RuntimeError, match="budget"):
        m.require_complete_budget(dict(evaluations=599), 600)
    m.require_complete_budget(dict(evaluations=600), 600)


def test_full_audit_report_is_written_once(tmp_path):
    from waveforge.experiments import run_mcl_stage51 as runner
    from waveforge.lab.mcl.stage5_admission import admission, system

    assert hasattr(runner, "write_audit_report"), "single audit report writer missing"
    rows = []
    for grid in (64, 128):
        for regime in ("nominal", "power", "flow15", "flow16.7"):
            for task in range(8):
                uniform = row(55)
                for method in (
                    "Uniform-Parallel",
                    "Manifold-Adam",
                    "Manifold-MMA",
                    "Manifold-UNet-R0",
                    "Manifold-UNet-R25-C",
                ):
                    r = dict(
                        row(55 if method == "Uniform-Parallel" else 50),
                        grid=grid,
                        regime=regime,
                        task=task,
                        method=method,
                        pressure_pa=10000,
                        flow_ml_s=10,
                        comparison_available=True,
                    )
                    r["admission"] = admission(r)
                    r["system"] = system(r, uniform)
                    rows.append(r)
    runner.write_audit_report(tmp_path, rows, smoke=False)
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "metrics.json").exists()
