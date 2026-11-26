import numpy as np
import pytest
import torch


def test_parallel_baseline_exact_budget_connected():
    from waveforge.lab.mcl.baselines import uniform_parallel
    from waveforge.lab.mcl.vascular import connectivity

    d = uniform_parallel()
    assert d.sum() == 1024
    assert connectivity(d)["all_cells_port_connected"]


@pytest.mark.parametrize("family", ["pixel", "vascular"])
def test_baseline_representation_gradients_and_exact_binary(family):
    from waveforge.lab.mcl.baselines import (
        binary_design,
        continuous_design,
        initial_parameters,
    )

    values = torch.tensor(
        initial_parameters(family, 0), dtype=torch.float64, requires_grad=True
    )
    design = continuous_design(values, family, beta=2.0)
    assert abs(float(design.mean().detach()) - 0.25) < 1e-6
    weight = torch.arange(4096, dtype=torch.float64).reshape(64, 64) / 4096
    gradient = torch.autograd.grad((design * weight).mean(), values)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    binary = binary_design(values, family)
    assert binary.sum() == 1024
    if family == "vascular":
        from waveforge.lab.mcl.vascular import connectivity

        assert connectivity(binary)["all_cells_port_connected"]
        assert gradient[:16].norm() > 0


def test_fail_closed_qualification_manifest(tmp_path):
    import json

    from waveforge.lab.mcl.baselines import check_qualification

    p = tmp_path / "qualification.json"
    p.write_text(json.dumps({"verdict": {"baseline_launch_allowed": False}}))
    with pytest.raises(ValueError, match="qualification"):
        check_qualification(p, tmp_path)


def test_accepted_checkpoint_selection_excludes_hot_flow_failure():
    from waveforge.lab.mcl.baselines import best_checkpoint

    rows = [
        dict(step=0, peak_rise_k=40.0, accepted=False),
        dict(step=50, peak_rise_k=80.0, accepted=True),
        dict(step=100, peak_rise_k=80.0, accepted=True),
    ]
    assert best_checkpoint(rows)["step"] == 50
    assert best_checkpoint(rows[:1]) is None


@pytest.mark.parametrize(
    "method", ["Pixel-Adam", "Pixel-MMA", "Vascular-Adam", "Vascular-MMA"]
)
def test_two_step_real_physics_baseline_smoke(tmp_path, method):
    from waveforge.lab.mcl import MCLProblem
    from waveforge.lab.mcl.baselines import optimize
    from waveforge.lab.mcl.contracts import stage12_spec

    torch.set_num_threads(1)
    chip = np.zeros((64, 64), bool)
    chip[26:38, 26:38] = True
    power = np.zeros((3, 64, 64))
    power[:, chip] = 360 / chip.sum()
    report = optimize(
        MCLProblem(stage12_spec(), chip, power), method, 0, tmp_path / "run", steps=2
    )
    assert report["completed"] and report["steps"] == 2
    assert report["counts"] == dict(
        objective_evaluations=2,
        optimizer_updates=2 if method.endswith("Adam") else None,
        forward_rhs=8,
        adjoint_rhs=8,
        independent_rhs=8,
    )
    assert len(report["history"]) == 2
    assert report["final"]["geometry"]["cells"] == 1024


def test_rejected_summary_has_no_selected_winner():
    from waveforge.experiments.run_mcl_stage2 import summary_row

    record = dict(
        accepted=False,
        peak_rise_k=40.0,
        step=600,
        flow_ml_s=1.0,
        diagnostics=dict(max_local_reynolds_estimate=600.0, air_heat_fraction=[0.2]),
    )
    row = summary_row(0, "bad", record, 3.0)
    assert row["selected_step"] is None and row["peak_rise_k"] is None
    assert row["diagnostic_final_peak_rise_k"] == 40.0


def test_resume_reuses_only_matching_immutable_completed_methods(tmp_path):
    import json

    from waveforge.experiments.run_mcl_stage2 import completed_method, mark_completed

    folder = tmp_path / "method"
    folder.mkdir()
    (folder / "result.json").write_text(json.dumps({"completed": True}))
    mark_completed(folder, "abc")
    assert completed_method(folder, "abc") == {"completed": True}
    with pytest.raises(ValueError, match="provenance"):
        completed_method(folder, "wrong")
    (folder / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        completed_method(folder, "abc")


def test_mma_early_termination_is_preserved(monkeypatch, tmp_path):
    import nlopt

    from waveforge.lab.mcl import MCLProblem
    from waveforge.lab.mcl.baselines import optimize
    from waveforge.lab.mcl.contracts import stage12_spec

    original = nlopt.opt

    def early_opt(*args):
        class Early:
            def __init__(self):
                self.opt = original(*args)

            def __getattr__(self, name):
                return getattr(self.opt, name)

            def set_maxeval(self, _count):
                self.opt.set_maxeval(1)

        return Early()

    monkeypatch.setattr(nlopt, "opt", early_opt)
    torch.set_num_threads(1)
    chip = np.zeros((64, 64), bool)
    chip[26:38, 26:38] = True
    power = np.zeros((3, 64, 64))
    power[:, chip] = 360 / chip.sum()
    report = optimize(
        MCLProblem(stage12_spec(), chip, power),
        "Pixel-MMA",
        0,
        tmp_path / "early",
        steps=2,
    )
    assert report["completed"] and report["terminated_early"] and report["steps"] == 1
    assert report["counts"]["optimizer_updates"] is None
