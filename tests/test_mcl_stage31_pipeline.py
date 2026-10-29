import importlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from waveforge.lab.epyc_tasks import sample_epyc_task
from waveforge.lab.mcl.stage3 import VascularUNet, anchor_parameters, binary


def module(suffix):
    name = "waveforge.lab.mcl.stage31_" + suffix
    assert importlib.util.find_spec(name) is not None, "Stage31 pipeline missing"
    return importlib.import_module(name)


def test_short_training_runs_all_phases_and_freezes(tmp_path):
    m = module("training")
    with ThreadPoolExecutor(max_workers=1) as pool:
        model, report = m.train(pool, tmp_path, (), steps=3, batch=1)
    assert report["optimizer_updates"] == 3
    assert report["unique_training_tasks"] == 3
    assert all(not p.requires_grad for p in model.parameters())
    history = [
        json.loads(line)
        for line in (tmp_path / "training_history.jsonl").read_text().splitlines()
    ]
    assert [r["phase"] for r in history] == [0, 1, 2]
    assert [r["hard_forward"] for r in history] == [False, False, True]
    assert history[-1]["learning_rate"] == pytest.approx(0.0001)
    frozen = VascularUNet()
    frozen.load_state_dict(
        torch.load(tmp_path / "model_frozen.pt", weights_only=True)["model"]
    )
    assert all(
        torch.equal(v, frozen.state_dict()[k]) for k, v in model.state_dict().items()
    )


def test_refinement_keeps_original_step_zero_despite_desaturation(tmp_path):
    m = module("evaluation")
    task = sample_epyc_task(np.random.default_rng(3101))
    initial = anchor_parameters()[0].numpy()
    initial[20:] = 3.0
    report = m.optimize(task, initial, tmp_path / "refine", budget=25, steps=2)
    with np.load(tmp_path / "refine/checkpoint_0000.npz") as z:
        np.testing.assert_array_equal(z["parameters"], initial)
        np.testing.assert_array_equal(
            z["binary"], binary(torch.from_numpy(initial)[None])[0]
        )
    assert report["selected"]["peak_rise_k"] <= report["checkpoints"][0]["peak_rise_k"]
    assert report["original_step0_retained"]
    assert report["initial_desaturation_max_change"] > 0
    assert report["steps"] == 2
    assert [r["step"] for r in report["checkpoints"]] == [0, 2]
    with pytest.raises(FileExistsError):
        m.optimize(task, initial, tmp_path / "refine", budget=25, steps=2)


def test_all_invalid_heads_never_start_r25(monkeypatch, tmp_path):
    m = module("evaluation")
    calls = []

    def rejected(*args, head=None, **kwargs):
        return dict(accepted=False, peak_rise_k=1.0, step=0, head=head)

    def control_only(*args, budget, **kwargs):
        calls.append(budget)
        assert budget == 600
        return dict(method="Vascular-Adam600-Decoupled", selected=None)

    monkeypatch.setattr(m, "record_candidate", rejected)
    monkeypatch.setattr(m, "optimize", control_only)
    task = sample_epyc_task(np.random.default_rng(3102))
    results = m.evaluate_task(task, anchor_parameters().numpy(), tmp_path / "task")
    assert results[1]["selected"] is None and results[1]["steps"] == 0
    assert calls == [600]
    assert not (tmp_path / "task/Vascular-UNet-S31-R25/selected.npz").exists()


def test_summary_does_not_rank_invalid_baseline_or_candidate(tmp_path):
    name = "waveforge.experiments.run_mcl_stage31"
    assert importlib.util.find_spec(name) is not None, "Stage31 runner missing"
    runner = importlib.import_module(name)
    from waveforge.lab.mcl.stage31_evaluation import R25

    rows = [
        dict(task=0, method=R25, accepted=True, peak_rise_k=90.0),
        dict(task=0, method="Uniform-Parallel", accepted=False, peak_rise_k=None),
        dict(task=0, method="Vascular-Adam", accepted=True, peak_rise_k=100.0),
        dict(task=1, method=R25, accepted=False, peak_rise_k=None),
        dict(task=1, method="Vascular-Adam", accepted=True, peak_rise_k=100.0),
    ]
    runner.write_comparison(tmp_path, rows)
    data = json.loads((tmp_path / "paired_metrics.json").read_text())
    assert data["Uniform-Parallel"]["paired_tasks"] == 0
    assert data["Uniform-Parallel"]["median_paired_gain_percent"] is None
    assert data["Vascular-Adam"]["paired_tasks"] == 1
    assert data["Vascular-Adam"]["wins"] == 1
    assert data["Vascular-Adam"]["reference_rejected"] == 0
    assert data["Vascular-Adam"]["candidate_rejected"] == 1
    assert data["Uniform-Parallel"]["reference_rejected"] == 1
    assert data["Uniform-Parallel"]["reference_missing"] == 1
    assert data["Vascular-Adam"]["median_paired_gain_percent"] == pytest.approx(10.0)
    assert "REJECTED" in (tmp_path / "SUMMARY_STAGE31.md").read_text()
    with pytest.raises(FileExistsError):
        runner.write_comparison(tmp_path, rows)


@pytest.mark.parametrize("broken", ["step_zero", "regression"])
def test_r25_contract_failure_is_runtime_error(monkeypatch, tmp_path, broken):
    m = module("evaluation")

    def accepted(*args, head=None, **kwargs):
        return dict(
            accepted=True,
            peak_rise_k=90.0,
            step=0,
            head=head,
            binary_sha256="original",
            file="head.npz",
        )

    def corrupt(*args, **kwargs):
        return dict(
            checkpoints=[
                dict(binary_sha256="changed" if broken == "step_zero" else "original")
            ],
            selected=dict(peak_rise_k=91.0 if broken == "regression" else 90.0),
        )

    monkeypatch.setattr(m, "record_candidate", accepted)
    monkeypatch.setattr(m, "optimize", corrupt)
    monkeypatch.setattr(m.shutil, "copyfile", lambda *args: None)
    task = sample_epyc_task(np.random.default_rng(3103))
    with pytest.raises(RuntimeError):
        m.evaluate_task(task, anchor_parameters().numpy(), tmp_path / "task")
