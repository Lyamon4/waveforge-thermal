import numpy as np
import pytest


def test_manifest_seal_rejects_changed_content(tmp_path):
    from waveforge.lab.mcl.stage4_inventory import seal_files, verify_seal

    path = tmp_path / "frozen.bin"
    path.write_bytes(b"original")
    seal = seal_files([path])
    verify_seal(seal)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_seal(seal)


def test_pairwise_gain_is_median_of_taskwise_percent_not_ratio_of_medians():
    from waveforge.lab.mcl.stage4_metrics import paired_gain

    rows = []
    for task, (a, b) in enumerate([(10, 8), (100, 95), (1000, 500)]):
        rows.extend(
            [
                dict(task=task, method="reference", peak_rise_k=a, numerical_ok=True),
                dict(task=task, method="candidate", peak_rise_k=b, numerical_ok=True),
            ]
        )
    got = paired_gain(rows, "candidate", "reference")
    assert got["median_gain_percent"] == 20 and got["paired_tasks"] == 3
    rows[1]["numerical_ok"] = False
    got = paired_gain(rows, "candidate", "reference")
    assert got["paired_tasks"] == 2 and got["invalid_or_missing_pairs"] == 1
    assert got["median_gain_percent"] == 27.5


def test_rank_comparison_pairs_identical_methods_per_task_and_handles_ties():
    from waveforge.lab.mcl.stage4_metrics import rank_agreement

    original = [
        dict(task=0, method=m, peak_rise_k=x, numerical_ok=True)
        for m, x in zip("abc", [1, 2, 3], strict=True)
    ]
    opposite = [
        dict(task=0, method=m, peak_rise_k=x, numerical_ok=True)
        for m, x in zip("abc", [3, 2, 1], strict=True)
    ]
    got = rank_agreement(original, opposite)
    assert got[0]["spearman"] == pytest.approx(-1) and got[0][
        "kendall_tau_b"
    ] == pytest.approx(-1)


def test_actual_evaluation_record_is_json_serializable(tmp_path):
    import json
    from pathlib import Path

    from waveforge.experiments.run_mcl_stage4 import evaluate_entry
    from waveforge.lab.mcl.contracts import stage12_spec, straight_channel
    from waveforge.lab.mcl.stage3_training import task_from_dict

    selection = Path(
        "artifacts/inputs/development_tasks.json"
    )
    task = task_from_dict(json.loads(selection.read_text(encoding="utf-8"))["tasks"][0])
    path = tmp_path / "frozen.npz"
    np.savez(path, binary=straight_channel(stage12_spec()))
    row = evaluate_entry(
        dict(task=0, method="Straight", path=str(path)),
        task,
        dict(h_air=10.0, pressure_pa=2500.0),
    )
    assert json.loads(json.dumps(row, allow_nan=False))["numerical_ok"] is True


def test_limited_field_reports_and_gates_actual_equation_residual():
    from waveforge.experiments.run_mcl_stage4 import numerical_summary

    diag = dict(
        pressure_residual=1e-15,
        compensated_pressure_residual=1e-15,
        thermal_residual=[1e-14],
        compensated_thermal_residual=[1e-15],
        mass_relative_error=1e-13,
        energy_relative_error=[1e-13],
        transport_converged=True,
        transport_equation_residual=9e-11,
    )
    row = numerical_summary(diag, 0.0)
    assert row["numerical_ok"]
    assert row["max_linear_system_residual"] == 1e-14
    assert row["max_residual"] == 9e-11
    diag["transport_equation_residual"] = 2e-10
    assert not numerical_summary(diag, 0.0)["numerical_ok"]


def test_original_re_gate_never_substitutes_local_cell_estimator():
    from waveforge.experiments.run_mcl_stage4 import reynolds_summary

    diag = [dict(max_local_reynolds_estimate=600.0, local_cell_re_variable_mu=300.0)]
    row = reynolds_summary(diag)
    assert row["max_local_re"] == 600.0
    assert row["max_cell_re_variable_mu"] == 300.0
    assert not row["within_original_re_gate"]
