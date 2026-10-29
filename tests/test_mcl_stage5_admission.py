"""Inherited Stage5 engineering gates must not disappear behind numerical gates."""

from copy import deepcopy

import pytest


def candidate(peak=50.0, re=800.0, water=40.0):
    return dict(
        accepted=True,
        peak_rise_k=peak,
        silicon_peak_c=peak + 25,
        step=0,
        diagnostics={
            "max_local_reynolds_estimate": re,
            "peak_open_water_temperature_c": water,
            "pressure_residual": 1e-14,
            "thermal_residual": [1e-14],
            "compensated_pressure_residual": 1e-14,
            "compensated_thermal_residual": [1e-14],
            "mass_relative_error": 1e-13,
            "energy_relative_error": [1e-13],
        },
    )


@pytest.mark.parametrize(
    "key,value", [("re", 1200.0), ("water", 60.0), ("peak", 70.001)]
)
def test_engineering_failure_cannot_win(key, value):
    from waveforge.lab.mcl.stage5_admission import admission, select, system

    bad = candidate(**{key: value})
    good = candidate(55.0)
    assert not admission(bad)["admissible"]
    assert select([bad, good]) is good
    assert system(bad, good)["fallback_used"]
    assert system(bad, good)["system_peak_rise_k"] == 55.0


def test_boundaries_and_nan_fail_closed():
    from waveforge.lab.mcl.stage5_admission import admission

    assert admission(candidate(70.0, 1199.99, 59.99))["admissible"]
    for key in ("re", "water", "peak"):
        assert not admission(candidate(**{key: float("nan")}))["admissible"]


def test_missing_diagnostics_and_numerical_failure_rejected():
    from waveforge.lab.mcl.stage5_admission import admission

    bad = deepcopy(candidate())
    bad["diagnostics"].pop("peak_open_water_temperature_c")
    assert not admission(bad)["admissible"]
    bad = candidate()
    bad["accepted"] = False
    assert not admission(bad)["admissible"]


def test_no_valid_candidate_does_not_become_fake_win():
    from waveforge.lab.mcl.stage5_admission import select, system

    bad = candidate(10.0, re=1500.0)
    assert select([bad]) is None
    r = system(bad, candidate(55.0))
    assert r["raw_gain_percent"] is None
    assert r["system_gain_percent"] == 0.0


def test_old_conservation_gate_is_not_relaxed():
    from waveforge.lab.mcl.stage5_admission import admission

    bad = candidate()
    bad["diagnostics"]["mass_relative_error"] = 2e-10
    assert not admission(bad)["admissible"]


def test_report_does_not_count_wins_against_inadmissible_uniform(tmp_path):
    import json

    from waveforge.experiments.audit_mcl_stage5_admission import METHODS, summarize
    from waveforge.lab.mcl.stage5_admission import admission
    from waveforge.lab.mcl.stage5_evaluation import REGIMES

    rows = []
    for grid in (64, 128):
        for regime in REGIMES:
            for task in range(8):
                for method in ("Uniform-Parallel", *METHODS):
                    uniform = method == "Uniform-Parallel"
                    row = candidate(55 if uniform else 40, re=1500 if uniform else 800)
                    row.update(
                        task=task,
                        grid=grid,
                        regime=regime,
                        method=method,
                        comparison_available=False,
                        system=None,
                        flow_ml_s=15.0,
                        pressure_pa=15000.0,
                    )
                    row["admission"] = admission(row)
                    rows.append(row)
    summarize(tmp_path, rows)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["128_flow15"][METHODS[-1]]["raw_wins_vs_uniform"] is None
