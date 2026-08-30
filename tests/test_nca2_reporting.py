"""Scientific reporting contracts for NCA-2."""

from __future__ import annotations

from waveforge.reporting.nca2 import render_nca2_report


def test_report_never_hides_failed_first_experiment_or_any_production_seed() -> None:
    qualification = {
        "selected_protocol": "B",
        "selection_reason": "PRACTICAL_TIE_FAVORS_DECAY",
    }
    verdict = {
        "campaign": {
            "status": "NCA2_STABILITY_GO",
            "mean_peak_256": 0.158,
            "median_peak_256": 0.157,
            "minimum_peak_256": 0.156,
            "maximum_peak_256": 0.161,
        },
        "seeds": [
            {
                "seed": seed,
                "verdict": {
                    "peak_256": peak,
                    "binary_fraction": 0.25,
                    "tree_improvement": 0.03,
                    "primary_pass": True,
                },
                "engineering_connectivity_pass": seed != 20260913,
            }
            for seed, peak in (
                (20260911, 0.156),
                (20260912, 0.157),
                (20260913, 0.161),
            )
        ],
    }

    report = render_nca2_report(
        qualification=qualification,
        verdict=verdict,
        implementation_shas=("a" * 40,),
        report_git_sha="b" * 40,
    )

    assert "NCA_NO_GO_EFFECT" in report
    assert "20260911" in report
    assert "20260912" in report
    assert "20260913" in report
    assert "0.158" in report
    assert "Protocol B" in report
    assert "unseen" in report
    assert "aaaaaaaa" in report
    assert "bbbbbbbb" in report
