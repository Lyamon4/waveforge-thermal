"""Scientific reporting and artifact-integrity contracts for pure NCA."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from waveforge.reporting.nca_spike import (
    add_comparator_roles,
    build_scientific_verdict,
    plot_final_design_gallery,
    render_russian_report,
    write_artifact_hash_manifest,
)


def _verification_payload() -> dict:
    seeds = []
    for seed, peak, fraction, status in (
        (20260901, 0.36, 0.245, "NO_GO_EFFECT"),
        (20260902, 0.155, 0.25, "PASS"),
        (20260903, 0.47, 0.289, "NO_GO_EFFECT"),
    ):
        seeds.append(
            {
                "candidate_id": f"nca_{seed}",
                "relative_128_to_256_change": 0.003,
                "verdict": {
                    "seed": seed,
                    "status": status,
                    "peak_256": peak,
                    "binary_fraction": fraction,
                },
            }
        )
    return {
        "status": "NCA_NO_GO_EFFECT",
        "campaign": {
            "status": "NCA_NO_GO_EFFECT",
            "passing_seed_count": 1,
            "required_passing_seed_count": 2,
        },
        "seed_verifications": seeds,
        "verification_git_sha": "verification-sha",
    }


def test_comparator_table_keeps_scientific_roles_explicit() -> None:
    frame = pd.DataFrame(
        {
            "comparator_id": [
                "waveforge_20260828",
                "parametric_branching_tree",
                "straight_path",
            ]
        }
    )

    result = add_comparator_roles(frame)

    assert result["comparator_role"].tolist() == [
        "existing_pixel_inverse_design",
        "post_result_geometric_challenge",
        "original_simple_baseline",
    ]


def test_final_verdict_contains_all_provenance_layers() -> None:
    payload = build_scientific_verdict(
        qualification={"selected_learning_rate": 0.001},
        production=[{"seed": 20260901, "implementation_git_sha": "impl"}],
        verification=_verification_payload(),
        reproducibility={"mode": "strict_exact", "status": "PASS"},
        provenance={"config_sha256": "config", "spec_sha256": "spec"},
    )

    assert payload["status"] == "NCA_NO_GO_EFFECT"
    assert payload["qualification"]["selected_learning_rate"] == 0.001
    assert payload["production"][0]["implementation_git_sha"] == "impl"
    assert payload["verification"]["verification_git_sha"] == "verification-sha"
    assert payload["reproducibility"]["status"] == "PASS"
    assert payload["provenance"]["config_sha256"] == "config"


def test_plotting_does_not_mutate_design_arrays(tmp_path) -> None:
    continuous = {
        seed: np.linspace(0.0, 1.0, 4096).reshape(64, 64) for seed in range(3)
    }
    binary = {seed: (array >= 0.5).astype(float) for seed, array in continuous.items()}
    before_continuous = {seed: array.copy() for seed, array in continuous.items()}
    before_binary = {seed: array.copy() for seed, array in binary.items()}

    plot_final_design_gallery(continuous, binary, tmp_path / "gallery.png")

    assert (tmp_path / "gallery.png").is_file()
    assert all(
        np.array_equal(continuous[seed], before_continuous[seed]) for seed in continuous
    )
    assert all(np.array_equal(binary[seed], before_binary[seed]) for seed in binary)


def test_hash_manifest_covers_every_declared_artifact_except_itself(tmp_path) -> None:
    (tmp_path / "report.md").write_text("line1\r\nline2\r\n", encoding="utf-8")
    (tmp_path / "metrics.csv").write_text("x\r\n1\r\n", encoding="utf-8")
    (tmp_path / "figure.png").write_bytes(b"PNG")
    paths = [
        tmp_path / "report.md",
        tmp_path / "metrics.csv",
        tmp_path / "figure.png",
    ]

    output = write_artifact_hash_manifest(
        tmp_path / "artifact_hashes.json",
        paths,
        root=tmp_path,
    )
    stored = json.loads(output.read_text(encoding="utf-8"))

    assert set(stored["artifacts"]) == {"report.md", "metrics.csv", "figure.png"}
    assert "artifact_hashes.json" not in stored["artifacts"]
    assert stored["hash_mode"] == "canonical_lf_text_raw_binary"


def test_russian_report_avoids_unearned_claims() -> None:
    report = render_russian_report(
        build_scientific_verdict(
            qualification={"selected_learning_rate": 0.001},
            production=[],
            verification=_verification_payload(),
            reproducibility={"mode": "strict_exact", "status": "PASS"},
            provenance={},
        ),
        pd.DataFrame(),
    )

    assert "NCA_NO_GO_EFFECT" in report
    for forbidden in ("generalizes", "surrogate", "first", "industrial-ready"):
        assert forbidden not in report.lower()
