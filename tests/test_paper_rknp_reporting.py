from __future__ import annotations

import json
from pathlib import Path

import pytest

from waveforge.reporting.paper_rknp import (
    FIGURE_SPECS,
    build_paper_figure_pack,
    load_paper_evidence,
)
from waveforge.reproducibility import artifact_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_paper_evidence_preserves_locked_negative_nca2_result() -> None:
    """Catch rounded, cherry-picked, or relabelled NCA-2 evidence."""
    evidence = load_paper_evidence(PROJECT_ROOT)

    assert evidence.nca2_status == "NCA2_NO_GO_EFFECT"
    assert evidence.tree_peak_256 == pytest.approx(0.1650978093408512)
    assert evidence.nca2_peaks_256 == {
        20260911: pytest.approx(0.1893982971956948),
        20260912: pytest.approx(0.15483529959128456),
        20260913: pytest.approx(0.1611499978371461),
    }
    assert evidence.primary_passing_seeds == (20260912, 20260913)
    assert evidence.catastrophic_seed == 20260911


def test_paper_registry_is_complete_unique_and_claim_bounded() -> None:
    """Catch missing panels, duplicate filenames, or an unlabeled summary claim."""
    assert len(FIGURE_SPECS) == 18
    assert len({spec.figure_id for spec in FIGURE_SPECS}) == 18
    assert len({spec.stem for spec in FIGURE_SPECS}) == 18
    assert {spec.figure_id for spec in FIGURE_SPECS} == {
        "fig01_problem_setup",
        "fig02_waveforge_workflow",
        "fig03_nca_architecture",
        "fig04_solver_validation",
        "fig05_gate2_design_evolution",
        "fig06_strong_tree_baseline",
        "fig07_topology_comparison",
        "fig08_nca2_seed_gallery",
        "fig09_success_failure_anatomy",
        "fig10_nca_growth_rollout",
        "fig11_temperature_scenarios",
        "fig12_training_stability",
        "fig13_protocol_qualification",
        "fig14_performance_against_tree",
        "fig15_grid_transfer",
        "fig16_budget_connectivity",
        "fig17_research_timeline",
        "fig18_graphical_abstract",
    }
    graphical_abstract = next(
        spec for spec in FIGURE_SPECS if spec.figure_id == "fig18_graphical_abstract"
    )
    assert "NCA2_NO_GO_EFFECT" in graphical_abstract.claim_limit


def test_small_pack_build_writes_manifest_without_mutating_sources(
    tmp_path: Path,
) -> None:
    """Catch plotting that changes frozen arrays or omits output provenance."""
    source = (
        PROJECT_ROOT / "artifacts/nca2_stabilization/production_seed_20260912/"
        "design_binary_64.npy"
    )
    before = artifact_sha256(source)

    payload = build_paper_figure_pack(
        PROJECT_ROOT,
        tmp_path,
        formats=("png",),
        dpi=72,
        selected_figure_ids=("fig01_problem_setup", "fig14_performance_against_tree"),
    )

    assert artifact_sha256(source) == before
    assert payload["scientific_verdict"] == "NCA2_NO_GO_EFFECT"
    assert payload["figure_count"] == 2
    assert (tmp_path / "fig01_problem_setup.png").is_file()
    assert (tmp_path / "fig14_performance_against_tree.png").is_file()
    stored = json.loads((tmp_path / "figure_manifest.json").read_text())
    assert stored == payload
    assert set(payload["artifact_hashes"]) == {
        "fig01_problem_setup.png",
        "fig14_performance_against_tree.png",
        "FIGURE_GUIDE.md",
    }
