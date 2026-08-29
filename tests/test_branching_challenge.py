"""Tests for independent strong branching-baseline verification."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from waveforge.design.branching_baseline import (
    BranchingTreeParameters,
    build_branching_tree,
    iter_candidate_parameters,
)
from waveforge.experiments.run_branching_challenge import (
    SearchRecord,
    SearchResult,
    run_comparison,
    run_search,
    write_challenge_figures,
)
from waveforge.verification import challenge as challenge_module
from waveforge.verification.challenge import (
    ChallengeEvaluation,
    ChallengeSeedComparison,
    ChallengeStatus,
    classify_challenge,
    evaluate_frozen_binary_design,
)
from waveforge.verification.high_fidelity import verify_candidate
from waveforge.verification.perturbations import (
    MorphologyRecord,
    PerturbationCase,
    PerturbationEvaluation,
    registered_primary_cases,
)


@pytest.mark.parametrize("resolution", [64, 128])
def test_reusable_factorization_evaluator_matches_public_verifier(
    resolution: int,
) -> None:
    """Wrong transfer, assembly or RHS replacement must fail."""
    design = build_branching_tree(BranchingTreeParameters(0.5, 0.5, 0.3, 1.0)).design
    actual = evaluate_frozen_binary_design("tree", design, resolution=resolution)
    fidelity = "low_64" if resolution == 64 else "reference_128"
    expected = verify_candidate("tree", design, fidelity=fidelity)
    expected_peaks = [record.peak_temperature for record in expected.scenario_records]
    np.testing.assert_allclose(
        actual.scenario_peaks,
        expected_peaks,
        rtol=0.0,
        atol=1e-12,
    )
    assert actual.worst_peak == pytest.approx(expected.worst_peak, abs=1e-12)
    assert actual.maximum_residual <= 1e-10
    assert actual.material_fraction == 0.25


def test_evaluator_factorizes_once_for_three_source_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accidentally refactorizing for each RHS must fail."""
    calls = 0
    original = challenge_module.factorize_system

    def counting_factorization(system: object) -> object:
        nonlocal calls
        calls += 1
        return original(system)  # type: ignore[arg-type]

    monkeypatch.setattr(challenge_module, "factorize_system", counting_factorization)
    design = build_branching_tree(BranchingTreeParameters(0.5, 0.5, 0.3, 1.0)).design
    evaluate_frozen_binary_design("tree", design, resolution=64)
    assert calls == 1


def _comparison(
    seed: int,
    nominal_improvement: float,
    robustness_passing_cases: int,
) -> ChallengeSeedComparison:
    return ChallengeSeedComparison(
        seed=seed,
        nominal_improvement=nominal_improvement,
        robustness_passing_cases=robustness_passing_cases,
    )


@pytest.mark.parametrize(
    ("comparisons", "valid", "expected"),
    [
        (
            (
                _comparison(1, 0.06, 23),
                _comparison(2, 0.05, 28),
                _comparison(3, 0.01, 28),
            ),
            True,
            ChallengeStatus.STRONG_CHALLENGE_PASS,
        ),
        (
            (
                _comparison(1, 0.04, 28),
                _comparison(2, 0.03, 22),
                _comparison(3, 0.01, 28),
            ),
            True,
            ChallengeStatus.CHALLENGE_COMPARABLE,
        ),
        (
            (
                _comparison(1, -0.01, 28),
                _comparison(2, -1e-12, 28),
                _comparison(3, 0.08, 28),
            ),
            True,
            ChallengeStatus.CHALLENGE_FAIL,
        ),
        (
            (
                _comparison(1, math.nan, 28),
                _comparison(2, 0.06, 28),
                _comparison(3, 0.06, 28),
            ),
            True,
            ChallengeStatus.INVALID_RUN,
        ),
    ],
)
def test_challenge_verdict_uses_locked_precedence(
    comparisons: tuple[ChallengeSeedComparison, ...],
    valid: bool,
    expected: ChallengeStatus,
) -> None:
    """Reordering invalid/fail/strong/comparable precedence must fail."""
    assert classify_challenge(comparisons, valid=valid).status is expected


def test_search_funnel_reranks_all_then_twenty_then_five(
    tmp_path: Path,
) -> None:
    """Pruning early or choosing the low-fidelity winner must fail."""
    parameters = tuple(iter_candidate_parameters())[:24]
    index_by_id = {
        parameter.candidate_id: index for index, parameter in enumerate(parameters)
    }
    calls: list[tuple[str, int]] = []

    def fake_evaluator(
        candidate_id: str,
        design: np.ndarray,
        *,
        resolution: int,
    ) -> ChallengeEvaluation:
        calls.append((candidate_id, resolution))
        index = index_by_id[candidate_id]
        if resolution == 64:
            peak = float(index + 1)
        elif resolution == 128:
            peak = float(20 - index)
        else:
            peak = float(abs(index - 15) + 1)
        return ChallengeEvaluation(
            candidate_id=candidate_id,
            resolution=resolution,
            design_hash_64=f"design-{index}",
            transferred_design_hash=f"transfer-{resolution}-{index}",
            material_fraction=float(design.mean()),
            scenario_peaks=(peak, peak - 0.1, peak - 0.2),
            scenario_residuals=(1e-12, 1e-12, 1e-12),
            worst_peak=peak,
            average_peak=peak - 0.1,
            maximum_residual=1e-12,
            wall_seconds=0.01,
        )

    result = run_search(
        tmp_path,
        parameters=parameters,
        evaluator=fake_evaluator,
    )

    assert Counter(resolution for _, resolution in calls) == {
        64: 24,
        128: 20,
        256: 5,
    }
    assert len(result.search_64) == 24
    assert len(result.finalists_128) == 20
    assert len(result.finalists_256) == 5
    assert result.winner.parameters.candidate_id == parameters[15].candidate_id
    for filename, count in (
        ("tree_search_64.csv", 24),
        ("tree_finalists_128.csv", 20),
        ("tree_finalists_256.csv", 5),
    ):
        rows = (tmp_path / filename).read_text(encoding="utf-8").splitlines()
        assert len(rows) == count + 1


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_comparison_writes_three_nominal_and_eighty_four_robustness_rows(
    tmp_path: Path,
) -> None:
    """Using the wrong denominator, seed or perturbation count must fail."""
    output_dir = tmp_path / "challenge"
    output_dir.mkdir()
    (output_dir / "challenge_spec.md").write_text("locked challenge\n")
    gate2_root = tmp_path / "gate2"
    verification_dir = gate2_root / "verification"
    verification_dir.mkdir(parents=True)
    seeds = (20260828, 20260829, 20260830)
    waveforge_nominal = (0.90, 0.94, 1.01)
    _write_rows(
        verification_dir / "nominal_binary_verification.csv",
        [
            {
                "candidate_id": f"robust_{seed}",
                "fidelity": "reference_256",
                "worst_peak": peak,
                "material_fraction": 0.25,
            }
            for seed, peak in zip(seeds, waveforge_nominal, strict=True)
        ],
    )
    cases = registered_primary_cases()
    (verification_dir / "perturbation_registry.json").write_text(
        json.dumps({"cases": [asdict(case) for case in cases]}),
        encoding="utf-8",
    )
    _write_rows(
        verification_dir / "perturbation_evaluations.csv",
        [
            {
                "candidate_id": f"robust_{seed}",
                "case_id": case.case_id,
                "worst_peak": 0.90,
                "material_fraction": 0.25,
            }
            for seed in seeds
            for case in cases
        ],
    )
    _write_rows(
        verification_dir / "morphology_metrics.csv",
        [
            {
                "candidate_id": f"robust_{seed}",
                "operation": operation,
                "material_fraction": 0.25,
                "worst_peak_256": 0.90,
                "component_count": 1,
                "relative_degradation": 0.0,
                "design_hash_64": f"waveforge-{seed}",
            }
            for seed in seeds
            for operation in ("unperturbed", "erosion", "dilation")
        ],
    )

    parameters = BranchingTreeParameters(0.5, 0.5, 0.3, 1.0)
    winner_evaluation = ChallengeEvaluation(
        candidate_id=parameters.candidate_id,
        resolution=256,
        design_hash_64="tree-hash",
        transferred_design_hash="tree-256-hash",
        material_fraction=0.25,
        scenario_peaks=(1.0, 0.9, 0.8),
        scenario_residuals=(1e-12, 1e-12, 1e-12),
        worst_peak=1.0,
        average_peak=0.9,
        maximum_residual=1e-12,
        wall_seconds=0.01,
    )
    winner = SearchRecord(parameters, winner_evaluation, rank=1)
    search = SearchResult(
        search_64=(winner,) * 24,
        finalists_128=(winner,) * 20,
        finalists_256=(winner,) * 5,
        winner=winner,
    )

    def fake_perturbation(
        candidate_id: str,
        frozen_design_64: np.ndarray,
        case: PerturbationCase,
    ) -> PerturbationEvaluation:
        return PerturbationEvaluation(
            candidate_id=candidate_id,
            case_id=case.case_id,
            grid_shape=(256, 256),
            design_hash_64="tree-hash",
            material_fraction=float(frozen_design_64.mean()),
            scenario_peaks=(1.0, 0.9, 0.8),
            scenario_residuals=(1e-12, 1e-12, 1e-12),
            worst_peak=1.0,
            k_high=case.k_high,
            wall_seconds=0.01,
        )

    def fake_morphology(
        candidate_id: str,
        frozen_design_64: np.ndarray,
    ) -> tuple[MorphologyRecord, ...]:
        return tuple(
            MorphologyRecord(
                candidate_id=candidate_id,
                operation=operation,  # type: ignore[arg-type]
                material_fraction=float(frozen_design_64.mean()),
                worst_peak_256=1.0,
                component_count=1,
                relative_degradation=0.0,
                design_hash_64="tree-hash",
            )
            for operation in ("unperturbed", "erosion", "dilation")
        )

    verdict = run_comparison(
        output_dir,
        search,
        gate2_root=gate2_root,
        perturbation_evaluator=fake_perturbation,
        morphology_evaluator=fake_morphology,
        enforce_production_counts=False,
    )

    nominal_rows = list(csv.DictReader((output_dir / "waveforge_vs_tree.csv").open()))
    robustness_rows = list(
        csv.DictReader((output_dir / "challenge_robustness.csv").open())
    )
    assert len(nominal_rows) == 3
    assert len(robustness_rows) == 84
    assert float(nominal_rows[0]["relative_improvement"]) == pytest.approx(0.10)
    assert all(float(row["tree_peak"]) == 1.0 for row in robustness_rows)
    assert all(
        float(row["relative_improvement"]) == pytest.approx(0.10)
        for row in robustness_rows
    )
    assert verdict.status is ChallengeStatus.STRONG_CHALLENGE_PASS
    registry = json.loads((output_dir / "candidate_registry.json").read_text())
    assert registry["post_result_challenge"] is True
    assert registry["candidate_count"] == 41055
    assert registry["search_counts"] == {"64": 24, "128": 20, "256": 5}


def test_figures_do_not_mutate_metrics_and_verdict_hashes_final_artifacts(
    tmp_path: Path,
) -> None:
    """Plot mutation or an incomplete artifact manifest must fail."""
    output_dir = tmp_path / "challenge"
    output_dir.mkdir()
    final_names = (
        "challenge_spec.md",
        "candidate_registry.json",
        "tree_search_64.csv",
        "tree_finalists_128.csv",
        "tree_finalists_256.csv",
        "waveforge_vs_tree.csv",
        "challenge_robustness.csv",
        "challenge_morphology.csv",
    )
    for name in final_names:
        (output_dir / name).write_text(f"fixture:{name}\n", encoding="utf-8")
    (output_dir / "challenge_verdict.json").write_text(
        json.dumps({"status": "CHALLENGE_COMPARABLE"}),
        encoding="utf-8",
    )
    parameters = BranchingTreeParameters(0.5, 0.5, 0.3, 1.0)
    evaluation = ChallengeEvaluation(
        candidate_id=parameters.candidate_id,
        resolution=256,
        design_hash_64="tree-hash",
        transferred_design_hash="tree-256-hash",
        material_fraction=0.25,
        scenario_peaks=(1.0, 0.9, 0.8),
        scenario_residuals=(1e-12, 1e-12, 1e-12),
        worst_peak=1.0,
        average_peak=0.9,
        maximum_residual=1e-12,
        wall_seconds=0.01,
    )
    winner = SearchRecord(parameters, evaluation, rank=1)
    search = SearchResult((winner,), (winner,), (winner,), winner)
    metric_snapshot = evaluation

    def plotting_evaluator(
        candidate_id: str,
        design: np.ndarray,
        *,
        resolution: int,
        include_temperature_fields: bool = False,
    ) -> ChallengeEvaluation:
        fields = tuple(
            np.full((resolution, resolution), value, dtype=np.float64)
            for value in (1.0, 0.9, 0.8)
        )
        return ChallengeEvaluation(
            candidate_id=candidate_id,
            resolution=resolution,
            design_hash_64="tree-hash",
            transferred_design_hash="tree-256-hash",
            material_fraction=float(design.mean()),
            scenario_peaks=(1.0, 0.9, 0.8),
            scenario_residuals=(1e-12, 1e-12, 1e-12),
            worst_peak=1.0,
            average_peak=0.9,
            maximum_residual=1e-12,
            wall_seconds=0.01,
            temperature_fields=fields if include_temperature_fields else None,
        )

    write_challenge_figures(output_dir, search, evaluator=plotting_evaluator)

    assert search.winner.evaluation == metric_snapshot
    for name in ("best_tree_design.png", "best_tree_temperature_maps.png"):
        assert (output_dir / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    verdict = json.loads((output_dir / "challenge_verdict.json").read_text())
    expected_hash_names = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "challenge_verdict.json"
    }
    assert set(verdict["artifact_hashes"]) == expected_hash_names
