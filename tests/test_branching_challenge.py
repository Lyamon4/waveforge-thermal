"""Tests for independent strong branching-baseline verification."""

from __future__ import annotations

import math

import numpy as np
import pytest

from waveforge.design.branching_baseline import (
    BranchingTreeParameters,
    build_branching_tree,
)
from waveforge.verification import challenge as challenge_module
from waveforge.verification.challenge import (
    ChallengeSeedComparison,
    ChallengeStatus,
    classify_challenge,
    evaluate_frozen_binary_design,
)
from waveforge.verification.high_fidelity import verify_candidate


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
