"""Tests for locked Gate 2A nominal effect comparisons."""

import pytest

from waveforge.verification.compare import (
    Gate2Status,
    classify_nominal_seed,
    select_strongest_baseline,
)


def test_strongest_baseline_is_lowest_unrounded_verified_peak() -> None:
    """Display rounding or insertion order must not select the comparator."""
    selection = select_strongest_baseline(
        {
            "straight_path": 0.2500000002,
            "random_filtered_seed_9101": 0.2500000001,
            "single_A_20260828": 0.3,
        }
    )

    assert selection.candidate_id == "random_filtered_seed_9101"
    assert selection.worst_peak == pytest.approx(0.2500000001, abs=0.0)


def test_nominal_seed_passes_at_exact_five_percent_and_both_budgets() -> None:
    """The registered unrounded 5% boundary and ±0.01 budgets are inclusive."""
    verdict = classify_nominal_seed(
        candidate_peak=0.95,
        baseline_peaks={"straight_path": 1.0, "single_A_20260828": 1.1},
        continuous_fraction=0.24,
        binary_fraction=0.26,
        valid=True,
    )

    assert verdict.status is Gate2Status.PASS
    assert verdict.metrics["relative_improvement"] == pytest.approx(0.05)
    assert verdict.metrics["strongest_baseline_id"] == "straight_path"


def test_nominal_seed_reports_effect_and_budget_as_scientific_no_go() -> None:
    """A valid weak or over-budget design is NO_GO_EFFECT, not invalid."""
    weak = classify_nominal_seed(
        candidate_peak=0.951,
        baseline_peaks={"straight_path": 1.0},
        continuous_fraction=0.25,
        binary_fraction=0.25,
        valid=True,
    )
    over_budget = classify_nominal_seed(
        candidate_peak=0.90,
        baseline_peaks={"straight_path": 1.0},
        continuous_fraction=0.25,
        binary_fraction=0.261,
        valid=True,
    )

    assert weak.status is Gate2Status.NO_GO_EFFECT
    assert weak.reason_codes == ("NOMINAL_EFFECT_BELOW_THRESHOLD",)
    assert over_budget.status is Gate2Status.NO_GO_EFFECT
    assert over_budget.reason_codes == ("MATERIAL_BUDGET_FAILURE",)


@pytest.mark.parametrize(
    ("candidate_peak", "baseline_peaks"),
    [
        (float("nan"), {"straight_path": 1.0}),
        (0.9, {}),
        (0.9, {"straight_path": float("inf")}),
    ],
)
def test_nominal_numerical_or_registry_failure_is_invalid(
    candidate_peak: float,
    baseline_peaks: dict[str, float],
) -> None:
    """Corrupt mandatory metrics must not masquerade as a negative result."""
    verdict = classify_nominal_seed(
        candidate_peak=candidate_peak,
        baseline_peaks=baseline_peaks,
        continuous_fraction=0.25,
        binary_fraction=0.25,
        valid=True,
    )

    assert verdict.status is Gate2Status.INVALID_RUN
    assert verdict.reason_codes == ("NOMINAL_NUMERICAL_OR_REGISTRY_FAILURE",)


def test_explicit_invalidity_precedes_good_effect() -> None:
    """A solver/integrity failure forces INVALID_RUN even with a good number."""
    verdict = classify_nominal_seed(
        candidate_peak=0.5,
        baseline_peaks={"straight_path": 1.0},
        continuous_fraction=0.25,
        binary_fraction=0.25,
        valid=False,
    )

    assert verdict.status is Gate2Status.INVALID_RUN
