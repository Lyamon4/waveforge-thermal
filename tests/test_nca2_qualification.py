from __future__ import annotations

from dataclasses import replace

import pytest

from waveforge.ml.nca2_qualification import (
    classify_development_seed,
    select_nca2_protocol,
    summarize_protocol,
)


def _seed(
    protocol_id: str,
    seed: int,
    *,
    final_peak: float = 0.17,
    best_peak: float = 0.17,
    binary_fraction: float = 0.25,
    numerically_valid: bool = True,
    connectivity_pass: bool = True,
):
    return classify_development_seed(
        protocol_id=protocol_id,
        seed=seed,
        checkpoint_peaks=(best_peak, best_peak, best_peak, best_peak, final_peak),
        binary_fraction=binary_fraction,
        numerically_valid=numerically_valid,
        connectivity_pass=connectivity_pass,
    )


def _protocol(
    protocol_id: str,
    *,
    stable_count: int = 3,
    degradation: float = 0.0,
    peak: float = 0.17,
):
    seeds = []
    for index, seed in enumerate((20260901, 20260902, 20260903)):
        stable = index < stable_count
        best = peak / (1.0 + degradation) if degradation > 0.0 else peak
        seeds.append(
            _seed(
                protocol_id,
                seed,
                final_peak=peak if stable else 1.2 * peak,
                best_peak=best if stable else peak,
                binary_fraction=0.25 if stable else 0.27,
            )
        )
    return summarize_protocol(protocol_id, tuple(seeds))


def test_thermally_stable_disconnected_seed_remains_stable() -> None:
    metrics = classify_development_seed(
        protocol_id="A",
        seed=20260901,
        checkpoint_peaks=(0.18, 0.17, 0.171, 0.172, 0.1785),
        binary_fraction=0.25,
        numerically_valid=True,
        connectivity_pass=False,
    )

    assert metrics.late_best_tmax == 0.17
    assert metrics.late_degradation == pytest.approx(0.05)
    assert metrics.stable is True
    assert metrics.engineering_connectivity_pass is False
    assert metrics.reason_codes == ()


@pytest.mark.parametrize(
    ("binary_fraction", "numerically_valid", "final_peak", "reason"),
    [
        (0.2600001, True, 0.17, "BINARY_BUDGET_FAILURE"),
        (0.25, False, 0.17, "NUMERICAL_INVALIDITY"),
        (0.25, True, 0.19, "LATE_DEGRADATION_FAILURE"),
    ],
)
def test_development_seed_failures_are_explicit(
    binary_fraction: float,
    numerically_valid: bool,
    final_peak: float,
    reason: str,
) -> None:
    metrics = classify_development_seed(
        protocol_id="A",
        seed=20260901,
        checkpoint_peaks=(0.17, 0.17, 0.17, 0.17, final_peak),
        binary_fraction=binary_fraction,
        numerically_valid=numerically_valid,
        connectivity_pass=True,
    )

    assert metrics.stable is False
    assert reason in metrics.reason_codes


def test_selection_prefers_more_stable_seeds() -> None:
    verdict = select_nca2_protocol(
        _protocol("A", stable_count=3, peak=0.18),
        _protocol("B", stable_count=2, peak=0.16),
    )

    assert verdict.selected_protocol == "A"
    assert verdict.selection_reason == "MORE_STABLE_DEVELOPMENT_SEEDS"


def test_selection_prefers_lower_median_late_degradation() -> None:
    verdict = select_nca2_protocol(
        _protocol("A", degradation=0.02, peak=0.17),
        _protocol("B", degradation=0.01, peak=0.18),
    )

    assert verdict.selected_protocol == "B"
    assert verdict.selection_reason == "LOWER_MEDIAN_LATE_DEGRADATION"


def test_selection_prefers_lower_median_then_worst_final_peak() -> None:
    protocol_a = _protocol("A", peak=0.17)
    protocol_b = _protocol("B", peak=0.16)
    verdict = select_nca2_protocol(protocol_a, protocol_b)
    assert verdict.selected_protocol == "B"
    assert verdict.selection_reason == "LOWER_MEDIAN_FINAL_TMAX"

    tied_median_a = replace(
        protocol_a,
        median_final_tmax=0.16,
        worst_final_tmax=0.18,
    )
    tied_median_b = replace(
        protocol_b,
        median_final_tmax=0.16,
        worst_final_tmax=0.17,
    )
    verdict = select_nca2_protocol(tied_median_a, tied_median_b)
    assert verdict.selected_protocol == "B"
    assert verdict.selection_reason == "LOWER_WORST_FINAL_TMAX"


def test_full_practical_tie_selects_protocol_b() -> None:
    verdict = select_nca2_protocol(
        _protocol("A", peak=0.17),
        _protocol("B", peak=0.17001),
    )

    assert verdict.selected_protocol == "B"
    assert verdict.selection_reason == "PRACTICAL_TIE_FAVORS_DECAY"


def test_ineligible_or_insufficient_protocol_stops_production() -> None:
    ineligible_a = _protocol("A")
    ineligible_a = replace(ineligible_a, eligible=False)
    insufficient_b = _protocol("B", stable_count=1)

    verdict = select_nca2_protocol(ineligible_a, insufficient_b)

    assert verdict.status == "NCA2_QUALIFICATION_FAIL"
    assert verdict.production_authorized is False
    assert verdict.selected_protocol is None
