"""Tests for paired unseen-layout campaign statistics."""

from __future__ import annotations

import pytest

from waveforge.verification.multitask_verification import (
    GeneralizationSeedSummary,
    MultitaskCampaignStatus,
    bootstrap_median_interval,
    classify_campaign,
    relative_gap,
    summarize_seed,
)


def test_negative_gap_means_nca_is_better() -> None:
    assert relative_gap(nca_peak=0.18, gradient_peak=0.20) == pytest.approx(-0.10)


def test_bootstrap_interval_is_deterministic() -> None:
    gaps = (-0.10, -0.05, 0.0, 0.02, 0.03, 0.04)
    first = bootstrap_median_interval(gaps, seed=2026083151, resamples=2000)
    second = bootstrap_median_interval(gaps, seed=2026083151, resamples=2000)
    assert first == second
    assert first.lower <= first.median <= first.upper


def test_seed_summary_uses_paired_unrounded_values() -> None:
    result = summarize_seed(
        seed=2026083102,
        nca_peaks=(0.18, 0.21, 0.19, 0.20),
        gradient_peaks=(0.20, 0.20, 0.20, 0.20),
        bootstrap_seed=17,
        bootstrap_resamples=1000,
        condition_matched_wins=25,
        valid=True,
    )
    assert result.task_count == 4
    assert result.median_gap == pytest.approx(-0.025)
    assert result.p90_gap == pytest.approx(0.035)
    assert result.win_rate == pytest.approx(0.5)
    assert result.primary_seed_pass is True


def passing_seed(seed: int) -> GeneralizationSeedSummary:
    return GeneralizationSeedSummary(
        seed=seed,
        valid=True,
        task_count=32,
        median_gap=0.01,
        p90_gap=0.08,
        worst_gap=0.2,
        win_count=10,
        win_rate=10 / 32,
        bootstrap_median_lower=-0.01,
        bootstrap_median_upper=0.03,
        condition_matched_wins=25,
        primary_seed_pass=True,
        better_tested_gradient=False,
    )


def failing_seed(seed: int) -> GeneralizationSeedSummary:
    return GeneralizationSeedSummary(
        seed=seed,
        valid=True,
        task_count=32,
        median_gap=0.06,
        p90_gap=0.2,
        worst_gap=0.4,
        win_count=2,
        win_rate=2 / 32,
        bootstrap_median_lower=0.03,
        bootstrap_median_upper=0.1,
        condition_matched_wins=20,
        primary_seed_pass=False,
        better_tested_gradient=False,
    )


def test_primary_go_requires_two_of_three_registered_seeds() -> None:
    result = classify_campaign(
        [passing_seed(2026083102), passing_seed(2026083103), failing_seed(2026083104)]
    )
    assert result.status is MultitaskCampaignStatus.MULTITASK_NCA_GO
    assert result.passing_seed_count == 2


def test_any_invalid_seed_has_precedence_over_scientific_effect() -> None:
    invalid = passing_seed(2026083104)
    invalid = GeneralizationSeedSummary(**(invalid.__dict__ | {"valid": False}))
    result = classify_campaign(
        [passing_seed(2026083102), passing_seed(2026083103), invalid]
    )
    assert result.status is MultitaskCampaignStatus.INVALID_RUN
