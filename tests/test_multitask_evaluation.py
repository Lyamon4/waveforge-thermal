"""Tests for frozen multi-task NCA validation policy."""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.ml.multitask_evaluation import (
    ValidationSummary,
    condition_causality_summary,
    pairwise_binary_diversity,
    select_validation_checkpoint,
    summarize_against_reference,
)


def summary(
    step: int,
    *,
    median: float,
    p90: float,
    invalid: int = 0,
) -> ValidationSummary:
    return ValidationSummary(
        completed_updates=step,
        split_name="validation",
        task_count=32,
        invalid_count=invalid,
        median_peak=median,
        p90_peak=p90,
        worst_peak=0.5,
        median_relative_gap=median,
        p90_relative_gap=p90,
        worst_relative_gap=0.5,
    )


def test_checkpoint_selection_uses_median_p90_invalid_count_then_earlier() -> None:
    selected = select_validation_checkpoint(
        [
            summary(250, median=0.20, p90=0.24),
            summary(500, median=0.19, p90=0.23),
            summary(750, median=0.19, p90=0.23),
        ]
    )
    assert selected.completed_updates == 500


def test_checkpoint_selection_rejects_nonvalidation_or_nonfinite_summary() -> None:
    wrong_split = ValidationSummary(
        completed_updates=250,
        split_name="test_id",
        task_count=32,
        invalid_count=0,
        median_peak=0.1,
        p90_peak=0.2,
        worst_peak=0.3,
        median_relative_gap=0.1,
        p90_relative_gap=0.2,
        worst_relative_gap=0.3,
    )
    with pytest.raises(ValueError, match="validation"):
        select_validation_checkpoint([wrong_split])
    with pytest.raises(ValueError, match="finite"):
        select_validation_checkpoint([summary(250, median=float("nan"), p90=0.2)])


def test_condition_causality_requires_23_of_32_matched_wins() -> None:
    result = condition_causality_summary(
        matched=[0.1] * 23 + [0.3] * 9,
        shuffled=[0.2] * 32,
    )
    assert result.task_count == 32
    assert result.matched_win_count == 23
    assert result.pass_gate is True


def test_condition_causality_counts_ties_as_not_wins() -> None:
    result = condition_causality_summary(
        matched=[0.1] * 22 + [0.2] * 10,
        shuffled=[0.2] * 32,
    )
    assert result.matched_win_count == 22
    assert result.pass_gate is False


def test_validation_summary_uses_unrounded_paired_relative_gaps() -> None:
    result = summarize_against_reference(
        completed_updates=500,
        split_name="validation",
        task_ids=("a", "b", "c"),
        candidate_peaks=(0.9, 1.1, 1.2),
        reference_peaks={"a": 1.0, "b": 1.0, "c": 1.0},
    )
    assert result.task_count == 3
    assert result.invalid_count == 0
    assert result.median_peak == pytest.approx(1.1)
    assert result.p90_peak == pytest.approx(1.18)
    assert result.median_relative_gap == pytest.approx(0.1)
    assert result.p90_relative_gap == pytest.approx(0.18)
    assert result.worst_relative_gap == pytest.approx(0.2)


def test_pairwise_binary_diversity_reports_hamming_and_jaccard() -> None:
    first = np.array([[1, 1], [0, 0]], dtype=np.float64)
    second = np.array([[1, 0], [1, 0]], dtype=np.float64)
    result = pairwise_binary_diversity([first, second])

    assert result.pair_count == 1
    assert result.mean_hamming_fraction == pytest.approx(0.5)
    assert result.mean_jaccard_similarity == pytest.approx(1.0 / 3.0)


def test_pairwise_binary_diversity_rejects_nonbinary_designs() -> None:
    with pytest.raises(ValueError, match="binary"):
        pairwise_binary_diversity([np.array([[0.5]]), np.array([[1.0]])])
