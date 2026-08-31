from __future__ import annotations

import numpy as np
import pytest

from waveforge.ml.mt2b_evaluation import (
    MT2BCheckpointSummary,
    evaluate_solver_consistent_gaps,
    independent_scipy64_tmax,
    paired_bootstrap,
    select_mt2b_checkpoint,
)
from waveforge.ml.multitask_tasks import sample_primary_task


def _design(column: int) -> np.ndarray:
    design = np.zeros((64, 64), dtype=np.float64)
    flat = design.ravel()
    selected = (np.arange(1024) * 3 + column) % flat.size
    flat[selected] = 1.0
    return design


def test_solver_consistent_gaps_score_both_methods_through_same_callable() -> None:
    tasks = tuple(sample_primary_task(2026083141, index) for index in range(2))
    candidate = {task.task_id: _design(index) for index, task in enumerate(tasks)}
    reference = {task.task_id: _design(index + 2) for index, task in enumerate(tasks)}
    calls: list[tuple[str, str]] = []

    def scipy64(design: np.ndarray, task) -> float:
        calls.append((task.task_id, str(np.sum(design * np.arange(64)[None, :]))))
        return 0.1 + 1.0e-5 * float(np.sum(design * np.arange(64)[None, :]))

    result = evaluate_solver_consistent_gaps(
        candidate,
        reference,
        tasks,
        scipy64_evaluator=scipy64,
    )

    assert result.solver_id == "independent_scipy_64"
    assert len(result.rows) == 2
    assert len(calls) == 4
    assert [task_id for task_id, _ in calls] == [
        tasks[0].task_id,
        tasks[0].task_id,
        tasks[1].task_id,
        tasks[1].task_id,
    ]
    for row in result.rows:
        assert row.relative_gap == pytest.approx(
            (row.candidate_tmax - row.reference_tmax) / row.reference_tmax
        )


def test_independent_scipy64_evaluator_scores_exact_binary_design() -> None:
    task = sample_primary_task(2026083141, 0)

    peak = independent_scipy64_tmax(_design(0), task)

    assert np.isfinite(peak)
    assert peak > 0.0


def test_solver_consistent_gaps_rejects_missing_or_invalid_binary_designs() -> None:
    task = sample_primary_task(2026083141, 0)
    valid = {task.task_id: _design(0)}
    missing: dict[str, np.ndarray] = {}
    invalid = {task.task_id: np.full((64, 64), 0.25)}

    with pytest.raises(KeyError, match="missing"):
        evaluate_solver_consistent_gaps(
            missing, valid, (task,), scipy64_evaluator=lambda _d, _t: 1.0
        )
    with pytest.raises(ValueError, match="binary"):
        evaluate_solver_consistent_gaps(
            invalid, valid, (task,), scipy64_evaluator=lambda _d, _t: 1.0
        )


def _summary(
    update: int,
    median_gap: float,
    p90_gap: float,
    median_tmax: float,
    *,
    invalid_count: int = 0,
) -> MT2BCheckpointSummary:
    return MT2BCheckpointSummary(
        completed_updates=update,
        split_name="validation",
        task_count=32,
        invalid_count=invalid_count,
        median_relative_gap=median_gap,
        p90_relative_gap=p90_gap,
        worst_relative_gap=0.5,
        median_absolute_tmax=median_tmax,
    )


def test_mt2b_checkpoint_selection_uses_locked_gap_first_precedence() -> None:
    summaries = [
        _summary(250, 0.10, 0.30, 0.15),
        _summary(500, 0.09, 0.50, 0.20),
        _summary(750, 0.09, 0.20, 0.22),
        _summary(1000, 0.09, 0.20, 0.21),
        _summary(1250, 0.09, 0.20, 0.21),
        _summary(1500, 0.01, 0.01, 0.10, invalid_count=1),
    ]

    selected = select_mt2b_checkpoint(summaries)

    assert selected.completed_updates == 1000


def test_mt2b_checkpoint_selection_rejects_non_validation_or_no_eligible_rows() -> None:
    wrong_split = _summary(250, 0.1, 0.2, 0.2)
    object.__setattr__(wrong_split, "split_name", "test_id")
    with pytest.raises(ValueError, match="validation"):
        select_mt2b_checkpoint([wrong_split])
    with pytest.raises(ValueError, match="eligible"):
        select_mt2b_checkpoint([_summary(250, 0.1, 0.2, 0.2, invalid_count=1)])


def test_paired_bootstrap_uses_exact_seed_resamples_statistic_and_percentiles() -> None:
    raw = np.linspace(0.10, 0.41, 32, dtype=np.float64)
    physics = raw - np.linspace(0.01, 0.04, 32, dtype=np.float64)

    result = paired_bootstrap(raw, physics)
    repeated = paired_bootstrap(raw, physics)

    assert result == repeated
    assert result.resamples == 10000
    assert result.seed == 2026092203
    assert result.statistic == "median"
    assert result.median_paired_delta == pytest.approx(0.025)
    assert result.lower_bound > 0.0
    assert result.conditioning_ci_pass is True

    rng = np.random.default_rng(2026092203)
    deltas = raw - physics
    indices = rng.integers(0, 32, size=(10000, 32))
    medians = np.median(deltas[indices], axis=1)
    expected = np.percentile(medians, [2.5, 97.5])
    assert result.lower_bound == pytest.approx(expected[0])
    assert result.upper_bound == pytest.approx(expected[1])


def test_paired_bootstrap_rejects_non_32_or_nonfinite_pairs() -> None:
    with pytest.raises(ValueError, match="32"):
        paired_bootstrap(np.ones(31), np.ones(31))
    invalid = np.ones(32)
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap(invalid, np.ones(32))
