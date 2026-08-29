"""Проверки eligibility и deterministic selection learning rate."""

from __future__ import annotations

from dataclasses import replace

import pytest

from waveforge.ml.nca_qualification import (
    QualificationReason,
    evaluate_lr_eligibility,
    select_learning_rate,
)
from waveforge.ml.nca_training import NCAIterationRecord


def _record(iteration: int, objective: float = 0.8) -> NCAIterationRecord:
    return NCAIterationRecord(
        iteration=iteration,
        total_objective=objective,
        thermal_smooth=objective,
        exact_continuous_tmax=objective,
        total_variation=0.1,
        binarization_penalty=0.1,
        continuous_material_fraction=0.25,
        binary_material_fraction=0.25,
        projection_absolute_error=1.0e-8,
        material_logit_mean=0.0,
        material_logit_std=0.1,
        material_logit_minimum=-0.2,
        material_logit_maximum=0.2,
        material_std=0.02,
        hidden_state_rms=0.1,
        delta_state_rms=0.01,
        maximum_absolute_delta=0.05,
        maximum_absolute_state=1.0,
        gradient_norm_before_clipping=0.2,
        gradient_norm_after_clipping=0.2,
        conv3x3_weight_gradient_norm=1.0e-3 if iteration >= 1 else 0.0,
        conv1x1_weight_gradient_norm=1.0e-3,
        all_parameter_gradients_finite=True,
        maximum_cg_iterations=100,
        maximum_explicit_relative_residual=9.0e-7,
        all_cg_converged=True,
        finite=True,
        wall_seconds=1.0,
    )


def _records(objectives: list[float] | None = None) -> tuple[NCAIterationRecord, ...]:
    values = objectives if objectives is not None else [0.8] * 200
    return tuple(_record(index, value) for index, value in enumerate(values))


def test_fast_early_learning_remains_eligible_when_early_late_is_flat() -> None:
    result = evaluate_lr_eligibility(
        learning_rate=1.0e-3,
        initial_objective=1.0,
        records=_records(),
    )

    assert result.early_loss == pytest.approx(0.8)
    assert result.late_loss == pytest.approx(0.8)
    assert result.objective_learning_fraction == pytest.approx(0.2)
    assert result.relative_improvement == pytest.approx(0.0)
    assert result.eligible is True
    assert result.reason_codes == ()


@pytest.mark.parametrize(
    ("records", "initial_objective", "reason"),
    [
        (_records()[:199], 1.0, QualificationReason.INCOMPLETE_QUALIFICATION_RUN),
        (
            tuple(
                replace(record, projection_absolute_error=2.0e-6)
                if record.iteration == 50
                else record
                for record in _records()
            ),
            1.0,
            QualificationReason.INVALID_VOLUME_PROJECTION,
        ),
        (
            tuple(
                replace(record, conv3x3_weight_gradient_norm=0.0)
                for record in _records()
            ),
            1.0,
            QualificationReason.MISSING_UPSTREAM_GRADIENT,
        ),
        (_records(), 0.8, QualificationReason.INSUFFICIENT_OBJECTIVE_LEARNING),
        (
            tuple(replace(record, material_std=5.0e-4) for record in _records()),
            1.0,
            QualificationReason.DESIGN_REMAINS_NEAR_UNIFORM,
        ),
    ],
)
def test_eligibility_failures_are_explicit(
    records: tuple[NCAIterationRecord, ...],
    initial_objective: float,
    reason: QualificationReason,
) -> None:
    result = evaluate_lr_eligibility(
        learning_rate=1.0e-3,
        initial_objective=initial_objective,
        records=records,
    )

    assert result.eligible is False
    assert reason in result.reason_codes


def test_selection_uses_primary_then_late_then_smaller_lr_ties() -> None:
    base = evaluate_lr_eligibility(1.0e-3, 1.0, _records())
    candidates = (
        replace(
            base,
            learning_rate=3.0e-4,
            relative_improvement=0.10000,
            late_loss=0.70000,
        ),
        replace(
            base,
            learning_rate=1.0e-3,
            relative_improvement=0.10005,
            late_loss=0.70002,
        ),
        replace(
            base,
            learning_rate=3.0e-3,
            relative_improvement=0.08,
            late_loss=0.6,
        ),
    )

    verdict = select_learning_rate(candidates)

    assert verdict.selected_learning_rate == 3.0e-4
    assert verdict.qualification_status == "PASS"
    assert "SMALLER_LEARNING_RATE" in verdict.selection_reason
    by_lr = {candidate.learning_rate: candidate for candidate in verdict.candidates}
    assert by_lr[3.0e-4].primary_score_delta == pytest.approx(5.0e-5)
    assert by_lr[1.0e-3].primary_score_delta == pytest.approx(0.0)
    assert by_lr[3.0e-3].primary_score_delta == pytest.approx(0.02005)


def test_no_eligible_lr_stops_before_production() -> None:
    incomplete = evaluate_lr_eligibility(3.0e-4, 1.0, _records()[:199])
    verdict = select_learning_rate(
        (
            incomplete,
            replace(incomplete, learning_rate=1.0e-3),
            replace(incomplete, learning_rate=3.0e-3),
        )
    )

    assert verdict.qualification_status == "NCA_QUALIFICATION_NO_ELIGIBLE_LR"
    assert verdict.umbrella_spike_status == "NCA_SPIKE_INVALID_TRAINING_PATHOLOGY"
    assert verdict.production_started is False
    assert verdict.selected_learning_rate is None
