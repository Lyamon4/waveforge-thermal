"""Pure preregistered eligibility and ranking logic for NCA learning rates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from waveforge.ml.nca_training import NCAIterationRecord


class QualificationReason(StrEnum):
    INCOMPLETE_QUALIFICATION_RUN = "INCOMPLETE_QUALIFICATION_RUN"
    NONFINITE_QUALIFICATION_RUN = "NONFINITE_QUALIFICATION_RUN"
    INVALID_VOLUME_PROJECTION = "INVALID_VOLUME_PROJECTION"
    CG_NONCONVERGENCE = "CG_NONCONVERGENCE"
    MISSING_INITIAL_FINAL_LAYER_GRADIENT = "MISSING_INITIAL_FINAL_LAYER_GRADIENT"
    MISSING_UPSTREAM_GRADIENT = "MISSING_UPSTREAM_GRADIENT"
    UPDATE_BOUND_VIOLATION = "UPDATE_BOUND_VIOLATION"
    STATE_BOUND_VIOLATION = "STATE_BOUND_VIOLATION"
    INSUFFICIENT_OBJECTIVE_LEARNING = "INSUFFICIENT_OBJECTIVE_LEARNING"
    DESIGN_REMAINS_NEAR_UNIFORM = "DESIGN_REMAINS_NEAR_UNIFORM"
    INITIAL_MODEL_MISMATCH = "INITIAL_MODEL_MISMATCH"


@dataclass(frozen=True)
class LRQualification:
    learning_rate: float
    initial_objective: float | None
    record_count: int
    early_loss: float | None
    late_loss: float | None
    late_material_std: float | None
    objective_learning_fraction: float | None
    relative_improvement: float | None
    eligible: bool
    reason_codes: tuple[QualificationReason, ...]
    primary_score_delta: float | None = None
    late_loss_delta: float | None = None
    selected: bool = False


@dataclass(frozen=True)
class QualificationVerdict:
    qualification_status: str
    umbrella_spike_status: str | None
    production_authorized: bool
    production_started: bool
    selected_learning_rate: float | None
    selection_reason: str
    candidates: tuple[LRQualification, ...]


def _records_are_finite(records: Sequence[NCAIterationRecord]) -> bool:
    numeric_fields = (
        "total_objective",
        "thermal_smooth",
        "exact_continuous_tmax",
        "total_variation",
        "binarization_penalty",
        "continuous_material_fraction",
        "projection_absolute_error",
        "material_logit_mean",
        "material_logit_std",
        "material_logit_minimum",
        "material_logit_maximum",
        "material_std",
        "hidden_state_rms",
        "delta_state_rms",
        "maximum_absolute_delta",
        "maximum_absolute_state",
        "gradient_norm_before_clipping",
        "gradient_norm_after_clipping",
        "conv3x3_weight_gradient_norm",
        "conv1x1_weight_gradient_norm",
        "maximum_explicit_relative_residual",
        "wall_seconds",
    )
    return all(
        record.finite
        and record.all_parameter_gradients_finite
        and all(
            math.isfinite(float(getattr(record, field))) for field in numeric_fields
        )
        for record in records
    )


def evaluate_lr_eligibility(
    learning_rate: float,
    initial_objective: float | None,
    records: Sequence[NCAIterationRecord],
) -> LRQualification:
    """Evaluate every locked pathology gate without using result appearance."""
    reason_codes: list[QualificationReason] = []
    record_count = len(records)
    complete = record_count == 200 and [record.iteration for record in records] == list(
        range(200)
    )
    if not complete:
        reason_codes.append(QualificationReason.INCOMPLETE_QUALIFICATION_RUN)

    initial_finite = initial_objective is not None and math.isfinite(initial_objective)
    records_finite = _records_are_finite(records)
    if not initial_finite or not records_finite:
        reason_codes.append(QualificationReason.NONFINITE_QUALIFICATION_RUN)

    if any(
        abs(record.continuous_material_fraction - 0.25) > 1.0e-6
        or record.projection_absolute_error > 1.0e-6
        for record in records
    ):
        reason_codes.append(QualificationReason.INVALID_VOLUME_PROJECTION)
    if any(
        not record.all_cg_converged
        or record.maximum_explicit_relative_residual > 1.0e-6
        or record.maximum_cg_iterations > 2000
        for record in records
    ):
        reason_codes.append(QualificationReason.CG_NONCONVERGENCE)
    if complete and records[0].conv1x1_weight_gradient_norm <= 1.0e-12:
        reason_codes.append(QualificationReason.MISSING_INITIAL_FINAL_LAYER_GRADIENT)
    if (
        complete
        and max(record.conv3x3_weight_gradient_norm for record in records[1:6])
        <= 1.0e-12
    ):
        reason_codes.append(QualificationReason.MISSING_UPSTREAM_GRADIENT)
    if any(record.maximum_absolute_delta > 0.100001 for record in records):
        reason_codes.append(QualificationReason.UPDATE_BOUND_VIOLATION)
    if any(record.maximum_absolute_state > 6.4001 for record in records):
        reason_codes.append(QualificationReason.STATE_BOUND_VIOLATION)

    early_loss: float | None = None
    late_loss: float | None = None
    late_material_std: float | None = None
    learning_fraction: float | None = None
    relative_improvement: float | None = None
    if complete and initial_finite and records_finite:
        objectives = np.asarray(
            [record.total_objective for record in records], dtype=np.float64
        )
        material_stds = np.asarray(
            [record.material_std for record in records], dtype=np.float64
        )
        early_loss = float(np.median(objectives[20:40]))
        late_loss = float(np.median(objectives[180:200]))
        late_material_std = float(np.median(material_stds[180:200]))
        learning_fraction = (float(initial_objective) - late_loss) / max(
            abs(float(initial_objective)), 1.0e-12
        )
        relative_improvement = (early_loss - late_loss) / max(abs(early_loss), 1.0e-12)
        if learning_fraction < 0.01:
            reason_codes.append(QualificationReason.INSUFFICIENT_OBJECTIVE_LEARNING)
        if late_material_std < 1.0e-3:
            reason_codes.append(QualificationReason.DESIGN_REMAINS_NEAR_UNIFORM)

    return LRQualification(
        learning_rate=learning_rate,
        initial_objective=initial_objective,
        record_count=record_count,
        early_loss=early_loss,
        late_loss=late_loss,
        late_material_std=late_material_std,
        objective_learning_fraction=learning_fraction,
        relative_improvement=relative_improvement,
        eligible=not reason_codes,
        reason_codes=tuple(reason_codes),
    )


def select_learning_rate(
    results: Sequence[LRQualification],
) -> QualificationVerdict:
    """Apply exact unrounded primary and two-level tie-breaking rules."""
    if tuple(result.learning_rate for result in results) != (3.0e-4, 1.0e-3, 3.0e-3):
        raise ValueError("qualification candidates must use the locked order")
    eligible = [result for result in results if result.eligible]
    if not eligible:
        return QualificationVerdict(
            qualification_status="NCA_QUALIFICATION_NO_ELIGIBLE_LR",
            umbrella_spike_status="NCA_SPIKE_INVALID_TRAINING_PATHOLOGY",
            production_authorized=False,
            production_started=False,
            selected_learning_rate=None,
            selection_reason="NO_ELIGIBLE_LEARNING_RATE",
            candidates=tuple(results),
        )

    if any(
        result.relative_improvement is None or result.late_loss is None
        for result in eligible
    ):
        raise ValueError("eligible candidates require finite ranking metrics")
    maximum_score = max(float(result.relative_improvement) for result in eligible)
    primary_tied = [
        result
        for result in eligible
        if maximum_score - float(result.relative_improvement) <= 1.0e-4
    ]
    best_late_loss = min(float(result.late_loss) for result in primary_tied)
    late_tied = [
        result
        for result in primary_tied
        if abs(float(result.late_loss) - best_late_loss)
        <= 1.0e-4 * max(abs(best_late_loss), 1.0e-12)
    ]
    selected = min(late_tied, key=lambda result: result.learning_rate)
    if len(primary_tied) == 1:
        selection_reason = "HIGHEST_RELATIVE_IMPROVEMENT"
    elif len(late_tied) == 1:
        selection_reason = "PRIMARY_SCORE_TIE_BROKEN_BY_LOWER_LATE_LOSS"
    else:
        selection_reason = "PRIMARY_AND_LATE_LOSS_TIE_BROKEN_BY_SMALLER_LEARNING_RATE"

    enriched: list[LRQualification] = []
    primary_ids = {id(result) for result in primary_tied}
    for result in results:
        score_delta = (
            maximum_score - float(result.relative_improvement)
            if result.eligible and result.relative_improvement is not None
            else None
        )
        late_delta = (
            float(result.late_loss) - best_late_loss
            if id(result) in primary_ids and result.late_loss is not None
            else None
        )
        enriched.append(
            replace(
                result,
                primary_score_delta=score_delta,
                late_loss_delta=late_delta,
                selected=result.learning_rate == selected.learning_rate,
            )
        )
    return QualificationVerdict(
        qualification_status="PASS",
        umbrella_spike_status=None,
        production_authorized=True,
        production_started=False,
        selected_learning_rate=selected.learning_rate,
        selection_reason=selection_reason,
        candidates=tuple(enriched),
    )
