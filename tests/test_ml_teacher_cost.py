from __future__ import annotations

import json
import math

import pytest

from waveforge.experiments.assess_ml_teacher_cost import (
    PilotMeasurement,
    TeacherCostStatus,
    assess_teacher_cost,
    write_pre_dataset_gate_outcome,
)
from waveforge.ml.teacher import TeacherStatus


def _measurements(
    peaks_32: tuple[float, float, float] = (1.05, 2.10, 3.15),
    peaks_64: tuple[float, float, float] = (1.00, 2.00, 3.00),
    *,
    wall_32: float = 10.0,
    wall_64: float = 30.0,
    status: TeacherStatus = TeacherStatus.PASS,
) -> tuple[PilotMeasurement, ...]:
    records: list[PilotMeasurement] = []
    for index, (peak_32, peak_64) in enumerate(
        zip(peaks_32, peaks_64, strict=True),
        start=1,
    ):
        for resolution, peak, wall in (
            (32, peak_32, wall_32),
            (64, peak_64, wall_64),
        ):
            records.append(
                PilotMeasurement(
                    pilot_id=f"pilot_{index}",
                    resolution=resolution,
                    status=status,
                    wall_seconds=wall,
                    verified_peak_64=peak,
                    binary_fraction=0.25,
                    maximum_scipy_residual=1.0e-12,
                    artifact_bytes=1000 * resolution,
                    result_sha256=f"result-{index}-{resolution}",
                    binary_sha256=f"binary-{index}-{resolution}",
                )
            )
    return tuple(records)


def test_teacher_cost_passes_only_with_fidelity_cost_and_storage() -> None:
    result = assess_teacher_cost(
        _measurements(),
        base_spec_sha256="base",
        amendment_sha256="amendment",
    )

    assert result.status is TeacherCostStatus.PASS
    assert result.accepted_teacher_resolution == 32
    assert result.spearman_rank_correlation == pytest.approx(1.0)
    assert result.median_relative_degradation == pytest.approx(0.05)
    assert result.maximum_relative_degradation == pytest.approx(0.05)
    expected_seconds = 6 * 20.0 + 1.15 * (20 * 10.0 + 8 * 30.0)
    assert result.projected_teacher_hours == pytest.approx(expected_seconds / 3600)
    assert result.base_spec_sha256 == "base"
    assert result.amendment_sha256 == "amendment"


def test_fidelity_failure_stops_without_64_fallback() -> None:
    result = assess_teacher_cost(
        _measurements(peaks_32=(3.0, 2.0, 1.0)),
        base_spec_sha256="base",
        amendment_sha256="amendment",
    )

    assert result.status is TeacherCostStatus.ML_NO_GO_TEACHER_FIDELITY
    assert result.accepted_teacher_resolution is None
    assert result.spearman_rank_correlation == pytest.approx(-1.0)
    assert "RANKING_NOT_PRESERVED" in result.reason_codes


def test_cost_failure_is_separate_from_fidelity_failure() -> None:
    result = assess_teacher_cost(
        _measurements(wall_32=1800.0, wall_64=3600.0),
        base_spec_sha256="base",
        amendment_sha256="amendment",
    )

    assert result.status is TeacherCostStatus.ML_NO_GO_TEACHER_COST
    assert result.projected_teacher_hours > 8.0
    assert result.reason_codes == ("TEACHER_WALLCLOCK_EXCEEDS_8_HOURS",)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda record: {**record, "status": TeacherStatus.INVALID_RUN},
        lambda record: {**record, "verified_peak_64": math.nan},
        lambda record: {**record, "maximum_scipy_residual": 1.0e-8},
    ],
)
def test_invalid_numerics_never_look_like_scientific_no_go(mutator) -> None:
    records = [measurement.__dict__ for measurement in _measurements()]
    records[0] = mutator(records[0])
    result = assess_teacher_cost(
        tuple(PilotMeasurement(**record) for record in records),
        base_spec_sha256="base",
        amendment_sha256="amendment",
    )

    assert result.status is TeacherCostStatus.INVALID_RUN
    assert result.accepted_teacher_resolution is None


def test_incomplete_or_duplicate_pilot_matrix_is_invalid() -> None:
    records = _measurements()
    missing = assess_teacher_cost(
        records[:-1],
        base_spec_sha256="base",
        amendment_sha256="amendment",
    )
    duplicate = assess_teacher_cost(
        (*records[:-1], records[0]),
        base_spec_sha256="base",
        amendment_sha256="amendment",
    )

    assert missing.status is TeacherCostStatus.INVALID_RUN
    assert duplicate.status is TeacherCostStatus.INVALID_RUN


def test_fidelity_no_go_writes_honest_stop_without_placeholder_results(
    tmp_path,
) -> None:
    assessment = assess_teacher_cost(
        _measurements(peaks_32=(1.11, 2.24, 3.30)),
        base_spec_sha256="base",
        amendment_sha256="amendment",
    )
    report_path = tmp_path / "teacher_cost_report.json"
    report_path.write_text("{}\n", encoding="utf-8")

    outcome_path = write_pre_dataset_gate_outcome(tmp_path, assessment)

    assert outcome_path == tmp_path / "ml_verdict.json"
    verdict = json.loads(outcome_path.read_text(encoding="utf-8"))
    break_even = json.loads(
        (tmp_path / "break_even_analysis.json").read_text(encoding="utf-8")
    )
    assert verdict["verdict"] == "ML_NO_GO"
    assert verdict["teacher_gate_status"] == "ML_NO_GO_TEACHER_FIDELITY"
    assert verdict["dataset_generation_started"] is False
    assert verdict["network_training_started"] is False
    assert break_even["status"] == "NOT_COMPUTABLE_PRE_DATASET_GATE_FAILED"
    assert break_even["number_of_design_tasks_to_break_even"] is None
    assert not (tmp_path / "training_curves.csv").exists()
    assert not (tmp_path / "initializer_comparison.csv").exists()


def test_passing_teacher_gate_writes_authorization_not_ml_verdict(tmp_path) -> None:
    assessment = assess_teacher_cost(
        _measurements(),
        base_spec_sha256="base",
        amendment_sha256="amendment",
    )
    (tmp_path / "teacher_cost_report.json").write_text("{}\n", encoding="utf-8")

    outcome_path = write_pre_dataset_gate_outcome(tmp_path, assessment)

    assert outcome_path == tmp_path / "teacher_gate_authorization.json"
    assert not (tmp_path / "ml_verdict.json").exists()
    assert not (tmp_path / "break_even_analysis.json").exists()
