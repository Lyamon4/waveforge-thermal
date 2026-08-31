from __future__ import annotations

import json
from pathlib import Path

import pytest

from waveforge.experiments.benchmark_mt2b import (
    BatchMeasurement,
    FixedOperatorMeasurement,
    build_benchmark_report,
    main,
    project_paired_runtime,
)


def test_runtime_projection_uses_measured_tasks_per_second_and_locked_exposures() -> (
    None
):
    projection = project_paired_runtime(
        raw_tasks_per_second=4.0,
        physics_tasks_per_second=2.0,
        validation_seconds=320.0,
        reference_seconds=640.0,
    )

    assert projection.raw_training_hours == pytest.approx(8000 / 4 / 3600)
    assert projection.physics_training_hours == pytest.approx(8000 / 2 / 3600)
    assert projection.validation_hours == pytest.approx(320 / 3600)
    assert projection.reference_hours == pytest.approx(640 / 3600)
    assert projection.total_hours == pytest.approx((2000 + 4000 + 320 + 640) / 3600)


def test_benchmark_report_is_fail_closed_and_declares_sealed_tests() -> None:
    batch = (
        BatchMeasurement(
            mode="vectorized",
            variant="RAW",
            batch_size=4,
            median_seconds_per_update=2.0,
            tasks_per_second=2.0,
            peak_memory_bytes=100,
            agreement_pass=True,
        ),
    )
    fixed = FixedOperatorMeasurement(
        rhs_count=12,
        ordinary_seconds=0.12,
        reusable_seconds=0.03,
        speedup=4.0,
        maximum_absolute_error=1e-12,
        maximum_relative_error=1e-12,
        agreement_pass=True,
    )

    report = build_benchmark_report(
        batch_measurements=batch,
        fixed_operator=fixed,
        environment={"device": "NVIDIA A100-SXM4-40GB"},
        runtime_projection=project_paired_runtime(
            raw_tasks_per_second=2.0,
            physics_tasks_per_second=1.5,
            validation_seconds=0.0,
            reference_seconds=0.0,
        ),
    )

    assert report["schema_version"] == 1
    assert report["status"] == "PASS"
    assert report["long_training_started"] is False
    assert report["test_id_accessed"] is False
    assert report["test_ood_accessed"] is False


def test_benchmark_report_selects_passing_fallback_not_failed_fast_candidate() -> None:
    measurements = (
        BatchMeasurement(
            mode="vectorized",
            variant="RAW",
            batch_size=4,
            median_seconds_per_update=0.8,
            tasks_per_second=5.0,
            peak_memory_bytes=800,
            agreement_pass=False,
            gradient_relative_l2_error=4.0e-6,
        ),
        BatchMeasurement(
            mode="vectorized",
            variant="PHYSICS",
            batch_size=4,
            median_seconds_per_update=0.9,
            tasks_per_second=4.4,
            peak_memory_bytes=800,
            agreement_pass=False,
            gradient_relative_l2_error=4.0e-6,
        ),
        BatchMeasurement(
            mode="scenario_vectorized_sequential",
            variant="RAW",
            batch_size=4,
            median_seconds_per_update=3.0,
            tasks_per_second=4 / 3,
            peak_memory_bytes=400,
            agreement_pass=True,
            gradient_relative_l2_error=0.0,
        ),
        BatchMeasurement(
            mode="scenario_vectorized_sequential",
            variant="PHYSICS",
            batch_size=4,
            median_seconds_per_update=3.1,
            tasks_per_second=4 / 3.1,
            peak_memory_bytes=400,
            agreement_pass=True,
            gradient_relative_l2_error=3.3e-10,
        ),
    )
    fixed = FixedOperatorMeasurement(
        rhs_count=12,
        ordinary_seconds=0.49,
        reusable_seconds=0.003,
        speedup=161.0,
        maximum_absolute_error=3.0e-14,
        maximum_relative_error=3.0e-14,
        agreement_pass=True,
    )

    report = build_benchmark_report(
        batch_measurements=measurements,
        fixed_operator=fixed,
        environment={"device": "NVIDIA A100-SXM4-40GB"},
        runtime_projection=project_paired_runtime(
            raw_tasks_per_second=4 / 3,
            physics_tasks_per_second=4 / 3.1,
            validation_seconds=0.0,
            reference_seconds=0.0,
        ),
    )

    assert report["status"] == "PASS"
    assert report["selected_training_mode"] == "scenario_vectorized_sequential"
    assert report["rejected_training_modes"] == ["vectorized"]


def test_fixed_operator_only_cli_writes_benchmark_without_training(
    tmp_path: Path,
) -> None:
    output = tmp_path / "benchmark.json"

    exit_code = main(
        [
            "--output",
            str(output),
            "--device",
            "cpu",
            "--fixed-operator-only",
            "--quick",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["fixed_operator"]["agreement_pass"] is True
    assert payload["batch_measurements"] == []
    assert payload["long_training_started"] is False
    assert payload["test_id_accessed"] is False
    assert payload["test_ood_accessed"] is False
