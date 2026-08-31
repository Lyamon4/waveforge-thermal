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
