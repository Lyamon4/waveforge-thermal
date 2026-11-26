"""Blocking directional-gradient tests for the complete Gate 2A pipeline."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from waveforge.design.gradient_validation import (
    GradientCheckRecord,
    GradientValidationConfig,
    GradientValidationReport,
    direction_for_seed,
    validate_full_pipeline_gradient,
    write_gradient_validation_artifacts,
)


def test_gradient_protocol_uses_exact_registered_directions_and_steps() -> None:
    """Changing a seed, step, or tolerance after inspection must fail."""
    config = GradientValidationConfig()

    assert config.direction_seeds == (7201, 7202, 7203, 7204, 7205)
    assert config.cpu_steps == (1e-2, 3e-3, 1e-3, 3e-4)
    assert config.cuda_steps == (1e-2, 3e-3, 1e-3)
    assert config.cpu_tolerance == 1e-4
    assert config.cuda_tolerance == 5e-3

    directions = [
        direction_for_seed(seed, dtype=torch.float64, device=torch.device("cpu"))
        for seed in config.direction_seeds
    ]
    for direction in directions:
        assert torch.linalg.vector_norm(direction).item() == pytest.approx(
            1.0,
            abs=1e-14,
        )
    hashes = {direction.numpy().tobytes() for direction in directions}
    assert len(hashes) == 5


def _assert_two_adjacent_steps_pass_per_direction(report: object) -> None:
    records = report.records  # type: ignore[attr-defined]
    for seed in (7201, 7202, 7203, 7204, 7205):
        passing = [record.passed for record in records if record.direction_seed == seed]
        assert any(first and second for first, second in pairwise(passing))


def test_complete_cpu_float64_gradient_pipeline() -> None:
    """Any broken stage from logits through adjoint must fail on CPU."""
    report = validate_full_pipeline_gradient(
        GradientValidationConfig(),
        device=torch.device("cpu"),
        design_dtype=torch.float64,
    )

    assert report.passed
    assert len(report.records) == 20
    assert {record.dtype for record in report.records} == {"float64"}
    assert report.maximum_explicit_residual <= 1e-6
    assert np.isfinite([record.relative_error for record in report.records]).all()
    _assert_two_adjacent_steps_pass_per_direction(report)


def test_complete_cuda_mixed_precision_gradient_pipeline() -> None:
    """A float32 physics solve or wrong gradient return dtype must fail on CUDA."""
    assert torch.cuda.is_available(), "Gate 2A locked environment requires CUDA"
    report = validate_full_pipeline_gradient(
        GradientValidationConfig(),
        device=torch.device("cuda"),
        design_dtype=torch.float32,
    )

    assert report.passed
    assert len(report.records) == 15
    assert {record.dtype for record in report.records} == {"float32"}
    assert report.physics_dtypes == ("float64",)
    assert report.gradient_dtype == "float32"
    assert report.maximum_explicit_residual <= 1e-6
    assert np.isfinite([record.relative_error for record in report.records]).all()
    _assert_two_adjacent_steps_pass_per_direction(report)


def test_gradient_artifact_writer_preserves_machine_status(tmp_path: Path) -> None:
    """Dropping schema, config identity, or numerical status must fail."""
    record = GradientCheckRecord(
        device="cpu",
        dtype="float64",
        direction_seed=7201,
        step_size=1e-3,
        automatic_derivative=0.25,
        finite_difference_derivative=0.25,
        relative_error=0.0,
        passed=True,
    )
    report = GradientValidationReport(
        records=(record,),
        solve_records=(),
        passed=True,
        maximum_explicit_residual=9e-7,
        physics_dtypes=("float64",),
        gradient_dtype="float64",
        config_sha256="abc123",
    )

    csv_path, json_path = write_gradient_validation_artifacts(
        report,
        output_dir=tmp_path,
        label="cpu",
    )

    assert csv_path.is_file() and csv_path.stat().st_size > 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["status"] == "PASS"
    assert payload["config_sha256"] == "abc123"
    assert payload["record_count"] == 1
