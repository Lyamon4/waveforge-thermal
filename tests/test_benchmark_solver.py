"""Проверки benchmark semantics без long production matrix."""

import numpy as np
import pytest

from waveforge.experiments.benchmark_solver import (
    benchmark_steady_case,
    benchmark_transient_case,
    build_argument_parser,
    generate_conductivity_maps,
    summarize_timings,
    validate_config_dir,
)
from waveforge.physics.grid import Grid2D
from waveforge.reproducibility import content_hash


def test_summary_uses_sample_standard_deviation_and_p90() -> None:
    summary = summarize_timings([1.0, 2.0, 3.0, 4.0])
    assert summary.mean == pytest.approx(2.5)
    assert summary.median == pytest.approx(2.5)
    assert summary.p90 == pytest.approx(3.7)
    assert summary.std == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
    assert summary.runs == 4


def test_cold_maps_change_conductivity_for_every_run() -> None:
    grid = Grid2D(nx=8, ny=8)
    maps = generate_conductivity_maps(grid, count=4, seed=20260828)
    hashes = [content_hash(conductivity) for conductivity in maps]

    assert len(set(hashes)) == 4
    assert all(np.all(conductivity > 0.0) for conductivity in maps)


def test_steady_smoke_separates_warm_and_cold_phases() -> None:
    records = benchmark_steady_case(
        resolution=8,
        warmup_runs=1,
        measured_runs=3,
        scenario_count=3,
        seed=20260828,
    )
    phases = {(record.mode, record.phase) for record in records}

    assert ("warm_reused", "solve") in phases
    assert ("cold_design", "assembly") in phases
    assert ("cold_design", "factorization") in phases
    assert ("cold_design", "solve") in phases
    assert ("cold_design", "total_evaluation") in phases
    assert all(record.runs == 3 for record in records)
    assert all(record.mean > 0.0 for record in records)


def test_transient_smoke_separates_trajectory_and_cold_phases() -> None:
    records = benchmark_transient_case(
        resolution=8,
        time_steps=3,
        warmup_runs=0,
        measured_runs=2,
        scenario_count=3,
        seed=20260828,
    )
    phases = {(record.mode, record.phase) for record in records}

    assert ("warm_reused", "trajectory") in phases
    assert ("cold_design", "assembly") in phases
    assert ("cold_design", "factorization") in phases
    assert ("cold_design", "trajectory") in phases
    assert ("cold_design", "total_evaluation") in phases
    assert all(record.runs == 2 for record in records)


def test_transient_step_cases_use_same_conductivity_family() -> None:
    """Time-step count не должен менять benchmark design inputs."""
    short = benchmark_transient_case(
        resolution=8,
        time_steps=2,
        warmup_runs=0,
        measured_runs=1,
        scenario_count=3,
        seed=20260828,
    )
    long = benchmark_transient_case(
        resolution=8,
        time_steps=4,
        warmup_runs=0,
        measured_runs=1,
        scenario_count=3,
        seed=20260828,
    )

    assert short[0].conductivity_family_hash == long[0].conductivity_family_hash


def test_benchmark_cli_accepts_registered_config_dir(tmp_path) -> None:
    arguments = build_argument_parser().parse_args(
        ["--config-dir", str(tmp_path), "--warmups", "1", "--runs", "2"]
    )

    assert arguments.config_dir == tmp_path
    assert arguments.warmups == 1
    assert arguments.runs == 2


def test_config_dir_requires_both_gate1_configs(tmp_path) -> None:
    (tmp_path / "steady_validation.yaml").write_text("seed: 1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"transient_validation\.yaml"):
        validate_config_dir(tmp_path)
