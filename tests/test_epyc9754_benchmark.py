"""Tests for the frozen EPYC 9754-scale synthetic extreme-OOD registry."""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.design.epyc9754_benchmark import (
    BENCHMARK_LABEL,
    PACKAGE_HEIGHT_MM,
    PACKAGE_WIDTH_MM,
    build_epyc9754_scale_benchmark,
)
from waveforge.ml.multitask_tasks import build_frozen_splits


def test_epyc_scale_registry_has_public_package_scale_and_nine_regions() -> None:
    benchmark = build_epyc9754_scale_benchmark(resolution=64)

    assert BENCHMARK_LABEL == "EPYC_9754_SCALE_SYNTHETIC"
    assert pytest.approx(75.4) == PACKAGE_WIDTH_MM
    assert pytest.approx(72.0) == PACKAGE_HEIGHT_MM
    assert len(benchmark.regions) == 9
    assert [region.region_id for region in benchmark.regions[:8]] == [
        f"CCD_{index}" for index in range(1, 9)
    ]
    assert benchmark.regions[-1].region_id == "IO_DIE"


def test_every_synthetic_workload_integrates_to_exact_360_watts() -> None:
    benchmark = build_epyc9754_scale_benchmark(resolution=64)
    cell_area = 1.0 / (64 * 64)

    assert len(benchmark.workloads) == 3
    for workload, source in zip(benchmark.workloads, benchmark.sources, strict=True):
        assert sum(workload.region_powers_watts) == pytest.approx(360.0)
        assert np.sum(source) * cell_area == pytest.approx(360.0, abs=1.0e-10)


def test_epyc_registry_is_secondary_and_disjoint_from_primary_task_manifest() -> None:
    benchmark = build_epyc9754_scale_benchmark(resolution=64)
    primary_ids = {task.task_id for task in build_frozen_splits().all_tasks}

    assert benchmark.benchmark_role == "secondary_extreme_ood"
    assert benchmark.task_id not in primary_ids
    assert benchmark.may_select_checkpoint is False
    assert benchmark.may_update_weights is False


def test_epyc_source_maps_are_finite_nonnegative_and_have_three_scenarios() -> None:
    benchmark = build_epyc9754_scale_benchmark(resolution=128)

    assert benchmark.sources.shape == (3, 128, 128)
    assert benchmark.sources.dtype == np.float64
    assert np.isfinite(benchmark.sources).all()
    assert np.all(benchmark.sources >= 0.0)
