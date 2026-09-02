from __future__ import annotations

import numpy as np
import pytest

from waveforge.design.epyc9754_benchmark import build_epyc9754_scale_benchmark
from waveforge.experiments.run_mt3_final_evaluation import (
    _write_task_manifest,
    exact_binary64,
    independent_source_maps_tmax,
    neural_equivalent_evaluations,
    strong_single_reference,
)
from waveforge.ml.multitask_tasks import build_frozen_splits


def test_strong_single_reference_is_prospectively_lower_verified_baseline() -> None:
    assert strong_single_reference(adam_tmax=0.18, mma_tmax=0.17) == (
        "MMA_600",
        0.17,
    )
    assert strong_single_reference(adam_tmax=0.16, mma_tmax=0.17) == (
        "ADAM_600",
        0.16,
    )
    assert strong_single_reference(adam_tmax=0.17, mma_tmax=0.17) == (
        "ADAM_600",
        0.17,
    )


def test_strong_single_reference_rejects_invalid_numerics() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        strong_single_reference(adam_tmax=np.nan, mma_tmax=0.17)


def test_exact_binary64_requires_locked_budget() -> None:
    design = np.zeros((64, 64), dtype=np.float64)
    design.reshape(-1)[:1024] = 1.0
    assert exact_binary64(design).shape == (64, 64)

    design.reshape(-1)[1024] = 1.0
    with pytest.raises(ValueError, match="1024"):
        exact_binary64(design)


def test_primary_neural_accounting_counts_only_one_refinement_chain() -> None:
    assert neural_equivalent_evaluations(refinement_updates=25) == 30
    assert neural_equivalent_evaluations(refinement_updates=50) == 55
    with pytest.raises(ValueError, match="25 or 50"):
        neural_equivalent_evaluations(refinement_updates=100)


def test_opened_task_manifest_is_stable_after_json_round_trip(tmp_path) -> None:
    splits = build_frozen_splits()

    _write_task_manifest(tmp_path, splits)
    _write_task_manifest(tmp_path, splits)


def test_epyc_scorer_uses_registered_source_maps_without_primary_task_adapter() -> None:
    benchmark = build_epyc9754_scale_benchmark(resolution=64)
    design = np.zeros((64, 64), dtype=np.float64)
    design[:, :16] = 1.0

    peak = independent_source_maps_tmax(design, benchmark)

    assert np.isfinite(peak)
    assert peak > 0.0
