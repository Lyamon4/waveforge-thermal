"""Tests for the immutable Gate 2A verification candidate registry."""

import json
from pathlib import Path

import numpy as np
import pytest

from waveforge.experiments.verify_gate2a import (
    CandidateIntegrityError,
    baseline_ids_for_seed,
    build_candidate_registry,
    load_production_candidate,
)


def test_registry_contains_every_locked_binary_and_continuous_design() -> None:
    """Dropping a seed/comparator or thresholding uniform relaxed must fail."""
    root = Path("artifacts/gate2_design/production")

    registry = build_candidate_registry(root)

    assert len(registry.binary) == 11
    assert len(registry.continuous) == 7
    assert {candidate.candidate_id for candidate in registry.binary} == {
        "random_filtered_seed_9101",
        "random_filtered_seed_9102",
        "random_filtered_seed_9103",
        "straight_path",
        "evenly_dispersed_binary",
        "single_A_20260828",
        "single_A_20260829",
        "single_A_20260830",
        "robust_20260828",
        "robust_20260829",
        "robust_20260830",
    }
    assert {candidate.candidate_id for candidate in registry.continuous} == {
        "uniform_relaxed",
        "single_A_20260828_continuous",
        "single_A_20260829_continuous",
        "single_A_20260830_continuous",
        "robust_20260828_continuous",
        "robust_20260829_continuous",
        "robust_20260830_continuous",
    }
    assert all(candidate.representation == "binary" for candidate in registry.binary)
    assert all(
        candidate.representation == "continuous" for candidate in registry.continuous
    )


def test_same_seed_baseline_set_is_exactly_the_locked_six_members() -> None:
    """Cross-seed single-scenario comparators or uniform relaxed must fail."""
    ids = baseline_ids_for_seed(20260829)

    assert ids == (
        "random_filtered_seed_9101",
        "random_filtered_seed_9102",
        "random_filtered_seed_9103",
        "straight_path",
        "evenly_dispersed_binary",
        "single_A_20260829",
    )


def test_loading_production_candidate_rejects_corrupt_frozen_array(
    tmp_path: Path,
) -> None:
    """An array whose content differs from optimization_result is invalid."""
    source = Path("artifacts/gate2_design/production/robust/20260828")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in (
        "design_binary_64.npy",
        "design_continuous_64.npy",
        "optimization_metrics.csv",
        "optimization_result.json",
    ):
        (run_dir / name).write_bytes((source / name).read_bytes())
    design = np.load(run_dir / "design_binary_64.npy")
    design[0, 0] = 1.0 - design[0, 0]
    np.save(run_dir / "design_binary_64.npy", design)
    result = json.loads((run_dir / "optimization_result.json").read_text())

    with pytest.raises(CandidateIntegrityError, match="binary design hash"):
        load_production_candidate(
            run_dir,
            candidate_id="robust_20260828",
            category="robust",
            seed=20260828,
            expected_config_hash=result["config_sha256"],
            expected_protocol_tag=result["protocol_tag"],
        )
