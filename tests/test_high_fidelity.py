"""Tests for frozen-design independent SciPy verification."""

import numpy as np
import pytest

from waveforge.design.baselines import straight_path_baseline
from waveforge.physics.grid import Grid2D
from waveforge.verification.high_fidelity import (
    VerificationIntegrityError,
    array_sha256,
    relative_improvement,
    replicate_design,
    verify_candidate,
)


def test_replication_maps_each_parent_to_exact_constant_child_block() -> None:
    """Interpolation, filtering, or re-thresholding during transfer must fail."""
    parent = np.arange(16, dtype=np.float64).reshape(4, 4)

    doubled = replicate_design(parent, factor=2)
    quadrupled = replicate_design(parent, factor=4)

    for row in range(4):
        for column in range(4):
            assert np.all(
                doubled[2 * row : 2 * row + 2, 2 * column : 2 * column + 2]
                == parent[row, column]
            )
            assert np.all(
                quadrupled[4 * row : 4 * row + 4, 4 * column : 4 * column + 4]
                == parent[row, column]
            )
    assert doubled.mean() == parent.mean()
    assert quadrupled.mean() == parent.mean()


def test_verifier_rejects_corrupt_frozen_map_hash() -> None:
    """Verifying a map different from the frozen candidate must invalidate."""
    design = straight_path_baseline(Grid2D(nx=64, ny=64)).design

    with pytest.raises(VerificationIntegrityError, match="hash"):
        verify_candidate(
            "straight",
            design,
            fidelity="reference_128",
            expected_design_hash="0" * 64,
        )


def test_verifier_uses_independent_scipy_metric_not_claimed_peak() -> None:
    """Trusting a low-fidelity claimed result must fail this boundary."""
    design = straight_path_baseline(Grid2D(nx=64, ny=64)).design
    frozen_hash = array_sha256(design)

    result = verify_candidate(
        "straight",
        design,
        fidelity="reference_128",
        expected_design_hash=frozen_hash,
        claimed_worst_peak=999.0,
    )

    assert result.fidelity == "reference_128"
    assert result.grid_shape == (128, 128)
    assert result.design_hash_64 == frozen_hash
    assert result.worst_peak < 999.0
    assert result.claimed_worst_peak == 999.0
    assert not result.claim_matches
    assert len(result.scenario_records) == 3
    assert all(
        record.normalized_residual <= 1e-10 for record in result.scenario_records
    )
    assert result.material_fraction == pytest.approx(0.25)


def test_relative_improvement_uses_unrounded_temperatures() -> None:
    """Rounded display values must not change the 5% decision quantity."""
    baseline_peak = 1.00004
    candidate_peak = 0.95002

    improvement = relative_improvement(baseline_peak, candidate_peak)

    assert improvement == pytest.approx(
        (baseline_peak - candidate_peak) / baseline_peak,
        abs=0.0,
    )
