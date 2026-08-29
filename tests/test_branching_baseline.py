"""Tests for the prospective parametric branching-tree baseline."""

from __future__ import annotations

import numpy as np

from waveforge.design.branching_baseline import (
    BranchingTreeParameters,
    branching_score,
    build_branching_tree,
    candidate_axes,
    iter_candidate_parameters,
    segment_distance,
)
from waveforge.physics.grid import Grid2D


def test_candidate_registry_has_exact_locked_cartesian_product() -> None:
    """Dropping an endpoint or duplicating an axis value must fail."""
    axes = candidate_axes()
    assert tuple(map(len, axes)) == (17, 21, 23, 5)
    parameters = tuple(iter_candidate_parameters())
    assert len(parameters) == 41055
    assert len({item.candidate_id for item in parameters}) == 41055
    assert parameters[0].as_tuple() == (0.30, 0.25, 0.10, 0.75)
    assert parameters[-1].as_tuple() == (0.70, 0.75, 0.65, 2.0)


def test_segment_distance_uses_clamped_finite_segment_projection() -> None:
    """Treating a finite segment as an infinite line must fail."""
    points = np.array([[0.5, 0.5], [2.0, 0.0], [-1.0, 0.0]])
    actual = segment_distance(points, (0.0, 0.0), (1.0, 0.0))
    np.testing.assert_allclose(actual, [0.5, 1.0, 1.0], rtol=0.0, atol=1e-15)


def test_tree_mask_is_strict_binary_exact_budget_and_repeatable() -> None:
    """Any non-binary, repaired or non-deterministic map must fail."""
    parameters = BranchingTreeParameters(0.5, 0.5, 0.3, 1.5)
    first = build_branching_tree(parameters)
    second = build_branching_tree(parameters)
    assert first.name == parameters.candidate_id
    assert np.array_equal(first.design, second.design)
    assert int(first.design.sum()) == 1024
    assert float(first.design.mean()) == 0.25
    assert set(np.unique(first.design)) == {0.0, 1.0}
    assert first.design.flags.writeable is False


def test_larger_trunk_ratio_never_reduces_locked_geometric_score() -> None:
    """Applying the width ratio to branches or with the wrong sign must fail."""
    grid = Grid2D(nx=64, ny=64)
    narrow = branching_score(
        BranchingTreeParameters(0.5, 0.45, 0.3, 0.75),
        grid,
    )
    wide = branching_score(
        BranchingTreeParameters(0.5, 0.45, 0.3, 2.0),
        grid,
    )
    assert np.all(wide >= narrow)
    assert np.any(wide > narrow)
