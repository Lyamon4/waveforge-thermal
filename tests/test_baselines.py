"""Tests for pre-registered deterministic Gate 2A baselines."""

import numpy as np

from waveforge.design.baselines import (
    dispersed_baseline,
    random_filtered_baseline,
    stable_top_k_mask,
    straight_path_baseline,
    uniform_relaxed_baseline,
)
from waveforge.physics.grid import Grid2D


def test_stable_top_k_breaks_ties_by_lower_row_major_index() -> None:
    """An unstable top-k implementation must fail on equal values."""
    values = np.array([[4.0, 4.0, 2.0], [4.0, 1.0, 0.0]])

    mask = stable_top_k_mask(values, count=2)

    expected = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(mask, expected)


def test_random_filtered_baselines_are_deterministic_and_budget_exact() -> None:
    """A global RNG, wrong seed, or approximate selection must fail."""
    grid = Grid2D(nx=64, ny=64)
    designs = [random_filtered_baseline(grid, seed) for seed in (9101, 9102, 9103)]
    repeated = random_filtered_baseline(grid, 9101)

    for baseline in designs:
        assert baseline.design.sum() == 1024
        assert baseline.design.mean() == 0.25
        assert set(np.unique(baseline.design)) == {0.0, 1.0}
        assert baseline.algorithm == "random_filtered_top_k"
    np.testing.assert_array_equal(designs[0].design, repeated.design)
    assert designs[0].parameter_hash == repeated.parameter_hash
    assert not np.array_equal(designs[0].design, designs[1].design)


def test_straight_path_selects_exact_locked_columns() -> None:
    """Moving an endpoint or including the upper endpoint must fail."""
    grid = Grid2D(nx=64, ny=64)

    baseline = straight_path_baseline(grid)

    selected_columns = np.flatnonzero(np.any(baseline.design == 1.0, axis=0))
    expected_columns = np.arange(24, 40)
    np.testing.assert_array_equal(selected_columns, expected_columns)
    assert baseline.design.sum() == 1024
    assert np.all(baseline.design[:, expected_columns] == 1.0)


def test_dispersed_baseline_uses_even_row_even_column() -> None:
    """Changing the registered cell within each 2x2 block must fail."""
    grid = Grid2D(nx=64, ny=64)

    baseline = dispersed_baseline(grid)

    expected = np.zeros(grid.shape)
    expected[::2, ::2] = 1.0
    np.testing.assert_array_equal(baseline.design, expected)
    assert baseline.design.mean() == 0.25


def test_uniform_relaxed_baseline_is_exactly_quarter_material() -> None:
    """A binary substitute or grid-dependent relaxed value must fail."""
    grid = Grid2D(nx=64, ny=64)

    baseline = uniform_relaxed_baseline(grid)

    np.testing.assert_array_equal(baseline.design, np.full(grid.shape, 0.25))
    assert not baseline.is_binary
