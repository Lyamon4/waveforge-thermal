"""Tests for deterministic Gate 2A source rasterization."""

import numpy as np
import pytest

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.grid import Grid2D


def test_area_overlap_source_preserves_power_when_edges_cut_cells() -> None:
    """Changing from exact overlap to center inclusion must break this test."""
    grid = Grid2D(nx=7, ny=5)

    source = area_overlap_rectangular_source(
        grid,
        bounds=(0.13, 0.61, 0.27, 0.83),
        power=1.0,
    )

    assert np.sum(source) * grid.dx * grid.dy == pytest.approx(1.0, abs=1e-14)


def test_area_overlap_source_uses_fractional_boundary_cells() -> None:
    """Treating every intersected cell as fully covered must break this test."""
    grid = Grid2D(nx=4, ny=4)

    source = area_overlap_rectangular_source(
        grid,
        bounds=(0.20, 0.55, 0.20, 0.55),
        power=1.0,
    )

    assert 0.0 < source[0, 0] < source[1, 1]


@pytest.mark.parametrize(
    "bounds",
    [
        (-0.01, 0.5, 0.2, 0.4),
        (0.2, 1.01, 0.2, 0.4),
        (0.2, 0.4, -0.01, 0.5),
        (0.2, 0.4, 0.2, 1.01),
        (0.5, 0.5, 0.2, 0.4),
    ],
)
def test_area_overlap_source_rejects_invalid_rectangle(
    bounds: tuple[float, float, float, float],
) -> None:
    grid = Grid2D(nx=8, ny=8)

    with pytest.raises(ValueError, match="rectangle"):
        area_overlap_rectangular_source(grid, bounds=bounds, power=1.0)

