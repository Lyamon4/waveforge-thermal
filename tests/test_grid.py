"""Проверки cell-centered finite-volume grid."""

import numpy as np
import pytest

from waveforge.physics.grid import Grid2D


def test_grid_uses_y_x_array_layout_and_cell_centers() -> None:
    """Перестановка x/y axes должна ломать этот test."""
    grid = Grid2D(nx=4, ny=2)

    assert grid.shape == (2, 4)
    assert grid.dx == pytest.approx(0.25)
    assert grid.dy == pytest.approx(0.5)
    np.testing.assert_allclose(grid.x_centers, [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(grid.y_centers, [0.25, 0.75])
    x, y = grid.mesh
    assert x.shape == y.shape == (2, 4)
    np.testing.assert_allclose(x[0], grid.x_centers)
    np.testing.assert_allclose(y[:, 0], grid.y_centers)
    assert y[0, 0] < y[-1, 0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nx": 1, "ny": 2}, "at least 2"),
        ({"nx": 2, "ny": 1}, "at least 2"),
        ({"nx": 2, "ny": 2, "lx": 0.0}, "positive"),
        ({"nx": 2, "ny": 2, "ly": -1.0}, "positive"),
    ],
)
def test_grid_rejects_invalid_resolution_or_domain(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    """Degenerate control volumes должны отклоняться до assembly."""
    with pytest.raises(ValueError, match=message):
        Grid2D(**kwargs)
