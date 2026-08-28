"""Uniform cell-centered finite-volume grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Grid2D:
    """Прямоугольный grid с array layout `[ny, nx]`."""

    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must be at least 2")
        if self.lx <= 0.0 or self.ly <= 0.0:
            raise ValueError("domain lengths must be positive")

    @property
    def dx(self) -> float:
        """Cell width along x."""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """Cell height along y."""
        return self.ly / self.ny

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape in `[ny, nx]` order."""
        return (self.ny, self.nx)

    @property
    def size(self) -> int:
        """Total number of control volumes."""
        return self.nx * self.ny

    @property
    def x_centers(self) -> NDArray[np.float64]:
        """One-dimensional x coordinates of cell centers."""
        return (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx

    @property
    def y_centers(self) -> NDArray[np.float64]:
        """One-dimensional y coordinates of cell centers."""
        return (np.arange(self.ny, dtype=np.float64) + 0.5) * self.dy

    @property
    def mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cell-center coordinate arrays in `[ny, nx]` layout."""
        return np.meshgrid(self.x_centers, self.y_centers, indexing="xy")
