"""Reusable material-free finite-volume operator for MT2B conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import SuperLU

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import assemble_steady_system, factorize_system


@dataclass(frozen=True)
class FixedOperatorResult:
    temperature: NDArray[np.float64]
    normalized_residuals: NDArray[np.float64]

    @property
    def maximum_normalized_residual(self) -> float:
        return float(np.max(self.normalized_residuals))


class UniformPlateFactorization:
    """Assemble and factor the fixed uniform plate operator exactly once."""

    def __init__(self, grid_size: int, conductivity: float) -> None:
        if not np.isfinite(conductivity) or conductivity <= 0.0:
            raise ValueError("conductivity must be finite and positive")
        self.grid = Grid2D(nx=grid_size, ny=grid_size)
        uniform = np.full(self.grid.shape, conductivity, dtype=np.float64)
        zero_source = np.zeros(self.grid.shape, dtype=np.float64)
        self.system = assemble_steady_system(
            self.grid,
            uniform,
            zero_source,
            BoundaryConditions.production(),
        )
        self.factorization: SuperLU = factorize_system(self.system)
        self.factorization_count = 1

    def solve_many(self, sources: NDArray[np.float64]) -> FixedOperatorResult:
        """Solve one matrix against all scenario RHS columns at once."""
        source_array = np.asarray(sources, dtype=np.float64)
        expected = (
            (source_array.shape[0], *self.grid.shape) if source_array.ndim == 3 else ()
        )
        if source_array.ndim != 3 or source_array.shape != expected:
            raise ValueError(
                f"sources must have shape [rhs,{self.grid.ny},{self.grid.nx}]"
            )
        if source_array.shape[0] < 1 or not np.all(np.isfinite(source_array)):
            raise ValueError("sources must be a nonempty finite RHS batch")

        flattened_sources = source_array.reshape(source_array.shape[0], -1)
        rhs = flattened_sources + self.system.dirichlet_rhs[None, :]
        flat_temperature = self.factorization.solve(rhs.T).T
        if not np.all(np.isfinite(flat_temperature)):
            raise FloatingPointError("fixed-operator solve produced NaN or Inf")
        residual = (self.system.matrix @ flat_temperature.T).T - rhs
        denominator = np.maximum(np.linalg.norm(rhs, axis=1), 1.0)
        normalized = np.linalg.norm(residual, axis=1) / denominator
        return FixedOperatorResult(
            temperature=flat_temperature.reshape(source_array.shape),
            normalized_residuals=normalized.astype(np.float64, copy=False),
        )
