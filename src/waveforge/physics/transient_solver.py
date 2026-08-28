"""Implicit-Euler SciPy reference solver for transient heat conduction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import identity
from scipy.sparse.linalg import splu

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import assemble_steady_system

SourceFunction = Callable[[float], NDArray[np.float64]]
SourceSpecification = NDArray[np.float64] | SourceFunction


@dataclass(frozen=True)
class TransientConfig:
    """Fixed implicit-Euler integration settings."""

    dt: float
    n_steps: int
    rho_c: float = 1.0
    store_every: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if not np.isfinite(self.rho_c) or self.rho_c <= 0.0:
            raise ValueError("rho_c must be finite and positive")
        if self.store_every <= 0:
            raise ValueError("store_every must be positive")


@dataclass(frozen=True)
class TransientResult:
    """Stored times and temperature fields for one trajectory."""

    times: NDArray[np.float64]
    temperatures: NDArray[np.float64]


def _evaluate_source(
    source: SourceSpecification,
    time: float,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    evaluated = source(time) if callable(source) else source
    source_array = np.asarray(evaluated, dtype=np.float64)
    if source_array.shape != shape:
        raise ValueError(f"source shape {source_array.shape} != {shape}")
    if not np.all(np.isfinite(source_array)):
        raise ValueError("source must contain only finite values")
    return source_array


def solve_transient(
    *,
    grid: Grid2D,
    conductivity: NDArray[np.float64],
    source: SourceSpecification,
    bcs: BoundaryConditions,
    initial_temperature: NDArray[np.float64],
    config: TransientConfig,
    harmonic_epsilon: float = 1e-12,
) -> TransientResult:
    """Интегрировать trajectory с одной reusable sparse factorization."""
    initial = np.asarray(initial_temperature, dtype=np.float64)
    if initial.shape != grid.shape:
        raise ValueError(f"initial temperature shape {initial.shape} != {grid.shape}")
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial temperature must contain only finite values")

    zero_source = np.zeros(grid.shape, dtype=np.float64)
    steady_system = assemble_steady_system(
        grid,
        conductivity,
        zero_source,
        bcs,
        harmonic_epsilon=harmonic_epsilon,
    )
    mass_coefficient = config.rho_c / config.dt
    transient_matrix = steady_system.matrix + mass_coefficient * identity(
        grid.nx * grid.ny, format="csr"
    )
    factorization = splu(transient_matrix.tocsc())

    flat_temperature = initial.ravel().copy()
    stored_times = [0.0]
    stored_temperatures = [initial.copy()]

    for step in range(1, config.n_steps + 1):
        time = step * config.dt
        source_at_step = _evaluate_source(source, time, grid.shape)
        step_rhs = source_at_step.ravel() + steady_system.dirichlet_rhs
        rhs = mass_coefficient * flat_temperature + step_rhs
        flat_temperature = factorization.solve(rhs)
        if not np.all(np.isfinite(flat_temperature)):
            raise FloatingPointError("transient solution contains NaN or Inf")
        if step % config.store_every == 0 or step == config.n_steps:
            stored_times.append(time)
            stored_temperatures.append(flat_temperature.reshape(grid.shape).copy())

    return TransientResult(
        times=np.asarray(stored_times, dtype=np.float64),
        temperatures=np.stack(stored_temperatures),
    )
