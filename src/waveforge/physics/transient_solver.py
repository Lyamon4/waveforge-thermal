"""Implicit-Euler SciPy reference solver for transient heat conduction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import SuperLU, splu

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import AssembledSystem, assemble_steady_system

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


@dataclass(frozen=True)
class TransientLinearSystem:
    """Assembled implicit-Euler matrix до sparse factorization."""

    grid: Grid2D
    config: TransientConfig
    steady_system: AssembledSystem
    matrix: csc_matrix
    mass_coefficient: float


@dataclass(frozen=True)
class PreparedTransientSystem:
    """Fixed design/operator с reusable sparse factorization."""

    linear_system: TransientLinearSystem
    factorization: SuperLU


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


def assemble_transient_system(
    grid: Grid2D,
    conductivity: NDArray[np.float64],
    bcs: BoundaryConditions,
    config: TransientConfig,
    *,
    harmonic_epsilon: float = 1e-12,
) -> TransientLinearSystem:
    """Собрать fixed-design implicit-Euler matrix без factorization."""
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
    return TransientLinearSystem(
        grid=grid,
        config=config,
        steady_system=steady_system,
        matrix=transient_matrix.tocsc(),
        mass_coefficient=mass_coefficient,
    )


def factorize_transient_system(
    linear_system: TransientLinearSystem,
) -> PreparedTransientSystem:
    """Factorize assembled transient matrix один раз."""
    return PreparedTransientSystem(
        linear_system=linear_system,
        factorization=splu(linear_system.matrix),
    )


def prepare_transient_system(
    grid: Grid2D,
    conductivity: NDArray[np.float64],
    bcs: BoundaryConditions,
    config: TransientConfig,
    *,
    harmonic_epsilon: float = 1e-12,
) -> PreparedTransientSystem:
    """Собрать и factorize fixed-design transient system."""
    return factorize_transient_system(
        assemble_transient_system(
            grid,
            conductivity,
            bcs,
            config,
            harmonic_epsilon=harmonic_epsilon,
        )
    )


def solve_transient_prepared(
    prepared: PreparedTransientSystem,
    source: SourceSpecification,
    initial_temperature: NDArray[np.float64],
) -> TransientResult:
    """Интегрировать scenario с ранее подготовленной factorization."""
    linear_system = prepared.linear_system
    grid = linear_system.grid
    config = linear_system.config
    initial = np.asarray(initial_temperature, dtype=np.float64)
    if initial.shape != grid.shape:
        raise ValueError(f"initial temperature shape {initial.shape} != {grid.shape}")
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial temperature must contain only finite values")

    flat_temperature = initial.ravel().copy()
    stored_times = [0.0]
    stored_temperatures = [initial.copy()]

    for step in range(1, config.n_steps + 1):
        time = step * config.dt
        source_at_step = _evaluate_source(source, time, grid.shape)
        step_rhs = source_at_step.ravel() + linear_system.steady_system.dirichlet_rhs
        rhs = linear_system.mass_coefficient * flat_temperature + step_rhs
        flat_temperature = prepared.factorization.solve(rhs)
        if not np.all(np.isfinite(flat_temperature)):
            raise FloatingPointError("transient solution contains NaN or Inf")
        if step % config.store_every == 0 or step == config.n_steps:
            stored_times.append(time)
            stored_temperatures.append(flat_temperature.reshape(grid.shape).copy())

    return TransientResult(
        times=np.asarray(stored_times, dtype=np.float64),
        temperatures=np.stack(stored_temperatures),
    )


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
    """Собрать, factorize и интегрировать one-design trajectory."""
    prepared = prepare_transient_system(
        grid,
        conductivity,
        bcs,
        config,
        harmonic_epsilon=harmonic_epsilon,
    )
    return solve_transient_prepared(prepared, source, initial_temperature)
