"""Sparse SciPy reference solver for steady heat conduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import SuperLU, splu

from waveforge.physics.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditions,
)
from waveforge.physics.conductivity import harmonic_mean, validate_conductivity
from waveforge.physics.grid import Grid2D


@dataclass(frozen=True)
class AssembledSystem:
    """Finite-volume operator и разложенный assembled RHS."""

    grid: Grid2D
    matrix: csr_matrix
    rhs: NDArray[np.float64]
    source_rhs: NDArray[np.float64]
    dirichlet_rhs: NDArray[np.float64]


@dataclass(frozen=True)
class SteadyResult:
    """Steady temperature field, residual и использованная linear system."""

    temperature: NDArray[np.float64]
    normalized_residual: float
    system: AssembledSystem


def _validate_source(source: NDArray[np.float64], grid: Grid2D) -> None:
    if source.shape != grid.shape:
        raise ValueError(f"source shape {source.shape} != {grid.shape}")
    if not np.all(np.isfinite(source)):
        raise ValueError("source must contain only finite values")


def _apply_dirichlet_face(
    diagonal: NDArray[np.float64],
    dirichlet_rhs: NDArray[np.float64],
    indices: NDArray[np.int64],
    face_conductivity: NDArray[np.float64],
    spacing: float,
    condition: BoundaryCondition,
) -> None:
    if condition.kind == "neumann":
        return
    conductance = 2.0 * face_conductivity / spacing**2
    np.add.at(diagonal, indices, conductance)
    np.add.at(dirichlet_rhs, indices, conductance * condition.value)


def assemble_steady_system(
    grid: Grid2D,
    conductivity: NDArray[np.float64],
    source: NDArray[np.float64],
    bcs: BoundaryConditions,
    *,
    harmonic_epsilon: float = 1e-12,
) -> AssembledSystem:
    """Собрать `A T = b` для cell-centered flux-form discretization."""
    bcs.require_well_posed()
    conductivity_array = np.asarray(conductivity, dtype=np.float64)
    source_array = np.asarray(source, dtype=np.float64)
    validate_conductivity(conductivity_array, grid.shape)
    _validate_source(source_array, grid)

    cell_indices = np.arange(grid.nx * grid.ny, dtype=np.int64).reshape(grid.shape)
    diagonal = np.zeros(grid.nx * grid.ny, dtype=np.float64)
    dirichlet_rhs = np.zeros_like(diagonal)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    west_cells = cell_indices[:, :-1].ravel()
    east_cells = cell_indices[:, 1:].ravel()
    east_conductance = (
        harmonic_mean(
            conductivity_array[:, :-1],
            conductivity_array[:, 1:],
            harmonic_epsilon,
        ).ravel()
        / grid.dx**2
    )
    np.add.at(diagonal, west_cells, east_conductance)
    np.add.at(diagonal, east_cells, east_conductance)
    rows.extend(np.concatenate((west_cells, east_cells)).tolist())
    columns.extend(np.concatenate((east_cells, west_cells)).tolist())
    values.extend(np.concatenate((-east_conductance, -east_conductance)).tolist())

    south_cells = cell_indices[:-1, :].ravel()
    north_cells = cell_indices[1:, :].ravel()
    north_conductance = (
        harmonic_mean(
            conductivity_array[:-1, :],
            conductivity_array[1:, :],
            harmonic_epsilon,
        ).ravel()
        / grid.dy**2
    )
    np.add.at(diagonal, south_cells, north_conductance)
    np.add.at(diagonal, north_cells, north_conductance)
    rows.extend(np.concatenate((south_cells, north_cells)).tolist())
    columns.extend(np.concatenate((north_cells, south_cells)).tolist())
    values.extend(np.concatenate((-north_conductance, -north_conductance)).tolist())

    _apply_dirichlet_face(
        diagonal,
        dirichlet_rhs,
        cell_indices[:, 0],
        conductivity_array[:, 0],
        grid.dx,
        bcs.left,
    )
    _apply_dirichlet_face(
        diagonal,
        dirichlet_rhs,
        cell_indices[:, -1],
        conductivity_array[:, -1],
        grid.dx,
        bcs.right,
    )
    _apply_dirichlet_face(
        diagonal,
        dirichlet_rhs,
        cell_indices[0, :],
        conductivity_array[0, :],
        grid.dy,
        bcs.bottom,
    )
    _apply_dirichlet_face(
        diagonal,
        dirichlet_rhs,
        cell_indices[-1, :],
        conductivity_array[-1, :],
        grid.dy,
        bcs.top,
    )

    flat_indices = cell_indices.ravel()
    rows.extend(flat_indices.tolist())
    columns.extend(flat_indices.tolist())
    values.extend(diagonal.tolist())
    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(grid.nx * grid.ny, grid.nx * grid.ny),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()

    source_rhs = source_array.ravel().copy()
    rhs = source_rhs + dirichlet_rhs
    return AssembledSystem(
        grid=grid,
        matrix=matrix,
        rhs=rhs,
        source_rhs=source_rhs,
        dirichlet_rhs=dirichlet_rhs,
    )


def factorize_system(system: AssembledSystem) -> SuperLU:
    """Создать reusable sparse direct factorization."""
    return splu(system.matrix.tocsc())


def solve_factorized(
    system: AssembledSystem,
    factorization: SuperLU,
) -> SteadyResult:
    """Решить assembled system и проверить residual против полного RHS."""
    flat_temperature = factorization.solve(system.rhs)
    if not np.all(np.isfinite(flat_temperature)):
        raise FloatingPointError("steady solution contains NaN or Inf")
    residual_vector = system.matrix @ flat_temperature - system.rhs
    normalized_residual = float(
        np.linalg.norm(residual_vector) / max(np.linalg.norm(system.rhs), 1.0)
    )
    return SteadyResult(
        temperature=flat_temperature.reshape(system.grid.shape),
        normalized_residual=normalized_residual,
        system=system,
    )


def solve_steady(
    grid: Grid2D,
    conductivity: NDArray[np.float64],
    source: NDArray[np.float64],
    bcs: BoundaryConditions,
    *,
    harmonic_epsilon: float = 1e-12,
) -> SteadyResult:
    """Собрать и решить steady reference problem."""
    system = assemble_steady_system(
        grid,
        conductivity,
        source,
        bcs,
        harmonic_epsilon=harmonic_epsilon,
    )
    return solve_factorized(system, factorize_system(system))
