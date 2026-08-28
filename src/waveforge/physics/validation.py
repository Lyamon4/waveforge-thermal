"""Independent numerical and physical validation metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.conductivity import harmonic_mean, validate_conductivity
from waveforge.physics.grid import Grid2D


def relative_l2(
    predicted: NDArray[np.float64],
    exact: NDArray[np.float64],
) -> float:
    """Вычислить `||predicted-exact|| / max(||exact||, 1e-12)`."""
    predicted_array = np.asarray(predicted, dtype=np.float64)
    exact_array = np.asarray(exact, dtype=np.float64)
    if predicted_array.shape != exact_array.shape:
        raise ValueError("predicted and exact fields must have identical shapes")
    if not np.all(np.isfinite(predicted_array)) or not np.all(np.isfinite(exact_array)):
        raise ValueError("relative L2 inputs must be finite")
    return float(
        np.linalg.norm(predicted_array - exact_array)
        / max(np.linalg.norm(exact_array), 1e-12)
    )


def symmetry_defect(field: NDArray[np.float64]) -> float:
    """Вычислить normalized left-right symmetry defect."""
    field_array = np.asarray(field, dtype=np.float64)
    if field_array.ndim != 2 or not np.all(np.isfinite(field_array)):
        raise ValueError("symmetry field must be a finite two-dimensional array")
    return float(
        np.max(np.abs(field_array - np.flip(field_array, axis=1)))
        / max(np.max(np.abs(field_array)), 1e-12)
    )


def two_layer_interface_flux(
    grid: Grid2D,
    conductivity: NDArray[np.float64],
    temperature: NDArray[np.float64],
    *,
    harmonic_epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Вычислить magnitude discrete flux на aligned interface `x=0.5`."""
    if grid.nx % 2 != 0:
        raise ValueError("two-layer interface requires an even nx")
    conductivity_array = np.asarray(conductivity, dtype=np.float64)
    temperature_array = np.asarray(temperature, dtype=np.float64)
    validate_conductivity(conductivity_array, grid.shape)
    if temperature_array.shape != grid.shape or not np.all(
        np.isfinite(temperature_array)
    ):
        raise ValueError("temperature must be finite and match grid shape")

    right_column = grid.nx // 2
    left_column = right_column - 1
    face_conductivity = harmonic_mean(
        conductivity_array[:, left_column],
        conductivity_array[:, right_column],
        harmonic_epsilon,
    )
    gradient = (
        temperature_array[:, right_column] - temperature_array[:, left_column]
    ) / grid.dx
    return face_conductivity * gradient


def dirichlet_outward_flux(
    grid: Grid2D,
    conductivity: NDArray[np.float64],
    temperature: NDArray[np.float64],
    bcs: BoundaryConditions,
) -> float:
    """Независимо суммировать outward flux через Dirichlet faces."""
    conductivity_array = np.asarray(conductivity, dtype=np.float64)
    temperature_array = np.asarray(temperature, dtype=np.float64)
    validate_conductivity(conductivity_array, grid.shape)
    if temperature_array.shape != grid.shape or not np.all(
        np.isfinite(temperature_array)
    ):
        raise ValueError("temperature must be finite and match grid shape")

    outward = 0.0
    if bcs.left.kind == "dirichlet":
        outward += float(
            np.sum(
                2.0
                * conductivity_array[:, 0]
                * (temperature_array[:, 0] - bcs.left.value)
                / grid.dx
                * grid.dy
            )
        )
    if bcs.right.kind == "dirichlet":
        outward += float(
            np.sum(
                2.0
                * conductivity_array[:, -1]
                * (temperature_array[:, -1] - bcs.right.value)
                / grid.dx
                * grid.dy
            )
        )
    if bcs.bottom.kind == "dirichlet":
        outward += float(
            np.sum(
                2.0
                * conductivity_array[0, :]
                * (temperature_array[0, :] - bcs.bottom.value)
                / grid.dy
                * grid.dx
            )
        )
    if bcs.top.kind == "dirichlet":
        outward += float(
            np.sum(
                2.0
                * conductivity_array[-1, :]
                * (temperature_array[-1, :] - bcs.top.value)
                / grid.dy
                * grid.dx
            )
        )
    return outward
