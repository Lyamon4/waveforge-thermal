"""Reproducible feasibility probe for eager-CUDA explicit differentiation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import assemble_steady_system


@dataclass(frozen=True)
class ExplicitStabilityLimit:
    """Monotonicity-preserving forward-Euler bound for the locked operator."""

    max_diagonal: float
    dt_monotone: float


@dataclass(frozen=True)
class PreparedExplicitFluxCoefficients:
    """Face coefficients that stay fixed over one design trajectory."""

    x_face: torch.Tensor
    y_face: torch.Tensor
    cooled_bottom: torch.Tensor


def explicit_stability_limit(
    grid: Grid2D,
    *,
    k_max: float,
    rho_c: float,
) -> ExplicitStabilityLimit:
    """Вычислить `dt <= rho_c / max(diag(A))` из SciPy flux operator."""
    if not np.isfinite(k_max) or k_max <= 0.0:
        raise ValueError("k_max must be finite and positive")
    if not np.isfinite(rho_c) or rho_c <= 0.0:
        raise ValueError("rho_c must be finite and positive")
    conductivity = np.full(grid.shape, k_max, dtype=np.float64)
    system = assemble_steady_system(
        grid,
        conductivity,
        np.zeros(grid.shape, dtype=np.float64),
        BoundaryConditions.production(),
    )
    max_diagonal = float(np.max(system.matrix.diagonal()))
    return ExplicitStabilityLimit(
        max_diagonal=max_diagonal,
        dt_monotone=rho_c / max_diagonal,
    )


def steps_for_horizon(*, t_final: float, dt: float) -> int:
    """Вернуть минимальное число steps, не занижающее physical horizon."""
    if not np.isfinite(t_final) or t_final <= 0.0:
        raise ValueError("t_final must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return math.ceil(t_final / dt)


def explicit_flux_step(
    temperature: torch.Tensor,
    conductivity: torch.Tensor,
    source: torch.Tensor,
    *,
    dt: float,
    rho_c: float,
    dx: float,
    dy: float,
    harmonic_epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Один convenience step с preparation fixed-design coefficients."""
    coefficients = prepare_explicit_flux_coefficients(
        conductivity,
        dx=dx,
        dy=dy,
        harmonic_epsilon=harmonic_epsilon,
    )
    return explicit_flux_step_prepared(
        temperature,
        coefficients,
        source,
        dt=dt,
        rho_c=rho_c,
    )


def prepare_explicit_flux_coefficients(
    conductivity: torch.Tensor,
    *,
    dx: float,
    dy: float,
    harmonic_epsilon: float = 1.0e-12,
) -> PreparedExplicitFluxCoefficients:
    """Precompute harmonic face coefficients для fixed design."""
    k_x = (
        2.0
        * conductivity[:, :-1]
        * conductivity[:, 1:]
        / (conductivity[:, :-1] + conductivity[:, 1:] + harmonic_epsilon)
    ) / dx**2
    k_y = (
        2.0
        * conductivity[:-1, :]
        * conductivity[1:, :]
        / (conductivity[:-1, :] + conductivity[1:, :] + harmonic_epsilon)
    ) / dy**2
    cooled_bottom = 2.0 * conductivity[0, :] / dy**2
    return PreparedExplicitFluxCoefficients(
        x_face=k_x,
        y_face=k_y,
        cooled_bottom=cooled_bottom,
    )


def explicit_flux_step_prepared(
    temperature: torch.Tensor,
    coefficients: PreparedExplicitFluxCoefficients,
    source: torch.Tensor,
    *,
    dt: float,
    rho_c: float,
) -> torch.Tensor:
    """Один batched step с precomputed fixed-design face coefficients."""
    x_flux = coefficients.x_face.unsqueeze(0) * (
        temperature[:, :, 1:] - temperature[:, :, :-1]
    )
    x_divergence = functional.pad(x_flux, (0, 1)) - functional.pad(x_flux, (1, 0))

    y_flux = coefficients.y_face.unsqueeze(0) * (
        temperature[:, 1:, :] - temperature[:, :-1, :]
    )
    y_divergence = functional.pad(y_flux, (0, 0, 0, 1)) - functional.pad(
        y_flux, (0, 0, 1, 0)
    )

    cooled_bottom = -coefficients.cooled_bottom.unsqueeze(0) * temperature[:, 0, :]
    bottom_divergence = functional.pad(
        cooled_bottom.unsqueeze(1),
        (0, 0, 0, temperature.shape[1] - 1),
    )
    rate = source + x_divergence + y_divergence + bottom_divergence
    return temperature + (dt / rho_c) * rate
