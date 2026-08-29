"""Independent PyTorch matrix-free steady heat operator."""

from __future__ import annotations

import torch
from torch import Tensor

from waveforge.physics.grid import Grid2D

_HARMONIC_EPSILON = 1.0e-12


def _validate_inputs(
    temperature: Tensor,
    conductivity: Tensor,
    grid: Grid2D,
) -> None:
    if temperature.ndim not in (2, 3):
        raise ValueError("temperature must have shape [ny,nx] or [batch,ny,nx]")
    if tuple(temperature.shape[-2:]) != grid.shape:
        raise ValueError("temperature spatial shape does not match grid")
    if conductivity.ndim != 2 or tuple(conductivity.shape) != grid.shape:
        raise ValueError("conductivity must have shape [ny,nx]")
    if temperature.device != conductivity.device:
        raise ValueError("temperature and conductivity must share a device")
    if temperature.dtype != conductivity.dtype:
        raise ValueError("temperature and conductivity must share a dtype")
    if not torch.isfinite(temperature).all() or not torch.isfinite(conductivity).all():
        raise ValueError("operator inputs must be finite")
    if not torch.all(conductivity > 0.0):
        raise ValueError("conductivity must be strictly positive")


def _harmonic_faces(first: Tensor, second: Tensor) -> Tensor:
    return 2.0 * first * second / (first + second + _HARMONIC_EPSILON)


def _broadcast_faces(faces: Tensor, target: Tensor) -> Tensor:
    if target.ndim == faces.ndim:
        return faces
    return faces.unsqueeze(0)


def apply_steady_operator(
    temperature: Tensor,
    conductivity: Tensor,
    grid: Grid2D,
) -> Tensor:
    """Apply ``A=-div(k grad)`` with a cooled bottom and insulated other faces."""
    _validate_inputs(temperature, conductivity, grid)
    result = torch.zeros_like(temperature)

    x_faces = _harmonic_faces(conductivity[:, :-1], conductivity[:, 1:])
    x_conductance = _broadcast_faces(x_faces / grid.dx**2, temperature)
    x_jump = temperature[..., :, :-1] - temperature[..., :, 1:]
    result[..., :, :-1] += x_conductance * x_jump
    result[..., :, 1:] -= x_conductance * x_jump

    y_faces = _harmonic_faces(conductivity[:-1, :], conductivity[1:, :])
    y_conductance = _broadcast_faces(y_faces / grid.dy**2, temperature)
    y_jump = temperature[..., :-1, :] - temperature[..., 1:, :]
    result[..., :-1, :] += y_conductance * y_jump
    result[..., 1:, :] -= y_conductance * y_jump

    bottom_conductance = 2.0 * conductivity[0, :] / grid.dy**2
    result[..., 0, :] += bottom_conductance * temperature[..., 0, :]
    return result


def operator_diagonal(conductivity: Tensor, grid: Grid2D) -> Tensor:
    """Return the Jacobi diagonal of the independent matrix-free operator."""
    if conductivity.ndim != 2 or tuple(conductivity.shape) != grid.shape:
        raise ValueError("conductivity must have shape [ny,nx]")
    if not torch.isfinite(conductivity).all():
        raise ValueError("conductivity must be finite")
    if not torch.all(conductivity > 0.0):
        raise ValueError("conductivity must be strictly positive")

    diagonal = torch.zeros_like(conductivity)
    x_conductance = (
        _harmonic_faces(conductivity[:, :-1], conductivity[:, 1:]) / grid.dx**2
    )
    diagonal[:, :-1] += x_conductance
    diagonal[:, 1:] += x_conductance

    y_conductance = (
        _harmonic_faces(conductivity[:-1, :], conductivity[1:, :]) / grid.dy**2
    )
    diagonal[:-1, :] += y_conductance
    diagonal[1:, :] += y_conductance
    diagonal[0, :] += 2.0 * conductivity[0, :] / grid.dy**2
    return diagonal
