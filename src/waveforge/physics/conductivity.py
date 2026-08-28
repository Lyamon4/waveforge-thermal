"""Conductivity interpolation and harmonic face values."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def harmonic_mean(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Вычислить harmonic conductivity между соседними cells."""
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    return 2.0 * left * right / (left + right + epsilon)


def validate_conductivity(
    conductivity: NDArray[np.float64],
    shape: tuple[int, int],
) -> None:
    """Проверить shape, finite values и строгую положительность k."""
    if conductivity.shape != shape:
        raise ValueError(f"conductivity shape {conductivity.shape} != {shape}")
    if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
        raise ValueError("conductivity must be finite and strictly positive")


def interpolate_conductivity(
    design: NDArray[np.float64],
    k_low: float = 1.0,
    k_high: float = 20.0,
    penalization: float = 3.0,
) -> NDArray[np.float64]:
    """Преобразовать relaxed design в conductivity по SIMP-like law."""
    if not np.all(np.isfinite(design)) or np.any((design < 0.0) | (design > 1.0)):
        raise ValueError("design must contain finite values in [0, 1]")
    if k_low <= 0.0 or k_high <= k_low:
        raise ValueError("conductivity bounds must satisfy 0 < k_low < k_high")
    if penalization <= 0.0:
        raise ValueError("penalization must be positive")
    return k_low + (k_high - k_low) * design**penalization
