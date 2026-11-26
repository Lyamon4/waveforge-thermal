"""Literal, separately logged Gate 2A objective components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ObjectiveComponents:
    """Thermal and geometric terms retained without hidden aggregation."""

    thermal_smooth: Tensor
    exact_peak: Tensor
    total_variation: Tensor
    binarization_penalty: Tensor
    total: Tensor


def normalized_smooth_max(values: Tensor, *, alpha: float) -> Tensor:
    """Compute the locked numerically stable normalized log-mean-exp."""
    if values.numel() == 0 or not values.is_floating_point():
        raise ValueError("smooth maximum requires a non-empty floating tensor")
    if not torch.isfinite(values).all():
        raise ValueError("smooth maximum values must be finite")
    if not torch.isfinite(torch.tensor(alpha)) or alpha <= 0.0:
        raise ValueError("smooth maximum alpha must be finite and positive")
    maximum = torch.max(values)
    normalized_exponential = torch.mean(torch.exp(alpha * (values - maximum)))
    return maximum + torch.log(normalized_exponential) / alpha


def total_variation(design: Tensor) -> Tensor:
    """Compute the locked mean-absolute horizontal plus vertical variation."""
    if design.ndim != 2 or min(design.shape) < 2:
        raise ValueError("TV requires a two-dimensional design with both extents >= 2")
    if not design.is_floating_point() or not torch.isfinite(design).all():
        raise ValueError("TV design must be finite floating point")
    horizontal = torch.mean(torch.abs(design[:, 1:] - design[:, :-1]))
    vertical = torch.mean(torch.abs(design[1:, :] - design[:-1, :]))
    return horizontal + vertical


def objective_components(
    temperatures: Tensor,
    design: Tensor,
    *,
    alpha: float,
    tv_weight: float = 1.0e-3,
    binarization_weight: float = 0.0,
) -> ObjectiveComponents:
    """Build the thermal objective while retaining every direct regularizer."""
    if temperatures.device != design.device:
        raise ValueError("temperatures and design must share a device")
    if tv_weight < 0.0 or binarization_weight < 0.0:
        raise ValueError("objective weights must be non-negative")

    thermal_smooth = normalized_smooth_max(temperatures, alpha=alpha)
    exact_peak = torch.max(temperatures)
    variation = total_variation(design)
    binarization = torch.mean(design * (1.0 - design))
    target_dtype = thermal_smooth.dtype
    total = (
        thermal_smooth
        + tv_weight * variation.to(dtype=target_dtype)
        + binarization_weight * binarization.to(dtype=target_dtype)
    )
    if not torch.isfinite(total):
        raise FloatingPointError("Gate 2A objective is non-finite")
    return ObjectiveComponents(
        thermal_smooth=thermal_smooth,
        exact_peak=exact_peak,
        total_variation=variation,
        binarization_penalty=binarization,
        total=total,
    )
