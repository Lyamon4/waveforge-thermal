"""Explicit Gate 2A material-budget checks."""

from __future__ import annotations

import torch
from torch import Tensor


def material_fraction(design: Tensor) -> float:
    """Return the unweighted mean material fraction."""
    if design.numel() == 0 or not torch.isfinite(design).all():
        raise ValueError("design must be non-empty and finite")
    return float(design.mean().item())


def binary_budget_satisfied(
    design: Tensor,
    *,
    target: float = 0.25,
    tolerance: float = 0.01,
) -> bool:
    """Check the locked inclusive binary-budget interval without repair."""
    if tolerance < 0.0:
        raise ValueError("budget tolerance must be non-negative")
    fraction = material_fraction(design)
    return abs(fraction - target) <= tolerance + 1.0e-12
