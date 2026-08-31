"""Deterministic exact-cardinality binary readout for fair comparison."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ExactBinaryDiagnostics:
    """Material-budget evidence for one binary readout."""

    selected_cells: int
    total_cells: int
    material_fraction: float


def exact_cardinality_binary(
    design: Tensor,
    count: int = 1024,
) -> tuple[Tensor, ExactBinaryDiagnostics]:
    """Select the highest scores with stable row-major tie-breaking."""
    if design.ndim != 2 or not design.is_floating_point():
        raise ValueError("design must be a two-dimensional floating-point tensor")
    if not bool(torch.isfinite(design).all().item()):
        raise ValueError("design must be finite")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if not 1 <= count <= design.numel():
        raise ValueError("count must lie within the design cell count")

    flat = design.detach().reshape(-1)
    ranking = torch.argsort(flat, descending=True, stable=True)
    binary = torch.zeros_like(flat)
    binary[ranking[:count]] = 1
    fraction = count / flat.numel()
    return binary.reshape_as(design), ExactBinaryDiagnostics(
        selected_cells=count,
        total_cells=flat.numel(),
        material_fraction=fraction,
    )
