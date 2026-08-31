"""Matched RAW and physics-transformed conditioning for NCA-MT2B."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

FIXED_TEMPERATURE_SCALE = 0.900613256638055
TemperatureSolver = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def canonical_temperature_scale() -> float:
    """Return the immutable, non-sample-derived temperature scale."""
    return FIXED_TEMPERATURE_SCALE


def _validate_sources(sources: Tensor) -> None:
    if sources.ndim != 4 or tuple(sources.shape[1:]) != (3, 64, 64):
        raise ValueError("sources must have shape [batch,3,64,64]")
    if sources.dtype is not torch.float64:
        raise ValueError("physical sources must be float64")
    if not torch.isfinite(sources).all():
        raise ValueError("physical sources must be finite")


def build_mt2b_conditioning(
    sources: Tensor,
    *,
    variant: Literal["RAW", "PHYSICS"],
    temperature_solver: TemperatureSolver | None = None,
) -> Tensor:
    """Build four persistent channels with no task-specific normalization."""
    _validate_sources(sources)
    detached = sources.detach()
    source_sum = detached.sum(dim=1).to(dtype=torch.float32) / 25.0
    sink = torch.zeros_like(source_sum)
    sink[:, 0, :] = 1.0

    if variant == "RAW":
        zeros = torch.zeros_like(source_sum)
        return torch.stack((source_sum, zeros, zeros, sink), dim=1)
    if variant != "PHYSICS":
        raise ValueError(f"unknown MT2B conditioning variant {variant!r}")
    if temperature_solver is None:
        raise ValueError("PHYSICS conditioning requires a temperature_solver")

    source_array = detached.cpu().numpy()
    temperature_fields = np.asarray(temperature_solver(source_array), dtype=np.float64)
    if temperature_fields.shape != source_array.shape:
        raise ValueError("temperature_solver must return shape [batch,3,64,64]")
    if not np.all(np.isfinite(temperature_fields)):
        raise FloatingPointError("conditioning temperature fields must be finite")

    field_mean = torch.from_numpy(temperature_fields.mean(axis=1)).to(
        device=sources.device, dtype=torch.float32
    )
    field_max = torch.from_numpy(temperature_fields.max(axis=1)).to(
        device=sources.device, dtype=torch.float32
    )
    field_mean = field_mean / FIXED_TEMPERATURE_SCALE
    field_max = field_max / FIXED_TEMPERATURE_SCALE
    return torch.stack((source_sum, field_mean, field_max, sink), dim=1).detach()
