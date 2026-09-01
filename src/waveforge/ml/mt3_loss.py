"""Teacher-free best-of-four objective for MT3 candidate training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MT3BestOfFourLoss:
    """Differentiable components of the locked MT3 candidate objective."""

    total: Tensor
    softmin_per_task: Tensor
    diversity_penalty: Tensor


def mt3_best_of_four_loss(
    candidate_totals: Tensor,
    designs: Tensor,
    *,
    softmin_temperature: float = 0.01,
    diversity_weight: float = 0.002,
    diversity_scale: float = 0.10,
) -> MT3BestOfFourLoss:
    """Aggregate four physics objectives without optimized teacher designs."""
    _validate_contract(candidate_totals, designs)
    if softmin_temperature <= 0.0 or diversity_scale <= 0.0:
        raise ValueError("loss scales must be positive")
    if diversity_weight < 0.0:
        raise ValueError("diversity weight must be non-negative")

    softmin_per_task = -softmin_temperature * (
        torch.logsumexp(-candidate_totals / softmin_temperature, dim=1) - math.log(4.0)
    )

    pair_penalties: list[Tensor] = []
    for first in range(4):
        for second in range(first + 1, 4):
            mean_distance = (
                (designs[:, first] - designs[:, second]).abs().mean(dim=(-2, -1))
            )
            pair_penalties.append(torch.exp(-mean_distance / diversity_scale))
    diversity_penalty = diversity_weight * torch.stack(pair_penalties, dim=1).mean()
    total = softmin_per_task.mean() + diversity_penalty
    return MT3BestOfFourLoss(
        total=total,
        softmin_per_task=softmin_per_task,
        diversity_penalty=diversity_penalty,
    )


def _validate_contract(candidate_totals: Tensor, designs: Tensor) -> None:
    if candidate_totals.ndim != 2 or candidate_totals.shape[1] != 4:
        raise ValueError("candidate totals must contain exactly four candidates")
    if designs.ndim != 4 or designs.shape[1] != 4:
        raise ValueError("designs must contain exactly four candidate grids")
    if candidate_totals.shape[0] != designs.shape[0]:
        raise ValueError("candidate totals and designs must share the same batch")
    if not candidate_totals.is_floating_point() or not designs.is_floating_point():
        raise ValueError("candidate totals and designs must be floating-point tensors")
    if not torch.isfinite(candidate_totals).all() or not torch.isfinite(designs).all():
        raise ValueError("candidate totals and designs must be finite")
