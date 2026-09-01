from __future__ import annotations

import math

import pytest
import torch

from waveforge.ml.mt3_loss import mt3_best_of_four_loss


def test_mt3_loss_matches_locked_softmin_and_diversity_formula() -> None:
    totals = torch.tensor(
        [[0.20, 0.21, 0.19, 0.23], [0.31, 0.29, 0.30, 0.28]],
        dtype=torch.float64,
    )
    designs = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.2, 0.2], [0.2, 0.2]],
                [[0.3, 0.3], [0.3, 0.3]],
            ],
            [
                [[0.0, 0.1], [0.2, 0.3]],
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.2, 0.3], [0.4, 0.5]],
                [[0.3, 0.4], [0.5, 0.6]],
            ],
        ],
        dtype=torch.float64,
    )

    result = mt3_best_of_four_loss(totals, designs)

    expected_softmin_per_task = -0.01 * (
        torch.logsumexp(-totals / 0.01, dim=1) - math.log(4.0)
    )
    pair_penalties = []
    for first in range(4):
        for second in range(first + 1, 4):
            distance = (designs[:, first] - designs[:, second]).abs().mean(dim=(-2, -1))
            pair_penalties.append(torch.exp(-distance / 0.10))
    expected_diversity = 0.002 * torch.stack(pair_penalties).mean()
    expected_total = expected_softmin_per_task.mean() + expected_diversity

    torch.testing.assert_close(result.softmin_per_task, expected_softmin_per_task)
    torch.testing.assert_close(result.diversity_penalty, expected_diversity)
    torch.testing.assert_close(result.total, expected_total)


def test_mt3_loss_backpropagates_to_every_candidate() -> None:
    totals = torch.tensor(
        [[0.200, 0.201, 0.202, 0.203], [0.210, 0.211, 0.212, 0.213]],
        dtype=torch.float64,
        requires_grad=True,
    )
    generator = torch.Generator().manual_seed(2026092314)
    designs = torch.rand(
        (2, 4, 8, 8),
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )

    result = mt3_best_of_four_loss(totals, designs)
    result.total.backward()

    assert totals.grad is not None
    assert designs.grad is not None
    assert torch.isfinite(totals.grad).all()
    assert torch.isfinite(designs.grad).all()
    assert torch.all(totals.grad.abs().sum(dim=0) > 0.0)
    assert torch.all(designs.grad.abs().sum(dim=(0, 2, 3)) > 0.0)


@pytest.mark.parametrize(
    ("totals", "designs", "message"),
    [
        (torch.zeros(2, 3), torch.zeros(2, 4, 8, 8), "four"),
        (torch.zeros(2, 4), torch.zeros(2, 3, 8, 8), "four"),
        (torch.zeros(2, 4), torch.zeros(1, 4, 8, 8), "batch"),
        (
            torch.tensor([[0.0, 0.0, float("nan"), 0.0]]),
            torch.zeros(1, 4, 8, 8),
            "finite",
        ),
    ],
)
def test_mt3_loss_rejects_invalid_contract(
    totals: torch.Tensor,
    designs: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mt3_best_of_four_loss(totals, designs)
