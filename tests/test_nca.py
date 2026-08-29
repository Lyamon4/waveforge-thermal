"""Проверки locked physical conditioning и pure NCA core."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from waveforge.experiments.run_inverse_design import gate2_source_batch
from waveforge.ml.nca import PureNCA, build_static_condition

SNAPSHOT_STEPS = (0, 1, 2, 4, 8, 16, 32, 48, 64)


def test_condition_uses_fixed_scale_sum_and_bottom_sink() -> None:
    sources = gate2_source_batch(device=torch.device("cpu"))

    condition = build_static_condition(sources)

    expected_source = sources.sum(dim=0).to(torch.float32) / 25.0
    torch.testing.assert_close(condition[0, 0], expected_source)
    assert condition.shape == (1, 2, 64, 64)
    assert condition.dtype is torch.float32
    assert torch.all(condition[0, 1, 0, :] == 1.0)
    assert torch.count_nonzero(condition[0, 1, 1:, :]) == 0


def test_condition_is_source_permutation_invariant_and_does_not_clamp() -> None:
    sources = torch.zeros((3, 64, 64), dtype=torch.float64)
    sources[:, 10, 10] = 25.0

    first = build_static_condition(sources)
    second = build_static_condition(sources[[2, 0, 1]])

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first[0, 0, 10, 10].item() == 3.0


@pytest.mark.parametrize(
    "sources",
    [
        torch.zeros((2, 64, 64), dtype=torch.float64),
        torch.zeros((3, 32, 32), dtype=torch.float64),
        torch.zeros((3, 64, 64), dtype=torch.float32),
    ],
)
def test_condition_rejects_wrong_scenario_shape_or_dtype(sources: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        build_static_condition(sources)


def test_condition_rejects_nonfinite_source() -> None:
    sources = torch.zeros((3, 64, 64), dtype=torch.float64)
    sources[0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        build_static_condition(sources)


def test_pure_nca_architecture_and_initialization_are_exact() -> None:
    torch.manual_seed(20260831)

    model = PureNCA()

    assert model.perception.in_channels == 18
    assert model.perception.out_channels == 64
    assert model.perception.kernel_size == (3, 3)
    assert model.perception.padding_mode == "reflect"
    assert model.update.in_channels == 64
    assert model.update.out_channels == 16
    assert model.update.kernel_size == (1, 1)
    assert sum(parameter.numel() for parameter in model.parameters()) == 11472
    assert torch.count_nonzero(model.update.weight) == 0
    assert torch.count_nonzero(model.update.bias) == 0
    forbidden = (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.Dropout)
    assert not any(isinstance(module, forbidden) for module in model.modules())


def test_zero_initialized_rollout_is_exactly_zero() -> None:
    model = PureNCA()
    condition = torch.zeros((1, 2, 64, 64), dtype=torch.float32)

    rollout = model.rollout(condition, snapshot_steps=SNAPSHOT_STEPS)

    assert torch.count_nonzero(rollout.final_state) == 0
    assert torch.count_nonzero(rollout.material_logit) == 0
    assert rollout.maximum_absolute_delta == 0.0
    assert rollout.maximum_absolute_state == 0.0
    assert tuple(rollout.snapshots) == SNAPSHOT_STEPS
    assert all(not snapshot.requires_grad for snapshot in rollout.snapshots.values())


def test_rollout_respects_update_and_accumulated_state_bounds() -> None:
    model = PureNCA()
    with torch.no_grad():
        model.update.bias.fill_(1.0)
    condition = torch.zeros((1, 2, 64, 64), dtype=torch.float32)

    rollout = model.rollout(condition, snapshot_steps=SNAPSHOT_STEPS)

    expected_delta = 0.1 * torch.tanh(torch.tensor(1.0)).item()
    assert rollout.maximum_absolute_delta == pytest.approx(expected_delta)
    assert rollout.maximum_absolute_delta <= 0.100001
    assert rollout.maximum_absolute_state == pytest.approx(64.0 * expected_delta)
    assert rollout.maximum_absolute_state <= 6.4001
    assert rollout.final_state.shape == (1, 16, 64, 64)
    assert rollout.material_logit.shape == (1, 1, 64, 64)
    assert rollout.hidden_state.shape == (1, 15, 64, 64)
    torch.testing.assert_close(
        rollout.snapshots[64],
        rollout.final_state.detach(),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "condition",
    [
        torch.zeros((2, 64, 64), dtype=torch.float32),
        torch.zeros((1, 3, 64, 64), dtype=torch.float32),
        torch.zeros((1, 2, 32, 32), dtype=torch.float32),
        torch.zeros((1, 2, 64, 64), dtype=torch.float64),
    ],
)
def test_rollout_rejects_invalid_condition(condition: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        PureNCA().rollout(condition)


def test_rollout_rejects_unregistered_step_count_or_snapshot() -> None:
    condition = torch.zeros((1, 2, 64, 64), dtype=torch.float32)
    model = PureNCA()

    with pytest.raises(ValueError, match="64"):
        model.rollout(condition, steps=63)
    with pytest.raises(ValueError, match="snapshot"):
        model.rollout(condition, snapshot_steps=(0, 65))
