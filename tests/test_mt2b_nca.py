from __future__ import annotations

import pytest
import torch

from waveforge.ml.mt2b_nca import MT2BNCA


def test_mt2b_nca_has_exact_locked_parameter_count_and_zero_output_layer() -> None:
    model = MT2BNCA()

    assert sum(parameter.numel() for parameter in model.parameters()) == 12624
    assert model.perception.in_channels == 20
    assert torch.count_nonzero(model.update.weight) == 0
    assert torch.count_nonzero(model.update.bias) == 0


def test_mt2b_nca_zero_initialization_produces_exact_zero_initial_rollout() -> None:
    model = MT2BNCA()
    condition = torch.randn((2, 4, 64, 64), dtype=torch.float32)

    rollout = model.rollout(condition, snapshot_steps=(0, 1, 64))

    assert rollout.final_state.shape == (2, 16, 64, 64)
    assert torch.count_nonzero(rollout.final_state) == 0
    assert torch.count_nonzero(rollout.snapshots[0]) == 0
    assert torch.count_nonzero(rollout.snapshots[1]) == 0
    assert torch.count_nonzero(rollout.snapshots[64]) == 0


def test_mt2b_nca_reuses_condition_at_every_step() -> None:
    model = MT2BNCA()
    with torch.no_grad():
        model.update.weight.fill_(0.01)
    condition = torch.ones((1, 4, 64, 64), dtype=torch.float32)
    calls = 0

    def count_condition(_module, inputs):
        nonlocal calls
        assert inputs[0].shape[1] == 20
        torch.testing.assert_close(inputs[0][:, 16:], condition)
        calls += 1

    handle = model.perception.register_forward_pre_hook(count_condition)
    try:
        model.rollout(condition)
    finally:
        handle.remove()

    assert calls == 64


@pytest.mark.parametrize(
    "condition",
    [
        torch.zeros((1, 2, 64, 64), dtype=torch.float32),
        torch.zeros((1, 4, 64, 64), dtype=torch.float64),
        torch.full((1, 4, 64, 64), float("inf"), dtype=torch.float32),
    ],
)
def test_mt2b_nca_rejects_invalid_condition(condition: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        MT2BNCA().rollout(condition)
