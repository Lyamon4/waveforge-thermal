"""Four-condition-channel NCA used by both matched MT2B variants."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from waveforge.ml.nca import NCARollout


class MT2BNCA(nn.Module):
    """Unchanged local recurrent rule with a matched four-channel input."""

    def __init__(self) -> None:
        super().__init__()
        self.perception = nn.Conv2d(
            20,
            64,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.update = nn.Conv2d(64, 16, kernel_size=1)
        nn.init.zeros_(self.update.weight)
        nn.init.zeros_(self.update.bias)

    def step(self, state: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one synchronous update with persistent conditioning."""
        features = functional.silu(
            self.perception(torch.cat((state, condition), dim=1))
        )
        delta = 0.1 * torch.tanh(self.update(features))
        return state + delta, delta

    def rollout(
        self,
        condition: Tensor,
        *,
        steps: int = 64,
        snapshot_steps: tuple[int, ...] = (),
    ) -> NCARollout:
        """Start from exact zeros and execute exactly 64 shared steps."""
        if condition.ndim != 4 or tuple(condition.shape[1:]) != (4, 64, 64):
            raise ValueError("condition must have shape [batch,4,64,64]")
        if condition.dtype is not torch.float32:
            raise ValueError("NCA condition must be float32")
        if not torch.isfinite(condition).all():
            raise ValueError("NCA condition must be finite")
        if steps != 64:
            raise ValueError("MT2B NCA rollout requires exactly 64 steps")
        if any(
            isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= 64
            for step in snapshot_steps
        ):
            raise ValueError("snapshot steps must be integer values in [0,64]")

        requested = set(snapshot_steps)
        state = condition.new_zeros((condition.shape[0], 16, 64, 64))
        snapshots: dict[int, Tensor] = {}
        if 0 in requested:
            snapshots[0] = state.detach().clone()
        maximum_delta = condition.new_zeros(())
        maximum_state = condition.new_zeros(())
        delta_square_sum = condition.new_zeros(())
        delta_element_count = 0

        for step_index in range(1, steps + 1):
            state, delta = self.step(state, condition)
            maximum_delta = torch.maximum(maximum_delta, delta.detach().abs().amax())
            maximum_state = torch.maximum(maximum_state, state.detach().abs().amax())
            delta_square_sum = delta_square_sum + delta.detach().square().sum()
            delta_element_count += delta.numel()
            if step_index in requested:
                snapshots[step_index] = state.detach().clone()

        hidden_state = state[:, 1:16]
        hidden_state_rms = torch.sqrt(hidden_state.detach().square().mean())
        delta_state_rms = torch.sqrt(delta_square_sum / delta_element_count)
        return NCARollout(
            final_state=state,
            material_logit=state[:, 0:1],
            hidden_state=hidden_state,
            snapshots=snapshots,
            hidden_state_rms=float(hidden_state_rms.item()),
            delta_state_rms=float(delta_state_rms.item()),
            maximum_absolute_delta=float(maximum_delta.item()),
            maximum_absolute_state=float(maximum_state.item()),
        )
