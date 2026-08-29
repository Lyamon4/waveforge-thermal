"""Pure local neural cellular automaton for physics-trained design."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from waveforge.design.parameterization import (
    ProjectionDiagnostics,
    filter_logits,
    project_volume,
)


@dataclass(frozen=True)
class NCARollout:
    """Final differentiable state and detached structural diagnostics."""

    final_state: Tensor
    material_logit: Tensor
    hidden_state: Tensor
    snapshots: dict[int, Tensor]
    hidden_state_rms: float
    delta_state_rms: float
    maximum_absolute_delta: float
    maximum_absolute_state: float


@dataclass(frozen=True)
class NCAProjectedDesign:
    """Locked filtered and exactly volume-projected NCA material readout."""

    material_logit: Tensor
    filtered_logits: Tensor
    design: Tensor
    projection: ProjectionDiagnostics


def build_static_condition(sources: Tensor) -> Tensor:
    """Aggregate three physical sources and append the immutable sink mask."""
    if sources.shape != (3, 64, 64):
        raise ValueError("sources must have shape [3,64,64]")
    if sources.dtype is not torch.float64:
        raise ValueError("physical sources must be float64")
    if not torch.isfinite(sources).all():
        raise ValueError("physical sources must be finite")

    source_condition = sources.sum(dim=0).to(dtype=torch.float32) / 25.0
    sink_mask = torch.zeros_like(source_condition)
    sink_mask[0, :] = 1.0
    return torch.stack((source_condition, sink_mask), dim=0).unsqueeze(0)


def project_nca_material(material_logit: Tensor) -> NCAProjectedDesign:
    """Map the final NCA material channel to the fixed-budget design."""
    if material_logit.shape != (1, 1, 64, 64):
        raise ValueError("material_logit must have shape [1,1,64,64]")
    if material_logit.dtype is not torch.float32:
        raise ValueError("material_logit must be float32")
    if not torch.isfinite(material_logit).all():
        raise ValueError("material_logit must be finite")
    material_field = material_logit[0, 0]
    filtered = filter_logits(
        material_field,
        sigma=1.0,
        radius=3,
        padding="reflect",
    )
    design, diagnostics = project_volume(
        filtered,
        beta=8.0,
        target=0.25,
        bracket=(-40.0, 40.0),
        maximum_iterations=80,
        mean_tolerance=1.0e-6,
    )
    return NCAProjectedDesign(
        material_logit=material_field,
        filtered_logits=filtered,
        design=design,
        projection=diagnostics,
    )


class PureNCA(nn.Module):
    """Shared 64-step local rule with exact-zero mutable initialization."""

    def __init__(self) -> None:
        super().__init__()
        self.perception = nn.Conv2d(
            18,
            64,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.update = nn.Conv2d(64, 16, kernel_size=1)
        nn.init.zeros_(self.update.weight)
        nn.init.zeros_(self.update.bias)

    def step(self, state: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one synchronous residual NCA update."""
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
        """Start from exact zeros and execute the locked synchronous rollout."""
        if condition.ndim != 4 or tuple(condition.shape[1:]) != (2, 64, 64):
            raise ValueError("condition must have shape [batch,2,64,64]")
        if condition.dtype is not torch.float32:
            raise ValueError("NCA condition must be float32")
        if not torch.isfinite(condition).all():
            raise ValueError("NCA condition must be finite")
        if steps != 64:
            raise ValueError("pure-NCA rollout requires exactly 64 steps")
        if any(
            isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= 64
            for step in snapshot_steps
        ):
            raise ValueError("snapshot steps must be integer values in [0,64]")

        requested_snapshots = set(snapshot_steps)
        state = condition.new_zeros((condition.shape[0], 16, 64, 64))
        snapshots: dict[int, Tensor] = {}
        if 0 in requested_snapshots:
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
            if step_index in requested_snapshots:
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
