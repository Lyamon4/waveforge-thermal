"""Spatial routing and paired workloads; the Stage5 decoder/physics are unchanged."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from waveforge.lab.contracts import Chiplet, ChipletTask
from waveforge.lab.hydraulic_probe import (
    HydraulicProbe,
    refined_spec,
    replicated_design,
)

from .stage3_training import physical_loss as thermal_loss
from .stage5_decoder import ManifoldUNet
from .stage5_evaluation import PUMP_W
from .stage51 import reynolds_samples


def reflected_task(task):
    """Exact Y symmetry of the unchanged centred-port Stage5 operator."""
    chips = tuple(
        Chiplet(c.x, 64 - c.y - c.height, c.width, c.height) for c in task.chiplets
    )
    return ChipletTask(chips, task.powers_w)


def contrast_pair(task, rng):
    """Opposite row workloads on identical silicon; retain all registered ID rules."""
    chips = task.chiplets
    bottom = [i for i, c in enumerate(chips) if c.width == 8 and c.y < 32]
    upper = [i for i, c in enumerate(chips) if c.width == 8 and c.y >= 32]
    io = [i for i, c in enumerate(chips) if c.width == 12]
    if len(bottom) != 4 or len(upper) != 4 or len(io) != 1:
        raise ValueError("eight CCD and one I/O required")
    # Modest within-row variation, strictly positive, each CCD <=80 W.
    hot = rng.dirichlet(np.full(4, 20.0)) * 232.5
    while hot.max() > 80:
        hot = rng.dirichlet(np.full(4, 20.0)) * 232.5
    cold = rng.dirichlet(np.full(4, 20.0)) * 77.5
    pairs = []
    for hot_indices, cold_indices in ((bottom, upper), (upper, bottom)):
        powers = np.full((3, len(chips)), 38.75)
        powers[:, io[0]] = 50
        powers[1, hot_indices], powers[1, cold_indices] = hot, cold
        powers[2, hot_indices], powers[2, cold_indices] = 58.125, 19.375
        pairs.append(ChipletTask(chips, tuple(map(tuple, powers))))
    return tuple(pairs)


class SpatialManifoldUNet(ManifoldUNet):
    """Keep 4x4 encoder positions plus 4x8 raw power/temperature descriptors."""

    def __init__(self):
        super().__init__()
        self.geometry = nn.Linear(128 * 16 + 6 * 32, 4 * 25)
        nn.init.normal_(self.geometry.weight, std=0.001)
        nn.init.zeros_(self.geometry.bias)

    @classmethod
    def from_global(cls, original):
        device = next(original.parameters()).device
        result = cls().to(device)
        inherited = {
            k: v
            for k, v in original.state_dict().items()
            if not k.startswith("geometry.")
        }
        result.load_state_dict(inherited, strict=False)
        with torch.no_grad():
            result.geometry.weight.zero_()
            result.geometry.weight[:, :2048].copy_(
                original.geometry.weight.repeat_interleave(16, 1) / 16
            )
            result.geometry.bias.copy_(original.geometry.bias)
        return result

    def forward(self, condition):
        if condition.ndim != 4 or condition.shape[1:] != (13, 64, 64):
            raise ValueError("conditioning must be [batch,13,64,64]")
        a = self.enc0(condition)
        b = self.enc1(a)
        c = self.enc2(b)
        d = self.enc3(c)
        local = F.adaptive_avg_pool2d(condition[:, :6], (4, 8)).flatten(1)
        feature = torch.cat((F.adaptive_avg_pool2d(d, (4, 4)).flatten(1), local), 1)
        g = self.geometry(feature).reshape(-1, 4, 25)
        geometry = (self.anchors[None, :, :25] + 2 * g.tanh()).clamp(-6, 6)
        up = self.up0(self.up1(self.up2(d, c), b), a)
        pixels = 12 * self.pixels(up).tanh().flatten(2)
        return torch.cat((geometry, pixels), 2)


def balanced_head_loss(losses, tau=0.05):
    if losses.ndim != 2 or losses.shape[1] != 4 or not math.isfinite(tau) or tau <= 0:
        raise ValueError("four finite head losses and positive temperature required")
    softmin = -tau * (torch.logsumexp(-losses / tau, 1) - math.log(4))
    return 0.5 * losses.mean() + 0.5 * softmin.mean()


def fine_flow_penalty(flow, spec):
    """Training margin only; inherited admission remains strictly Re <1200."""
    penalties = []
    for operating in (flow, flow.at_pump_power(PUMP_W)):
        samples = reynolds_samples(operating, spec)
        upper = 5 * torch.logsumexp(samples / 5, dim=1)
        penalties.append(torch.relu(upper / 1100 - 1).square())
    return 4 * torch.stack(penalties).mean()


class AdaptiveObjective:
    """Same 64-grid heat operator plus qualified pressure-only 128-grid gradients."""

    def __init__(self, thermal_solver):
        self.solver = thermal_solver
        self.fine_spec = refined_spec(thermal_solver.spec)
        self.probe = HydraulicProbe(self.fine_spec)

    def __call__(self, design, level=2):
        loss, result = thermal_loss(design, self.solver, level)
        flow = self.probe(replicated_design(design[None]))
        return loss + fine_flow_penalty(flow, self.fine_spec), result
