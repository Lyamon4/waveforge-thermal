"""Unchanged physical operators, shared dual-regime penalty and admission."""

import numpy as np
import torch
import torch.nn.functional as F

from waveforge.lab.hydraulic_probe import (
    HydraulicProbe,
    refined_spec,
    replicated_design,
)

from .adaptive_manifold import SpatialManifoldUNet
from .locked_channels import WIDTH_PRIORS
from .stage3_training import physical_loss as thermal_loss
from .stage5_admission import admission
from .stage5_evaluation import PUMP_W
from .stage51 import reynolds_samples

FLOW_WEIGHT = 0.01
FLOW_TAU = 5.0


def joint_reynolds(flow, spec):
    nominal = reynolds_samples(flow, spec)
    power = reynolds_samples(flow.at_pump_power(PUMP_W), spec)
    n = nominal.amax(1)
    p = power.amax(1)
    smooth = FLOW_TAU * torch.logsumexp(torch.cat((nominal, power), 1) / FLOW_TAU, 1)
    return dict(nominal=n, power=p, joint=torch.maximum(n, p), smooth=smooth)


def joint_flow_penalty(flow, spec):
    re = joint_reynolds(flow, spec)["smooth"]
    return FLOW_WEIGHT * F.softplus((re - 1125.0) / 25.0).square().mean()


def joint_admission(modes):
    checks = {
        f"{grid}:{mode}": admission(modes[str(grid)][mode])
        for grid in (64, 128)
        for mode in ("nominal", "power")
    }
    reasons = [
        f"{key}:{reason}"
        for key, value in checks.items()
        for reason in value["rejection_reasons"]
    ]
    return dict(admissible=not reasons, rejection_reasons=reasons, checks=checks)


class JointObjective:
    def __init__(self, solver):
        self.solver = solver
        self.fine_spec = refined_spec(solver.spec)
        self.probe = HydraulicProbe(self.fine_spec)

    def __call__(self, design, level=2):
        loss, result = thermal_loss(design, self.solver, level)
        flow = self.probe(replicated_design(design[None]))
        return loss + joint_flow_penalty(flow, self.fine_spec), result


def head_loss(values):
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("four head losses required")
    return 0.5 * values.min(1).values.mean() + 0.5 * values.mean()


class LockedUNet(SpatialManifoldUNet):
    """Residual action logits; geometry comes from each task's chiplet mask."""

    def __init__(self):
        super().__init__()
        del self.anchors
        width = (WIDTH_PRIORS - 0.6) / 1.6
        self.register_buffer(
            "width_prior",
            torch.tensor(np.log(width / (1 - width)), dtype=torch.float32),
        )
        torch.nn.init.zeros_(self.geometry.weight)
        torch.nn.init.zeros_(self.geometry.bias)
        torch.nn.init.zeros_(self.pixels.weight)
        torch.nn.init.zeros_(self.pixels.bias)

    def forward(self, condition):
        if condition.ndim != 4 or condition.shape[1:] != (13, 64, 64):
            raise ValueError("13channel64x64 inputs required")
        a = self.enc0(condition)
        b = self.enc1(a)
        c = self.enc2(b)
        d = self.enc3(c)
        local = F.adaptive_avg_pool2d(condition[:, :6], (4, 8)).flatten(1)
        features = torch.cat((F.adaptive_avg_pool2d(d, (4, 4)).flatten(1), local), 1)
        raw = self.geometry(features).reshape(-1, 4, 25)
        geometry = torch.cat(
            (raw[:, :, :20], raw[:, :, 20:] + self.width_prior[None]), 2
        ).clamp(-6, 6)
        up = self.up0(self.up1(self.up2(d, c), b), a)
        pixels = 12 * self.pixels(up).tanh().flatten(2)
        return torch.cat((geometry, pixels), 2)
