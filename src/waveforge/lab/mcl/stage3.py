"""Stage 3 wrappers; the qualified physical solver and decoder remain unchanged."""

from itertools import combinations

import numpy as np
import torch
from torch import nn

from waveforge.design.parameterization import filter_logits
from waveforge.ml.mt3_unet import _ConvBlock, _DownBlock, _UpBlock

from .vascular_stage12 import limited_waypoints, reserve_core, smooth_scores

ANCHORS = ((13, 25, 38, 50), (10, 16, 32, 49), (13, 31, 46, 52), (12, 28, 35, 51))


def anchor_parameters():
    rows = []
    for heights in ANCHORS:
        raw = np.zeros((4, 4))
        for branch, target in enumerate(heights):
            previous = 31.5
            for knot in range(4):
                low = max(6.0, 31.5 - 16 * (4 - knot))
                high = min(57.0, 31.5 + 16 * (4 - knot))
                y = np.clip(target, low, high)
                delta = np.clip((y - previous) / 16, -0.995, 0.995)
                raw[branch, knot] = np.arctanh(delta)
                previous = np.clip(previous + 16 * delta, low, high)
        rows.append(np.concatenate((raw.ravel(), np.zeros(4100))))
    return torch.tensor(np.stack(rows), dtype=torch.float64)


def halo_residual(pixel, distance):
    # Bounding u is necessary: an unbounded u defeats any finite sigmoid halo.
    u = pixel.tanh()
    return u * torch.sigmoid(6 * (1.45 - distance)) - torch.sigmoid(
        6 * (distance - 1.45)
    )


def fields(parameters):
    if parameters.ndim != 2 or parameters.shape[1] != 4116:
        raise ValueError("parameters must be [batch,4116]")
    waypoints = limited_waypoints(parameters[:, :16].reshape(-1, 4, 4))
    widths = 1 + 3 * parameters[:, 16:20].sigmoid()
    score, distances = smooth_scores(waypoints, widths)
    nearest = distances.amin(1)
    pixels = torch.stack([filter_logits(p.reshape(64, 64)) for p in parameters[:, 20:]])
    residual = halo_residual(pixels, nearest)
    # Identical residual term to the frozen smooth_scores, using its distances.
    score = score + 0.35 * residual.tanh()
    return score, nearest <= 0.75, waypoints, residual


class _CoreProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score, core, beta):
        if (
            score.shape != core.shape
            or core.dtype != torch.bool
            or score.ndim != 3
            or score.shape[-2:] != (64, 64)
            or not torch.isfinite(score).all()
            or beta <= 0
        ):
            raise ValueError("finite scores, boolean core and positive beta required")
        count = core.sum((1, 2), keepdim=True)
        if bool((count > 1024).any()):
            raise ValueError("core exceeds 1024 budget")
        remaining = 1024 - count
        lo = -score.amax((1, 2), keepdim=True) - 40 / beta
        hi = -score.amin((1, 2), keepdim=True) + 40 / beta
        for _ in range(80):
            mid = (lo + hi) / 2
            mass = (torch.sigmoid(beta * (score + mid)) * ~core).sum(
                (1, 2), keepdim=True
            )
            lo = torch.where(mass < remaining, mid, lo)
            hi = torch.where(mass >= remaining, mid, hi)
        outside = torch.sigmoid(beta * (score + (lo + hi) / 2))
        outside = torch.where(remaining == 0, 0.0, outside)
        density = torch.where(core, 1.0, outside)
        ctx.save_for_backward(density, core)
        ctx.beta = beta
        return density

    @staticmethod
    def backward(ctx, incoming):
        density, core = ctx.saved_tensors
        slope = ctx.beta * density * (1 - density) * ~core
        denominator = slope.sum((1, 2), keepdim=True).clamp_min(1e-30)
        average = (incoming * slope).sum((1, 2), keepdim=True) / denominator
        # Implicit derivative of the volume constraint for a FIXED core.
        return slope * (incoming - average), None, None


def continuous(parameters, beta=4.0):
    score, core, _, _ = fields(parameters)
    return _CoreProjection.apply(score, core, beta)


def binary(parameters):
    with torch.no_grad():
        score, core, _, _ = fields(parameters)
        return reserve_core(score, core)[0]


class VascularUNet(nn.Module):
    """13-channel encoder with four anchored geometry and pixel output heads."""

    def __init__(self):
        super().__init__()
        self.enc0 = _ConvBlock(13, 16)
        self.enc1 = _DownBlock(16, 32)
        self.enc2 = _DownBlock(32, 64)
        self.enc3 = _DownBlock(64, 128)
        self.up2 = _UpBlock(128, 64, 64)
        self.up1 = _UpBlock(64, 32, 32)
        self.up0 = _UpBlock(32, 16, 16)
        self.geometry = nn.Linear(128, 4 * 20)
        self.pixels = nn.Conv2d(16, 4, 1)
        self.register_buffer("anchors", anchor_parameters().float())
        for head in (self.geometry, self.pixels):
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)

    def forward(self, condition):
        if condition.ndim != 4 or condition.shape[1:] != (13, 64, 64):
            raise ValueError("conditioning must be [batch,13,64,64]")
        first = self.enc0(condition)
        second = self.enc1(first)
        third = self.enc2(second)
        fourth = self.enc3(third)
        geometry = self.geometry(fourth.mean((2, 3))).reshape(-1, 4, 20)
        increments = (self.anchors[None, :, :16] + geometry[:, :, :16].tanh()).clamp(
            -6, 6
        )
        widths = 6 * geometry[:, :, 16:20].tanh()
        up = self.up0(self.up1(self.up2(fourth, third), second), first)
        pixels = 12 * self.pixels(up).tanh().flatten(2)
        return torch.cat((increments, widths, pixels), dim=2)


def diversity_loss(parameters):
    trajectories = limited_waypoints(parameters[..., :16].reshape(-1, 4, 4))
    trajectories = trajectories.reshape(parameters.shape[0], 4, 4, 4)
    return torch.stack(
        [
            torch.exp(
                -(trajectories[:, a] - trajectories[:, b]).square().mean((1, 2)) / 36
            )
            for a, b in combinations(range(4), 2)
        ]
    ).mean()
