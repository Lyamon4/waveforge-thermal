"""Shared five-branch manifold; exact budget and constructive 4-connectivity.

Rasterization and connected fill are discrete. Hard-forward STE is explicitly
biased; smooth projection gradients are only locally valid for a fixed core.
"""

import heapq
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from waveforge.ml.mt3_unet import _ConvBlock, _DownBlock, _UpBlock

from .stage3 import _CoreProjection

KNOTS = (4, 14, 26, 37, 49, 59)
BASE_ROWS = (7, 19, 31, 43, 55)
N_PARAMETERS = 4121


def initial_parameters():
    p = np.zeros(N_PARAMETERS, dtype=np.float64)
    p[20:25] = np.log(1 / 3)  # initial half-width 1.0
    return p


def waypoints(parameters):
    raw = parameters[:, :20].reshape(-1, 5, 4)
    offsets = F.pad(8 * raw.tanh(), (1, 1))
    tv = offsets.diff(dim=-1).abs().sum(-1, keepdim=True)
    offsets = offsets * (12 / tv.clamp_min(12))
    base = parameters.new_tensor(BASE_ROWS)[None, :, None] + 0.5
    return base + offsets


def _core(y):
    """Integer knot raster, <=400+5*(56+16)=760 cells by construction."""
    with torch.no_grad():
        yi = (y - 0.5).round()
        x = torch.arange(4, 60, device=y.device)
        knots = y.new_tensor(KNOTS)
        seg = torch.bucketize(x.contiguous(), knots[1:-1], right=True)
        frac = (x - knots[seg]) / (knots[seg + 1] - knots[seg])
        rows = (yi[..., seg] * (1 - frac) + yi[..., seg + 1] * frac).round()
        previous = torch.cat((rows[..., :1], rows[..., :-1]), -1)
        ys = torch.arange(64, device=y.device)[None, None, None, :]
        paths = (
            (ys >= torch.minimum(rows, previous)[..., None])
            & (ys <= torch.maximum(rows, previous)[..., None])
        ).any(1)
        core = torch.zeros(len(y), 64, 64, dtype=torch.bool, device=y.device)
        core[:, 4:60] = paths
        core[:, :4, 7:57] = True
        core[:, 60:, 7:57] = True
        if bool((core.sum((-2, -1)) > 760).any()):
            raise RuntimeError("mandatory core exceeds proven 760-cell bound")
        return core


def fields(parameters):
    if parameters.ndim != 2 or parameters.shape[1] != N_PARAMETERS:
        raise ValueError("expected [batch,4121] manifold parameters")
    if not torch.isfinite(parameters).all():
        raise ValueError("nonfinite parameters")
    y = waypoints(parameters)
    knots = parameters.new_tensor(KNOTS)
    X, Y = torch.meshgrid(
        torch.arange(64, device=y.device, dtype=y.dtype),
        torch.arange(64, device=y.device, dtype=y.dtype),
        indexing="ij",
    )
    vx = knots.diff()[None, None, :, None, None]
    vy = y.diff(dim=-1)[..., None, None]
    rx = X[None, None, None] - knots[:-1][None, None, :, None, None]
    ry = Y[None, None, None] - y[..., :-1, None, None]
    t = ((rx * vx + ry * vy) / (vx.square() + vy.square())).clamp(0, 1)
    distances = ((rx - t * vx).square() + (ry - t * vy).square() + 1e-12).sqrt().amin(2)
    widths = 0.6 + 1.6 * parameters[:, 20:25].sigmoid()
    score = torch.logsumexp(10 * (widths[..., None, None] - distances), 1) / 10
    pixels = parameters[:, 25:].reshape(-1, 1, 64, 64)
    kernel = parameters.new_tensor([1.0, 2.0, 1.0])
    kernel = (kernel[:, None] * kernel[None, :] / 16)[None, None]
    pixels = F.conv2d(F.pad(pixels, (1, 1, 1, 1), mode="reflect"), kernel)[:, 0]
    score = score + 0.35 * pixels.tanh()
    return score, _core(y), y


def connected_fill(score, core):
    """Deterministic greedy frontier growth; no disconnected islands possible."""
    s = np.asarray(score, dtype=np.float64)
    d = np.asarray(core, dtype=bool).copy()
    if s.shape != (64, 64) or d.shape != s.shape or not np.isfinite(s).all():
        raise ValueError("finite 64x64 scores and core required")
    if not 1 <= d.sum() <= 760:
        raise ValueError("invalid mandatory core size")
    seen = d.copy()
    heap = []

    def offer(x, y):
        if 0 <= x < 64 and 0 <= y < 64 and not seen[x, y]:
            seen[x, y] = True
            heapq.heappush(heap, (-s[x, y], x, y))

    for x, y in np.argwhere(d):
        offer(x - 1, y)
        offer(x + 1, y)
        offer(x, y - 1)
        offer(x, y + 1)
    for _ in range(1024 - int(d.sum())):
        _, x, y = heapq.heappop(heap)
        d[x, y] = True
        offer(x - 1, y)
        offer(x + 1, y)
        offer(x, y - 1)
        offer(x, y + 1)
    return d


def binary(parameters):
    with torch.no_grad():
        score, core, _ = fields(parameters)
        maps = [
            connected_fill(s, c)
            for s, c in zip(score.cpu().numpy(), core.cpu().numpy(), strict=True)
        ]
        return torch.as_tensor(np.stack(maps), device=parameters.device)


def project_all(score, beta):
    return _CoreProjection.apply(score, torch.zeros_like(score, dtype=torch.bool), beta)


def continuous(parameters, beta):
    score, core, _ = fields(parameters)
    return _CoreProjection.apply(score, core, beta)


def density(parameters, *, beta, hard):
    score, core, _ = fields(parameters)
    if not hard:
        forward = _CoreProjection.apply(score, core, beta)
        return forward, forward
    smooth = project_all(score, beta)
    with torch.no_grad():
        if hard:
            maps = [
                connected_fill(s, c)
                for s, c in zip(score.cpu().numpy(), core.cpu().numpy(), strict=True)
            ]
            forward = torch.as_tensor(
                np.stack(maps), dtype=parameters.dtype, device=parameters.device
            )
    return forward + (smooth - smooth.detach()), smooth


class ManifoldUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc0 = _ConvBlock(13, 16)
        self.enc1 = _DownBlock(16, 32)
        self.enc2 = _DownBlock(32, 64)
        self.enc3 = _DownBlock(64, 128)
        self.up2 = _UpBlock(128, 64, 64)
        self.up1 = _UpBlock(64, 32, 32)
        self.up0 = _UpBlock(32, 16, 16)
        self.geometry = nn.Linear(128, 4 * 25)
        self.pixels = nn.Conv2d(16, 4, 1)
        anchor = np.stack([initial_parameters() for _ in range(4)])
        # Different fixed initial hypotheses, not fitted to development outcomes.
        for h, a in enumerate((-0.12, -0.04, 0.04, 0.12)):
            anchor[h, :20] = a
        self.register_buffer("anchors", torch.tensor(anchor, dtype=torch.float32))
        for head in (self.geometry, self.pixels):
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)

    def forward(self, condition):
        if condition.ndim != 4 or condition.shape[1:] != (13, 64, 64):
            raise ValueError("conditioning must be [batch,13,64,64]")
        a = self.enc0(condition)
        b = self.enc1(a)
        c = self.enc2(b)
        d = self.enc3(c)
        g = self.geometry(d.mean((2, 3))).reshape(-1, 4, 25)
        geometry = (self.anchors[None, :, :25] + 2 * g.tanh()).clamp(-6, 6)
        up = self.up0(self.up1(self.up2(d, c), b), a)
        pixels = 12 * self.pixels(up).tanh().flatten(2)
        return torch.cat((geometry, pixels), 2)


def diversity_loss(parameters):
    y = waypoints(parameters.flatten(0, 1)).reshape(-1, 4, 5, 6)
    return torch.stack(
        [
            torch.exp(-(y[:, a] - y[:, b]).square().mean((1, 2)) / 9)
            for a, b in combinations(range(4), 2)
        ]
    ).mean()


def logit_penalty(parameters):
    return parameters[..., :20].square().mean() + parameters[..., 25:].square().mean()
