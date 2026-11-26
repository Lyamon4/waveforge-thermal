"""Frozen refinement-distilled policies and explicit source/student portfolios."""

import torch
import torch.nn.functional as F
from torch import nn

from .ribbed_channels import RibbedDecoder, RibbedUNet


class ResidualRibbedUNet(nn.Module):
    prediction_space = "physical"

    def __init__(self):
        super().__init__()
        self.source = RibbedUNet().eval().requires_grad_(False)
        self.delta_geometry = nn.Sequential(
            nn.Linear(128 * 16 + 6 * 32 + 4 * 29, 128),
            nn.SiLU(),
            nn.Linear(128, 4 * 29),
        )
        channels = self.source.pixels.in_channels
        self.delta_pixels = nn.Sequential(
            nn.Conv2d(channels + 4, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 4, 1),
        )
        for layer in (self.delta_geometry[-1], self.delta_pixels[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        scale = torch.cat(
            (
                torch.ones(20),
                torch.full((5,), 0.75),
                torch.full((4,), 0.75),
                torch.full((4096,), 0.5),
            )
        )
        self.register_buffer("correction_scale", scale.double())

    def train(self, mode=True):
        super().train(mode)
        self.source.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.source.requires_grad_(False)
        return self

    def forward(self, condition):
        s = self.source
        with torch.no_grad():
            a = s.enc0(condition)
            b = s.enc1(a)
            c = s.enc2(b)
            d = s.enc3(c)
            local = F.adaptive_avg_pool2d(condition[:, :6], (4, 8)).flatten(1)
            feature = torch.cat((F.adaptive_avg_pool2d(d, (4, 4)).flatten(1), local), 1)
            raw = s.geometry(feature).reshape(-1, 4, 29)
            geometry = torch.cat(
                (
                    raw[:, :, :20],
                    raw[:, :, 20:25] + s.width_prior[None],
                    raw[:, :, 25:29],
                ),
                2,
            ).clamp(-6, 6)
            up = s.up0(s.up1(s.up2(d, c), b), a)
            original = torch.cat((geometry, 3 * s.pixels(up).tanh().flatten(2)), 2)
            base = RibbedDecoder.to_physical(
                original.flatten(0, 1).double()
            ).reshape_as(original)
        geometry_delta = self.delta_geometry(
            torch.cat((feature, geometry.flatten(1)), 1)
        ).reshape(-1, 4, 29)
        pixel_inputs = torch.cat(
            (up, base[:, :, 29:].float().reshape(-1, 4, 64, 64)), 1
        )
        pixel_delta = self.delta_pixels(pixel_inputs).flatten(2)
        delta = torch.cat((geometry_delta, pixel_delta), 2).tanh().double()
        q = base + delta * self.correction_scale
        return torch.cat(
            (
                q[:, :, :20].clamp(-5.5, 5.5),
                q[:, :, 20:25].clamp(0.6, 2.2),
                q[:, :, 25:29].clamp(-2, 2),
                q[:, :, 29:].clamp(-1, 1),
            ),
            2,
        )


class CandidatePortfolio(nn.Module):
    """Fixed neural proposals; no objective queries or online parameter updates."""

    prediction_space = "physical"

    def __init__(self, proposal_indices=(0, 1, 4, 5)):
        super().__init__()
        if not proposal_indices or any(i not in range(8) for i in proposal_indices):
            raise ValueError("proposal indices must select source/student heads")
        self.source = RibbedUNet()
        self.student = RibbedUNet()
        self.proposal_heads = tuple(i % 4 for i in proposal_indices)
        self.register_buffer("proposal_indices", torch.tensor(proposal_indices))

    def forward(self, condition):
        first = self.source(condition)
        second = self.student(condition)
        raw = torch.cat((first, second), 1)
        physical = RibbedDecoder.to_physical(raw.flatten(0, 1).double()).reshape(
            condition.shape[0], 8, 4125
        )
        return physical[:, self.proposal_indices]


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    architecture = checkpoint.get("architecture", "RibbedUNet")
    if architecture == "ResidualRibbedUNet":
        model = ResidualRibbedUNet()
    elif architecture == "CandidatePortfolio":
        model = CandidatePortfolio(tuple(checkpoint["proposal_indices"]))
    elif architecture == "RibbedUNet":
        model = RibbedUNet()
    else:
        raise ValueError("unsupported ribbed policy architecture")
    model = model.to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().requires_grad_(False)
