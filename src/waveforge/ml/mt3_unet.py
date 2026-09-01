"""Compact deterministic four-head U-Net for WaveForge MT3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from waveforge.design.binary_readout import exact_cardinality_binary
from waveforge.design.parameterization import filter_logits, project_volume


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=True),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=True),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                bias=True,
            ),
        )
        self.block = _ConvBlock(out_channels, out_channels)

    def forward(self, value: Tensor) -> Tensor:
        return self.block(self.down(value))


class _UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=True),
        )
        self.block = _ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        upsampled = functional.interpolate(
            value,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        projected = self.project(upsampled)
        return self.block(torch.cat((projected, skip), dim=1))


class MT3UNet(nn.Module):
    """Four-scale shared decoder with four deterministic material-logit heads."""

    def __init__(self) -> None:
        super().__init__()
        self.enc0 = _ConvBlock(5, 32)
        self.enc1 = _DownBlock(32, 64)
        self.enc2 = _DownBlock(64, 128)
        self.enc3 = _DownBlock(128, 256)
        self.up2 = _UpBlock(256, 128, 128)
        self.up1 = _UpBlock(128, 64, 64)
        self.up0 = _UpBlock(64, 32, 32)
        self.heads = nn.Conv2d(32, 4, kernel_size=1, bias=True)

    def forward(self, condition: Tensor) -> Tensor:
        if condition.ndim != 4 or tuple(condition.shape[1:]) != (5, 64, 64):
            raise ValueError("condition must have shape [batch,5,64,64]")
        if condition.dtype is not torch.float32:
            raise ValueError("condition must use float32 neural precision")
        if not torch.isfinite(condition).all():
            raise ValueError("condition must be finite")
        enc0 = self.enc0(condition)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        decoded = self.up0(self.up1(self.up2(enc3, enc2), enc1), enc0)
        logits = self.heads(decoded)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("MT3 U-Net logits contain NaN or Inf")
        return logits


@dataclass(frozen=True)
class MT3Candidates:
    """Projected continuous and strict binary designs for every candidate head."""

    logits: Tensor
    designs: Tensor
    binary: Tensor
    projection_errors: tuple[float, ...]


def project_mt3_candidates(logits: Tensor, *, beta: float) -> MT3Candidates:
    """Apply the locked filter, volume projection, and top-1024 readout."""
    if logits.ndim != 4 or tuple(logits.shape[1:]) != (4, 64, 64):
        raise ValueError("candidate logits must have shape [batch,4,64,64]")
    if logits.dtype is not torch.float32 or not torch.isfinite(logits).all():
        raise ValueError("candidate logits must be finite float32")

    task_designs: list[Tensor] = []
    task_binary: list[Tensor] = []
    errors: list[float] = []
    for task_logits in logits:
        candidate_designs: list[Tensor] = []
        candidate_binary: list[Tensor] = []
        for candidate_logits in task_logits:
            filtered = filter_logits(
                candidate_logits,
                sigma=1.0,
                radius=3,
                padding="reflect",
            )
            design, diagnostics = project_volume(filtered, beta=beta, target=0.25)
            if not diagnostics.converged or diagnostics.absolute_error > 1.0e-6:
                raise RuntimeError("MT3 candidate projection is invalid")
            binary, binary_diagnostics = exact_cardinality_binary(design, count=1024)
            if binary_diagnostics.selected_cells != 1024:
                raise RuntimeError("MT3 candidate binary budget is invalid")
            candidate_designs.append(design)
            candidate_binary.append(binary)
            errors.append(diagnostics.absolute_error)
        task_designs.append(torch.stack(candidate_designs))
        task_binary.append(torch.stack(candidate_binary))
    return MT3Candidates(
        logits=logits,
        designs=torch.stack(task_designs),
        binary=torch.stack(task_binary),
        projection_errors=tuple(errors),
    )


def count_mt3_parameters(model: nn.Module) -> int:
    """Return the exact number of trainable MT3 generator parameters."""
    return sum(parameter.numel() for parameter in model.parameters())
