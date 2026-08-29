"""Locked low-dimensional Gate 2A design parameterization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import Tensor


class VolumeProjectionError(RuntimeError):
    """Raised when the locked volume projection cannot produce a valid map."""


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Numerical evidence for one deterministic volume projection."""

    offset: float
    iterations: int
    achieved_mean: float
    absolute_error: float
    converged: bool


@dataclass(frozen=True)
class ParameterizedDesign:
    """Intermediate fields retained for reproducibility and diagnostics."""

    upsampled_logits: Tensor
    filtered_logits: Tensor
    design: Tensor
    projection: ProjectionDiagnostics


def _gaussian_kernel_1d(
    sigma: float,
    radius: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    if not torch.isfinite(torch.tensor(sigma)) or sigma <= 0.0:
        raise ValueError("Gaussian sigma must be finite and positive")
    if radius < 0:
        raise ValueError("Gaussian radius must be non-negative")
    coordinates = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float64,
        device=device,
    )
    kernel = torch.exp(-0.5 * (coordinates / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_kernel(
    sigma: float = 1.0,
    radius: int = 3,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the locked normalized separable two-dimensional Gaussian kernel."""
    kernel_1d = _gaussian_kernel_1d(sigma, radius, device=device)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.to(dtype=dtype)


def filter_logits(
    logits: Tensor,
    *,
    sigma: float = 1.0,
    radius: int = 3,
    padding: Literal["reflect"] = "reflect",
) -> Tensor:
    """Apply the locked separable Gaussian filter with reflect padding."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [height,width]")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    if padding != "reflect":
        raise ValueError("Gate 2A requires reflect padding")
    if radius >= min(logits.shape):
        raise ValueError("Gaussian radius must be smaller than each spatial extent")

    kernel_1d = _gaussian_kernel_1d(
        sigma,
        radius,
        device=logits.device,
    ).to(dtype=logits.dtype)
    field = logits[None, None]
    horizontal = kernel_1d.reshape(1, 1, 1, -1)
    vertical = kernel_1d.reshape(1, 1, -1, 1)
    field = functional.pad(field, (radius, radius, 0, 0), mode=padding)
    field = functional.conv2d(field, horizontal)
    field = functional.pad(field, (0, 0, radius, radius), mode=padding)
    return functional.conv2d(field, vertical)[0, 0]


class _ImplicitVolumeProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: Tensor,
        offset: Tensor,
        beta: float,
    ) -> Tensor:
        design = torch.sigmoid(beta * (logits + offset))
        ctx.save_for_backward(design)
        ctx.beta = beta
        return design

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        upstream: Tensor,
    ) -> tuple[Tensor, None, None]:
        (design,) = ctx.saved_tensors
        weights = ctx.beta * design * (1.0 - design)
        denominator = weights.sum()
        if not torch.isfinite(denominator) or denominator <= 1.0e-12:
            raise VolumeProjectionError("implicit projection derivative is singular")
        weighted_mean = torch.sum(upstream * weights) / denominator
        gradient = weights * (upstream - weighted_mean)
        if not torch.isfinite(gradient).all():
            raise VolumeProjectionError("implicit projection gradient is non-finite")
        return gradient, None, None


def project_volume(
    logits: Tensor,
    *,
    beta: float,
    target: float = 0.25,
    bracket: tuple[float, float] = (-40.0, 40.0),
    maximum_iterations: int = 80,
    mean_tolerance: float = 1.0e-6,
) -> tuple[Tensor, ProjectionDiagnostics]:
    """Project filtered logits to the target mean using deterministic bisection."""
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("projection logits must be finite floating-point values")
    if not 0.0 < target < 1.0:
        raise ValueError("target volume must lie strictly between zero and one")
    if not torch.isfinite(torch.tensor(beta)) or beta <= 0.0:
        raise ValueError("projection beta must be finite and positive")
    if maximum_iterations < 1 or mean_tolerance <= 0.0:
        raise ValueError("bisection limits must be positive")
    lower_value, upper_value = bracket
    if not lower_value < upper_value:
        raise ValueError("bisection bracket must be increasing")

    with torch.no_grad():
        lower = logits.new_tensor(lower_value)
        upper = logits.new_tensor(upper_value)
        lower_mean = torch.sigmoid(beta * (logits + lower)).mean()
        upper_mean = torch.sigmoid(beta * (logits + upper)).mean()
        if lower_mean > target or upper_mean < target:
            raise VolumeProjectionError(
                "target volume is outside the bisection bracket"
            )

        for _ in range(maximum_iterations):
            midpoint = 0.5 * (lower + upper)
            midpoint_mean = torch.sigmoid(beta * (logits + midpoint)).mean()
            lower = torch.where(midpoint_mean < target, midpoint, lower)
            upper = torch.where(midpoint_mean >= target, midpoint, upper)
        offset = 0.5 * (lower + upper)

    design = _ImplicitVolumeProjection.apply(logits, offset, beta)
    achieved_mean = float(design.detach().mean().item())
    absolute_error = abs(achieved_mean - target)
    converged = absolute_error <= mean_tolerance
    if not converged:
        raise VolumeProjectionError(
            "volume bisection did not converge within the locked tolerance"
        )
    return design, ProjectionDiagnostics(
        offset=float(offset.item()),
        iterations=maximum_iterations,
        achieved_mean=achieved_mean,
        absolute_error=absolute_error,
        converged=True,
    )


def parameterize_design(
    logits: Tensor,
    *,
    beta: float,
    simulation_shape: tuple[int, int] = (64, 64),
) -> ParameterizedDesign:
    """Map `16×16` design logits to the locked continuous simulation map."""
    if logits.ndim != 2:
        raise ValueError("latent logits must have shape [height,width]")
    upsampled = functional.interpolate(
        logits[None, None],
        size=simulation_shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    filtered = filter_logits(upsampled, sigma=1.0, radius=3, padding="reflect")
    design, diagnostics = project_volume(filtered, beta=beta)
    return ParameterizedDesign(
        upsampled_logits=upsampled,
        filtered_logits=filtered,
        design=design,
        projection=diagnostics,
    )


def binary_design(design: Tensor) -> Tensor:
    """Apply the immutable inclusive `0.5` threshold without budget repair."""
    if not design.is_floating_point() or not torch.isfinite(design).all():
        raise ValueError("design must be a finite floating-point tensor")
    return (design >= 0.5).to(dtype=design.dtype)
