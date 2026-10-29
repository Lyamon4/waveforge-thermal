"""Vectorized material projection for opt-in MT3 execution."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor

from waveforge.design.parameterization import VolumeProjectionError

_GAUSSIAN_SIGMA = 1.0
_GAUSSIAN_RADIUS = 3
_TARGET_FRACTION = 0.25
_BINARY_COUNT = 1024
_BISECTION_ITERATIONS = 80
_MEAN_TOLERANCE = 1.0e-6
_BRACKET = (-40.0, 40.0)


def _filter_candidate_maps(logits: Tensor) -> Tensor:
    coordinates = torch.arange(
        -_GAUSSIAN_RADIUS,
        _GAUSSIAN_RADIUS + 1,
        dtype=torch.float64,
        device=logits.device,
    )
    kernel = torch.exp(-0.5 * (coordinates / _GAUSSIAN_SIGMA) ** 2)
    kernel = (kernel / kernel.sum()).to(dtype=logits.dtype)

    batch, candidates, height, width = logits.shape
    field = logits.reshape(batch * candidates, 1, height, width)
    horizontal = kernel.reshape(1, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1)
    field = functional.pad(
        field,
        (_GAUSSIAN_RADIUS, _GAUSSIAN_RADIUS, 0, 0),
        mode="reflect",
    )
    field = functional.conv2d(field, horizontal)
    field = functional.pad(
        field,
        (0, 0, _GAUSSIAN_RADIUS, _GAUSSIAN_RADIUS),
        mode="reflect",
    )
    return functional.conv2d(field, vertical).reshape(
        batch,
        candidates,
        height,
        width,
    )


class _ImplicitBatchedVolumeProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: Tensor,
        offsets: Tensor,
        beta: float,
    ) -> Tensor:
        design = torch.sigmoid(beta * (logits + offsets))
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
        denominator = weights.sum(dim=(-2, -1), keepdim=True)
        if not torch.isfinite(denominator).all() or torch.any(denominator <= 1.0e-12):
            raise VolumeProjectionError("implicit projection derivative is singular")
        weighted_mean = (upstream * weights).sum(
            dim=(-2, -1),
            keepdim=True,
        ) / denominator
        gradient = weights * (upstream - weighted_mean)
        if not torch.isfinite(gradient).all():
            raise VolumeProjectionError("implicit projection gradient is non-finite")
        return gradient, None, None


def _project_filtered_maps(
    filtered_logits: Tensor,
    *,
    beta: float,
) -> tuple[Tensor, Tensor]:
    lower_value, upper_value = _BRACKET
    map_shape = (*filtered_logits.shape[:2], 1, 1)
    with torch.no_grad():
        lower = filtered_logits.new_full(map_shape, lower_value)
        upper = filtered_logits.new_full(map_shape, upper_value)
        lower_mean = torch.sigmoid(beta * (filtered_logits + lower)).mean(
            dim=(-2, -1),
            keepdim=True,
        )
        upper_mean = torch.sigmoid(beta * (filtered_logits + upper)).mean(
            dim=(-2, -1),
            keepdim=True,
        )
        if torch.any(lower_mean > _TARGET_FRACTION) or torch.any(
            upper_mean < _TARGET_FRACTION
        ):
            raise VolumeProjectionError(
                "target volume is outside the bisection bracket"
            )

        from waveforge.runtime import runtime_options

        if runtime_options().accelerated_projection:
            from waveforge.design.projection_graph import replay_bisection

            offsets = replay_bisection(
                filtered_logits,
                beta=beta,
                target=_TARGET_FRACTION,
                bracket=_BRACKET,
                iterations=_BISECTION_ITERATIONS,
                batched=True,
            )
        else:
            for _ in range(_BISECTION_ITERATIONS):
                midpoint = 0.5 * (lower + upper)
                midpoint_mean = torch.sigmoid(beta * (filtered_logits + midpoint)).mean(
                    dim=(-2, -1), keepdim=True
                )
                lower = torch.where(midpoint_mean < _TARGET_FRACTION, midpoint, lower)
                upper = torch.where(midpoint_mean >= _TARGET_FRACTION, midpoint, upper)
            offsets = 0.5 * (lower + upper)

    design = _ImplicitBatchedVolumeProjection.apply(filtered_logits, offsets, beta)
    errors = (design.detach().mean(dim=(-2, -1)) - _TARGET_FRACTION).abs()
    if torch.any(errors > _MEAN_TOLERANCE):
        raise VolumeProjectionError(
            "volume bisection did not converge within the locked tolerance"
        )
    return design, errors


def _exact_cardinality_binary(designs: Tensor) -> Tensor:
    flat = designs.detach().flatten(start_dim=-2)
    ranking = torch.argsort(flat, dim=-1, descending=True, stable=True)
    binary = torch.zeros_like(flat)
    binary.scatter_(dim=-1, index=ranking[..., :_BINARY_COUNT], value=1.0)
    return binary.reshape_as(designs)


def project_candidate_maps(
    logits: Tensor,
    *,
    beta: float,
) -> tuple[Tensor, Tensor, tuple[float, ...]]:
    """Project all MT3 candidate maps independently in one tensor program."""
    if logits.ndim != 4 or tuple(logits.shape[1:]) != (4, 64, 64):
        raise ValueError("candidate logits must have shape [batch,4,64,64]")
    if logits.shape[0] == 0:
        raise ValueError("candidate logits batch must be non-empty")
    if logits.dtype is not torch.float32:
        raise ValueError("candidate logits must use float32")
    if not torch.isfinite(logits).all():
        raise ValueError("candidate logits must be finite")
    if not torch.isfinite(torch.tensor(beta)) or beta <= 0.0:
        raise ValueError("projection beta must be finite and positive")

    filtered = _filter_candidate_maps(logits)
    designs, error_tensor = _project_filtered_maps(filtered, beta=beta)
    binary = _exact_cardinality_binary(designs)
    if torch.any(binary.sum(dim=(-2, -1)) != _BINARY_COUNT):
        raise RuntimeError("MT3 candidate binary budget is invalid")
    errors = tuple(float(value) for value in error_tensor.reshape(-1).tolist())
    return designs, binary, errors
