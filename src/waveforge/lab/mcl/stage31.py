"""Stage3.1 surrogate derivatives; qualified forward physics/decoder unchanged."""

import math
from dataclasses import dataclass

import torch

from waveforge.design.parameterization import filter_logits

from .stage3 import _CoreProjection, halo_residual
from .vascular_stage12 import limited_waypoints, reserve_core, smooth_scores


def score_fields(parameters):
    """Compute geometry once for both frozen forward and priority-free surrogate."""
    if parameters.ndim != 2 or parameters.shape[1] != 4116:
        raise ValueError("parameters must be [batch,4116]")
    waypoints = limited_waypoints(parameters[:, :16].reshape(-1, 4, 4))
    widths = 1 + 3 * parameters[:, 16:20].sigmoid()
    forward, distances = smooth_scores(waypoints, widths)
    nearest = distances.amin(1)
    pixels = torch.stack([filter_logits(p.reshape(64, 64)) for p in parameters[:, 20:]])
    residual = 0.35 * halo_residual(pixels, nearest).tanh()
    normalized = 0.75 + 1.5 * widths / widths.mean(-1, keepdim=True)
    auxiliary = (
        torch.logsumexp(10 * (normalized[..., None, None] - distances), dim=1) / 10
    )
    return forward + residual, nearest <= 0.75, auxiliary + residual


def project_all(score, beta):
    # Empty fixed core = implicit exact-volume derivative on all4096 cells.
    return _CoreProjection.apply(score, torch.zeros_like(score, dtype=torch.bool), beta)


def density(parameters, *, beta, hard):
    forward_score, core, auxiliary_score = score_fields(parameters)
    smooth = project_all(auxiliary_score, beta)
    with torch.no_grad():
        forward = (
            reserve_core(forward_score, core)[0].to(parameters.dtype)
            if hard
            else _CoreProjection.apply(forward_score, core, beta)
        )
    # Exact forward values. This biased STE is NOT the hard-map derivative.
    used = forward + (smooth - smooth.detach())
    return used, smooth


def desaturate(parameters):
    result = parameters.detach().clone()
    cap = math.atanh(0.85)
    result[..., :16].clamp_(-cap, cap)
    result[..., 20:].clamp_(-cap, cap)
    return result


def logit_penalty(parameters):
    """L2 of raw inputs to decoder branch and residual tanh, not binary maps."""
    return parameters[..., :16].square().mean() + parameters[..., 20:].square().mean()


@dataclass(frozen=True)
class Phase:
    beta: float
    alpha: float
    branch_lr: float
    pixel_lr: float
    hard: bool
    level: int


def refinement_phase(index, budget):
    if budget not in (25, 600) or not 0 <= index < budget:
        raise ValueError("index outside refinement/control budget")
    first, second = (9, 17) if budget == 25 else (200, 400)
    level = 0 if index < first else 1 if index < second else 2
    return Phase(
        (2.0, 4.0, 8.0)[level],
        (150.0, 300.0, 500.0)[level],
        (0.08, 0.05, 0.02)[level],
        (0.04, 0.03, 0.015)[level],
        level == 2,
        level,
    )


def training_level(index, steps=800):
    if steps < 3 or not 0 <= index < steps:
        raise ValueError("training index/step count invalid")
    return 0 if index < round(steps / 3) else 1 if index < round(2 * steps / 3) else 2


def learning_rate(index, steps=800):
    if steps < 2 or not 0 <= index < steps:
        raise ValueError("learning-rate index/step count invalid")
    return 0.0001 + 0.0002 * (1 + math.cos(math.pi * index / (steps - 1)))
