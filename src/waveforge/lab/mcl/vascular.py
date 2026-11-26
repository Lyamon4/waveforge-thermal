"""Exact Stage 1.1 candidate decoder, NOT a certified connected projection.

Vectorized Euclidean distances, smooth core priority and deterministic Top1024.
Connectivity is measured explicitly; no morphology silently repairs outputs.
"""

import numpy as np
import torch
from scipy.ndimage import label


def segment_distances(y_knots):
    """[batch,branch,6] -> minimum distances [batch,branch,64,64], in cells."""
    if (
        y_knots.ndim != 3
        or y_knots.shape[-1] != 6
        or not y_knots.is_floating_point()
        or not torch.isfinite(y_knots).all()
    ):
        raise ValueError("finite floating point y knots [batch,branch,6] required")
    opts = dict(dtype=y_knots.dtype, device=y_knots.device)
    x = torch.tensor([0, 12, 25, 38, 51, 63], **opts)
    grid = torch.arange(64, **opts)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    dx = (x[1:] - x[:-1])[None, None, :, None, None]
    dy = (y_knots[..., 1:] - y_knots[..., :-1])[..., None, None]
    rx = xx[None, None, None] - x[:-1][None, None, :, None, None]
    ry = yy[None, None, None] - y_knots[..., :-1, None, None]
    # dx is strictly positive, so there is no zero-length segment or epsilon bias.
    t = ((rx * dx + ry * dy) / (dx.square() + dy.square())).clamp(0, 1)
    delta = torch.stack(torch.broadcast_tensors(rx - t * dx, ry - t * dy), dim=-1)
    return torch.linalg.vector_norm(delta, dim=-1).amin(2)


def vascular_scores(waypoints, widths, pixel=None):
    """Physical bounded parameters, not raw sigmoid coordinates; tau=10."""
    if (
        waypoints.ndim != 3
        or waypoints.shape[1] not in (4, 6)
        or waypoints.shape[-1] != 4
        or widths.shape != waypoints.shape[:2]
        or widths.dtype != waypoints.dtype
        or widths.device != waypoints.device
        or not torch.isfinite(widths).all()
        or not torch.isfinite(waypoints).all()
        or bool(((waypoints < 4) | (waypoints > 59)).any())
        or bool(((widths < 1) | (widths > 4)).any())
    ):
        raise ValueError(
            "bounded waypoints [batch,4|6,4] and widths [batch,branch] required"
        )
    ends = torch.full_like(waypoints[..., :1], 31.5)
    distances = segment_distances(torch.cat((ends, waypoints, ends), -1))
    normalized = 0.8 + 1.5 * widths / widths.mean(-1, keepdim=True)
    priority = 10 * torch.sigmoid(10 * (0.8 - distances))
    branch_scores = priority + normalized[..., None, None] - distances
    scores = torch.logsumexp(10 * branch_scores, dim=1) / 10
    if pixel is not None:
        if (
            pixel.shape != scores.shape
            or pixel.dtype != scores.dtype
            or pixel.device != scores.device
            or not torch.isfinite(pixel).all()
        ):
            raise ValueError("colocated finite pixel field [batch,64,64] required")
        scores = scores + 0.35 * pixel.tanh()
    return scores, distances


def binary_topk(scores):
    """Exact 1024 cells, descending score with stable row-major tie breaking."""
    if (
        scores.ndim != 3
        or scores.shape[-2:] != (64, 64)
        or not torch.isfinite(scores).all()
    ):
        raise ValueError("finite scores [batch,64,64] required")
    flat = scores.flatten(1)
    indices = torch.argsort(flat, dim=1, descending=True, stable=True)[:, :1024]
    binary = torch.zeros_like(flat, dtype=torch.bool)
    return binary.scatter_(1, indices, True).reshape_as(scores)


def connectivity(binary, distances=None):
    binary = np.asarray(binary, dtype=bool)
    if binary.shape != (64, 64):
        raise ValueError("64x64 binary design required")
    components, count = label(binary)  # scipy default cross = four neighbours
    spanning = (set(components[0, 24:40]) & set(components[-1, 24:40])) - {0}
    connected = bool(spanning)
    result = {
        "cells": int(binary.sum()),
        "components": int(count),
        "spanning_components": len(spanning),
        "inlet_outlet_connected": connected,
        "all_cells_port_connected": connected and count == 1,
    }
    if distances is not None:
        core = np.min(distances, axis=0) <= 0.8
        result.update(
            core_cells=int(core.sum()),
            omitted_core_cells=int((core & ~binary).sum()),
            core_exceeds_budget=bool(core.sum() > 1024),
        )
    return result
