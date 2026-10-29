"""Bounded trajectories and explicit core reservation, without hidden repairs."""

import torch

from .vascular import binary_topk, connectivity, segment_distances


def limited_waypoints(raw):
    if (
        raw.ndim != 3
        or raw.shape[-1] != 4
        or raw.shape[1] not in (4, 6)
        or not raw.is_floating_point()
        or not torch.isfinite(raw).all()
    ):
        raise ValueError("finite [batch,4|6,4] raw increments required")
    previous = torch.full_like(raw[..., 0], 31.5)
    nodes = []
    for k in range(4):
        # There are 4-k remaining segments after this interior knot.
        low = max(6.0, 31.5 - 16 * (4 - k))
        high = min(57.0, 31.5 + 16 * (4 - k))
        previous = (previous + 16 * raw[..., k].tanh()).clamp(low, high)
        nodes.append(previous)
    return torch.stack(nodes, dim=-1)


def smooth_scores(waypoints, widths, pixel=None):
    if (
        waypoints.ndim != 3
        or waypoints.shape[-1] != 4
        or waypoints.shape[1] not in (4, 6)
        or widths.shape != waypoints.shape[:2]
        or not waypoints.is_floating_point()
        or widths.dtype != waypoints.dtype
        or widths.device != waypoints.device
        or not torch.isfinite(waypoints).all()
        or not torch.isfinite(widths).all()
        or bool(((waypoints < 4) | (waypoints > 59)).any())
        or bool(((widths < 1) | (widths > 4)).any())
    ):
        raise ValueError("bounded trajectory/width tensors required")
    ends = torch.full_like(waypoints[..., :1], 31.5)
    distance = segment_distances(torch.cat([ends, waypoints, ends], -1))
    width = 0.75 + 1.5 * widths / widths.mean(-1, keepdim=True)
    score = (
        10 * torch.sigmoid(10 * (0.75 - distance)) + width[..., None, None] - distance
    )
    score = torch.logsumexp(10 * score, dim=1) / 10
    if pixel is not None:
        if (
            pixel.shape != score.shape
            or pixel.dtype != score.dtype
            or pixel.device != score.device
            or not torch.isfinite(pixel).all()
        ):
            raise ValueError("finite pixel field [batch,64,64] required")
        score = score + 0.35 * pixel.tanh()
    return score, distance


def reserve_core(score, core):
    if (
        score.ndim != 3
        or score.shape[-2:] != (64, 64)
        or core.shape != score.shape
        or core.dtype != torch.bool
        or core.device != score.device
        or not torch.isfinite(score).all()
    ):
        raise ValueError("finite score and boolean core on [batch,64,64] required")
    if bool((core.sum((1, 2)) > 1024).any()):
        raise ValueError("Core exceeds 1024 budget: invalid branch parameters")
    # Derivative of this boolean offset is zero; TopK is still nondifferentiable.
    # A later optimization surrogate must explicitly define its own STE.
    spread = score.flatten(1).amax(1) - score.flatten(1).amin(1)
    if bool((spread >= 1000).any()):
        raise ValueError("score spread exceeds hard-reservation separation")
    ranking = score + 1000 * core.to(score.dtype)
    binary = binary_topk(ranking)
    if not bool(binary[core].all()):
        raise RuntimeError("core reservation invariant violated")
    return binary, ranking


def decode(raw, widths, pixel=None):
    waypoints = limited_waypoints(raw)
    score, distance = smooth_scores(waypoints, widths, pixel)
    core = distance.amin(1) <= 0.75
    binary, ranking = reserve_core(score, core)
    if any(
        not connectivity(d)["all_cells_port_connected"]
        for d in binary.detach().cpu().numpy()
    ):
        raise ValueError(
            "disconnected selected cells: candidate rejected without repair"
        )
    return binary, ranking, score, distance, waypoints
