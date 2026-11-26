"""Same binary channel decoder; only its priority queue is accelerated."""

import numpy as np
import torch

from .ribbed_channels import MOVEMENT_COST, MOVES, RibbedDecoder, separated_fill


class FastRibbedDecoder(RibbedDecoder):
    @staticmethod
    def masks(coordinates):
        from ._channel_kernel import skeleton_kernel

        target = np.ascontiguousarray(coordinates.detach().cpu().numpy()[:, :, 4:60, 1])
        core, wall, rows = skeleton_kernel(target, MOVES, MOVEMENT_COST)
        if (core & wall).any() or (core.sum((1, 2)) > 955).any():
            raise RuntimeError("invalid separated skeleton")
        return (
            torch.from_numpy(core).to(coordinates.device),
            torch.from_numpy(wall).to(coordinates.device),
            rows,
        )


def compiled_fill(score, core, wall):
    s = np.ascontiguousarray(score, dtype=np.float64)
    c = np.ascontiguousarray(core, dtype=bool)
    w = np.ascontiguousarray(wall, dtype=bool)
    if s.shape != (64, 64) or c.shape != s.shape or w.shape != s.shape:
        raise ValueError("three 64x64 fields required")
    if not np.isfinite(s).all() or c.sum() > 1024 or (c & w).any():
        raise ValueError("finite scores and feasible disjoint core/walls required")
    try:
        from ._channel_kernel import fill_kernel
    except ImportError:
        return separated_fill(s, c, w)
    return fill_kernel(s, c, w)


def decode_one(geometry, physical, head):
    decoder = FastRibbedDecoder(geometry)
    with torch.inference_mode():
        score, core, wall = decoder.fields(
            torch.as_tensor(physical, dtype=torch.float64)[None],
            [int(head)],
            physical=True,
        )
    return compiled_fill(score[0].numpy(), core[0].numpy(), wall[0].numpy())


def decode_many(geometry, physical, heads):
    decoder = FastRibbedDecoder(geometry)
    with torch.inference_mode():
        score, core, wall = decoder.fields(
            torch.as_tensor(physical, dtype=torch.float64), heads, physical=True
        )
    return np.stack(
        [
            compiled_fill(s.numpy(), c.numpy(), w.numpy())
            for s, c, w in zip(score, core, wall, strict=True)
        ]
    )


def flow_geometry_features(designs):
    from ._channel_kernel import hydraulic_signature

    d = np.ascontiguousarray(designs, dtype=bool)
    if d.ndim != 3 or d.shape[1:] != (64, 64):
        raise ValueError("binary candidate maps with shape [count,64,64] required")
    return hydraulic_signature(d)
