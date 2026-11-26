"""Five fluid lanes with constructive two-cell copper ribs, unchanged physics.

The 56-column skeleton has at most 400+5*(56+55)=955 cells. Discrete
transitions reserve two copper cells between adjacent path intervals. Growth
starts inside each lane, so it cannot bridge a rib or create another body lane.
"""

import heapq
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

from .joint_design import LockedUNet
from .locked_channels import LockedGeometry

RIB_ROWS = (
    ("bottom", "bottom", "io", "top", "top"),
    ("bottom", "bottom", "bottom", "io", "top"),
    ("bottom", "bottom", "io", "top", "top"),
    ("bottom", "bottom", "bottom", "io", "top"),
)
RIB_WIDTHS = np.array(
    [
        [1.0, 1.0, 0.85, 1.0, 1.0],
        [0.85, 0.85, 0.85, 0.8, 0.9],
        [1.0, 1.0, 0.9, 0.85, 0.85],
        [0.95, 1.0, 0.95, 0.65, 0.8],
    ]
)
MOVES = np.array(list(product((-1, 0, 1), repeat=5)), dtype=np.int64)
MOVEMENT_COST = 1e-4 * np.abs(MOVES).sum(1)
N_PARAMETERS = 4125


@dataclass(frozen=True)
class RibbedGeometry(LockedGeometry):
    hot_column: int = 0

    @classmethod
    def from_task(cls, task):
        g = super().from_task(task)
        lower = sorted(
            (c.x, i) for i, c in enumerate(task.chiplets) if c.width == 8 and c.y < 32
        )
        powers = np.asarray(task.powers_w)[:, [i for _, i in lower]].max(0)
        return replace(g, hot_column=int(powers.argmax()))

    def layout(self, head):
        x = []
        a = []
        for b, kind in enumerate(RIB_ROWS[head]):
            if kind == "io":
                columns = np.floor((self.bottom[:, 0] + self.top[:, 0]) / 2)
                centre = np.full(4, 29.0 if head in (0, 3) else 30.5)
                if head == 2:
                    centre = np.full(4, 29.0)
                    centre[self.hot_column] = 25.5
            else:
                points = self.bottom if kind == "bottom" else self.top
                columns = np.floor(points[:, 0])
                centre = points[:, 1].copy()
                if head == 0:
                    centre += (-2.6, 2.6, 0, -2.5, 2.5)[b]
                elif head == 1:
                    centre += (-3.6, 0, 3.6, 0, 0)[b]
                elif head == 2:
                    centre += (-2.8, 2.4, 0, -2.5, 2.5)[b]
                elif b in (0, 2):
                    spread = np.full(4, 4.4)
                    spread[self.hot_column] = 3.3
                    centre += spread * (-1 if b == 0 else 1)
            x.append(np.r_[4, columns, 59])
            a.append(np.clip(centre, 8.5, 54.5))
        return np.array(x), np.array(a)


class MaskedBudget(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score, core, wall, beta):
        body = torch.zeros_like(core)
        body[:, 4:60, 7:57] = True
        free = body & ~wall & ~core
        remaining = 1024 - core.sum((-2, -1), keepdim=True)
        if (remaining < 0).any() or (
            remaining > free.sum((-2, -1), keepdim=True)
        ).any():
            raise ValueError("infeasible separated-channel volume")
        lo = -score.amax((-2, -1), keepdim=True) - 40 / beta
        hi = -score.amin((-2, -1), keepdim=True) + 40 / beta
        for _ in range(80):
            mid = (lo + hi) / 2
            mass = (torch.sigmoid(beta * (score + mid)) * free).sum(
                (-2, -1), keepdim=True
            )
            lo = torch.where(mass < remaining, mid, lo)
            hi = torch.where(mass >= remaining, mid, hi)
        d = torch.where(core, 1.0, torch.sigmoid(beta * (score + (lo + hi) / 2)) * free)
        ctx.save_for_backward(d, free)
        ctx.beta = beta
        return d

    @staticmethod
    def backward(ctx, grad):
        d, free = ctx.saved_tensors
        w = ctx.beta * d * (1 - d) * free
        mean = (grad * w).sum((-2, -1), keepdim=True) / w.sum(
            (-2, -1), keepdim=True
        ).clamp_min(1e-30)
        return w * (grad - mean), None, None, None


def separated_fill(score, core, wall):
    d = np.array(core, copy=True)
    seen = d.copy()
    heap = []

    def offer(x, y):
        if 4 <= x < 60 and 7 <= y < 57 and not seen[x, y] and not wall[x, y]:
            seen[x, y] = True
            heapq.heappush(heap, (-float(score[x, y]), x, y))

    for x, y in np.argwhere(d[4:60]):
        x += 4
        offer(x - 1, y)
        offer(x + 1, y)
        offer(x, y - 1)
        offer(x, y + 1)
    for _ in range(1024 - int(d.sum())):
        if not heap:
            raise RuntimeError("not enough connected lane area")
        _, x, y = heapq.heappop(heap)
        d[x, y] = True
        offer(x - 1, y)
        offer(x + 1, y)
        offer(x, y - 1)
        offer(x, y + 1)
    return d


class RibbedDecoder:
    def __init__(self, geometries):
        self.geometries = (
            (geometries,)
            if isinstance(geometries, RibbedGeometry)
            else tuple(geometries)
        )

    @staticmethod
    def initial(head):
        p = np.zeros(N_PARAMETERS, dtype=np.float64)
        ratio = (RIB_WIDTHS[head] - 0.6) / 1.6
        p[20:25] = np.log(ratio / (1 - ratio))
        return p

    @staticmethod
    def to_physical(p):
        return torch.cat(
            (
                5.5 * p[:, :20].tanh(),
                0.6 + 1.6 * p[:, 20:25].sigmoid(),
                2 * p[:, 25:29].tanh(),
                p[:, 29:].tanh(),
            ),
            1,
        )

    @staticmethod
    def clamp_physical(q):
        with torch.no_grad():
            q[:20].clamp_(-5.5, 5.5)
            q[20:25].clamp_(0.6, 2.2)
            q[25:29].clamp_(-2, 2)
            q[29:].clamp_(-1, 1)

    def knots(self, p, heads, *, physical=False):
        q = p if physical else self.to_physical(p)
        gs = self.geometries * len(p) if len(self.geometries) == 1 else self.geometries
        if len(gs) != len(p):
            raise ValueError("geometry batch mismatch")
        layouts = [g.layout(int(h)) for g, h in zip(gs, heads, strict=True)]
        x = q.new_tensor(np.stack([v[0] for v in layouts]))
        a = q.new_tensor(np.stack([v[1] for v in layouts]))
        y = (a + q[:, :20].reshape(-1, 5, 4)).clamp(8.5, 54.5)
        # Central branch gets four independent horizontal offsets as requested.
        mask = q.new_tensor(
            [[float(k == "io") for k in RIB_ROWS[int(h)]] for h in heads]
        )
        inner = x[:, :, 1:-1] + mask[:, :, None] * q[:, None, 25:29]
        x = torch.cat((x[:, :, :1], inner, x[:, :, -1:]), 2)
        y = torch.cat((y[:, :, :1], y, y[:, :, -1:]), 2)
        return torch.stack((x, y), -1)

    def coordinates(self, p, heads, *, physical=False):
        knots = self.knots(p, heads, physical=physical)
        x, y = knots[..., 0], knots[..., 1]
        columns = torch.arange(64, device=p.device, dtype=p.dtype)[None, None]
        segment = (columns[:, :, None] >= x[:, :, 1:-1, None]).sum(2)
        x0 = x.gather(2, segment)
        x1 = x.gather(2, segment + 1)
        y0 = y.gather(2, segment)
        y1 = y.gather(2, segment + 1)
        length = x1 - x0
        # A movable IO knot can coincide with a collector endpoint. The end
        # heights are identical, so that zero-length segment is constant. Do
        # not evaluate 0/0; every positive-length segment stays bit-identical.
        denominator = torch.where(length > 0, length, torch.ones_like(length))
        fraction = torch.where(length > 0, (columns - x0) / denominator, 0.0).clamp(
            0, 1
        )
        desired = y0 + fraction * (y1 - y0)
        index = torch.arange(5, device=p.device, dtype=p.dtype)[None, :, None]
        z = desired - 3 * index
        major = torch.cummax(z, 1).values
        minor = torch.cummin(z.flip(1), 1).values.flip(1)
        ordered = ((major + minor) / 2).clamp(8.5, 42.5) + 3 * index
        return torch.stack((columns.expand_as(ordered), ordered), -1)

    @staticmethod
    def masks(coordinates):
        target = coordinates.detach().cpu().numpy()[:, :, 4:60, 1]
        n = len(target)
        rows = np.zeros((n, 5, 56), dtype=np.int64)
        base = np.rint(target[:, :, 0]).astype(np.int64)
        base = np.maximum.accumulate(base - 3 * np.arange(5), axis=1).clip(
            8, 43
        ) + 3 * np.arange(5)
        rows[:, :, 0] = base
        for column in range(1, 56):
            previous = rows[:, :, column - 1]
            candidates = previous[:, None] + MOVES[None]
            lower = np.minimum(previous[:, None], candidates)
            upper = np.maximum(previous[:, None], candidates)
            valid = ((candidates >= 8) & (candidates <= 55)).all(2)
            valid &= (lower[:, :, 1:] - upper[:, :, :-1] >= 3).all(2)
            cost = ((candidates - target[:, None, :, column]) ** 2).sum(
                2
            ) + MOVEMENT_COST
            choice = np.where(valid, cost, np.inf).argmin(1)
            rows[:, :, column] = candidates[np.arange(n), choice]
        previous = np.concatenate((rows[:, :, :1], rows[:, :, :-1]), 2)
        low = np.minimum(rows, previous)
        high = np.maximum(rows, previous)
        ys = np.arange(64)[None, None, None]
        paths = ((ys >= low[:, :, :, None]) & (ys <= high[:, :, :, None])).any(1)
        core = np.zeros((n, 64, 64), bool)
        core[:, 4:60] = paths
        core[:, :4, 7:57] = True
        core[:, 60:, 7:57] = True
        wall = np.zeros_like(core)
        gap_low = high[:, :-1] + 1
        gap_high = low[:, 1:] - 1
        if (gap_high - gap_low < 1).any():
            raise RuntimeError("two-cell rib construction failed")
        first = (gap_low + gap_high - 1) // 2
        for b in range(4):
            wall[np.arange(n)[:, None], np.arange(4, 60)[None], first[:, b]] = True
            wall[np.arange(n)[:, None], np.arange(4, 60)[None], first[:, b] + 1] = True
        if (core & wall).any() or (core.sum((1, 2)) > 955).any():
            raise RuntimeError("invalid separated skeleton")
        return (
            torch.tensor(core, device=coordinates.device),
            torch.tensor(wall, device=coordinates.device),
            rows,
        )

    def fields(self, p, heads, *, physical=False):
        q = p if physical else self.to_physical(p)
        xy = self.coordinates(q, heads, physical=True)
        centres = xy[..., 1]
        slope = F.pad(
            (centres[:, :, 2:] - centres[:, :, :-2]) / 2, (1, 1), mode="replicate"
        )
        yy = torch.arange(64, device=q.device, dtype=q.dtype)[None, None, None]
        distance = ((yy - centres[:, :, :, None]).square() + 1e-12).sqrt() / (
            1 + slope[:, :, :, None].square()
        ).sqrt()
        widths = q[:, 20:25, None, None].expand_as(distance)
        gs = self.geometries * len(p) if len(self.geometries) == 1 else self.geometries
        bonus = []
        for g, h in zip(gs, heads, strict=True):
            profile = np.zeros((5, 64))
            if int(h) == 3:
                peak = g.bottom[g.hot_column, 0]
                profile[:3] = 0.35 * np.exp(-0.5 * ((np.arange(64) - peak) / 8) ** 2)
            bonus.append(profile)
        score = (
            torch.logsumexp(
                10 * (widths + q.new_tensor(np.stack(bonus))[:, :, :, None] - distance),
                1,
            )
            / 10
        )
        pixels = q[:, 29:].reshape(-1, 1, 64, 64)
        kernel = q.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])[None, None] / 16
        pixel = F.conv2d(F.pad(pixels, (1, 1, 1, 1), mode="reflect"), kernel)[:, 0]
        core, wall, _ = self.masks(xy)
        return score + 0.35 * pixel, core, wall

    def binary(self, p, heads, *, physical=False):
        with torch.no_grad():
            score, core, wall = self.fields(p, heads, physical=physical)
            d = np.stack(
                [
                    separated_fill(s, c, w)
                    for s, c, w in zip(
                        score.cpu().numpy(),
                        core.cpu().numpy(),
                        wall.cpu().numpy(),
                        strict=True,
                    )
                ]
            )
            return torch.tensor(d, device=p.device)

    def density(self, p, heads, *, beta, hard, physical=False):
        score, core, wall = self.fields(p, heads, physical=physical)
        if not hard:
            return MaskedBudget.apply(score, core, wall, beta)
        header = torch.zeros_like(core)
        header[:, :4, 7:57] = True
        header[:, 60:, 7:57] = True
        smooth = MaskedBudget.apply(score, header, wall, beta)
        with torch.no_grad():
            d = np.stack(
                [
                    separated_fill(s, c, w)
                    for s, c, w in zip(
                        score.cpu().numpy(),
                        core.cpu().numpy(),
                        wall.cpu().numpy(),
                        strict=True,
                    )
                ]
            )
            real = torch.tensor(d, device=p.device, dtype=p.dtype)
        return real + (smooth - smooth.detach())


class RibbedUNet(LockedUNet):
    def __init__(self):
        super().__init__()
        self.geometry = torch.nn.Linear(128 * 16 + 6 * 32, 4 * 29)
        torch.nn.init.zeros_(self.geometry.weight)
        torch.nn.init.zeros_(self.geometry.bias)
        ratio = (RIB_WIDTHS - 0.6) / 1.6
        self.width_prior = torch.tensor(
            np.log(ratio / (1 - ratio)), dtype=torch.float32
        )

    def forward(self, condition):
        a = self.enc0(condition)
        b = self.enc1(a)
        c = self.enc2(b)
        d = self.enc3(c)
        local = F.adaptive_avg_pool2d(condition[:, :6], (4, 8)).flatten(1)
        features = torch.cat((F.adaptive_avg_pool2d(d, (4, 4)).flatten(1), local), 1)
        raw = self.geometry(features).reshape(-1, 4, 29)
        geometry = torch.cat(
            (
                raw[:, :, :20],
                raw[:, :, 20:25] + self.width_prior[None],
                raw[:, :, 25:29],
            ),
            2,
        ).clamp(-6, 6)
        up = self.up0(self.up1(self.up2(d, c), b), a)
        return torch.cat((geometry, 3 * self.pixels(up).tanh().flatten(2)), 2)
