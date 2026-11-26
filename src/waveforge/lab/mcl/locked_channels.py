"""Chiplet-centred channel families, exact crossings and canonical Y reflection."""

import heapq
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import find_objects, label

from .adaptive_manifold import reflected_task
from .stage3 import _CoreProjection
from .stage3_training import task_from_dict

WIDTH_PRIORS = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.4, 1.4, 1.4, 0.8, 1.0],
        [1.6, 1.6, 1.6, 0.7, 0.9],
        [1.6, 1.6, 1.6, 0.61, 0.85],
    ]
)
ROW_KINDS = (
    ("bottom", "bottom", "io", "top", "top"),
    ("bottom", "bottom", "bottom", "io", "top"),
    ("bottom", "bottom", "bottom", "io", "top"),
    ("bottom", "bottom", "bottom", "io", "top"),
)


@dataclass(frozen=True)
class CanonicalFrame:
    task: object
    reflected: bool
    self_symmetric: bool

    def to_original(self, array, *, y_velocity=False):
        if not self.reflected:
            return array.copy() if isinstance(array, np.ndarray) else array.clone()
        result = (
            np.flip(array, -1).copy()
            if isinstance(array, np.ndarray)
            else array.flip(-1)
        )
        return -result if y_velocity else result


def canonical_frame(task):
    original = task_from_dict(task.canonical())
    reflected = task_from_dict(reflected_task(task).canonical())
    bottom = [i for i, c in enumerate(task.chiplets) if c.width == 8 and c.y < 32]
    top = [i for i, c in enumerate(task.chiplets) if c.width == 8 and c.y >= 32]
    p = np.asarray(task.powers_w)
    lower = p[:, bottom]
    upper = p[:, top]
    differences = [
        lower.sum() - upper.sum(),
        lower.sum(1).max() - upper.sum(1).max(),
        lower.max() - upper.max(),
    ]
    a = json.dumps(original.canonical(), sort_keys=True, separators=(",", ":"))
    b = json.dumps(reflected.canonical(), sort_keys=True, separators=(",", ":"))
    flip = None
    for value in differences:
        if abs(value) > 1e-9:
            flip = bool(value < 0)
            break
    if flip is None:
        flip = b < a
    return CanonicalFrame(reflected if flip else original, flip, a == b)


@dataclass(frozen=True)
class LockedGeometry:
    bottom: np.ndarray
    top: np.ndarray
    io_y: float = 31.5

    @classmethod
    def from_mask(cls, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (64, 64):
            raise ValueError("64x64 chip mask required")
        regions, count = label(mask)
        rows = []
        io = []
        for slices in find_objects(regions):
            x, y = slices
            width = x.stop - x.start
            height = y.stop - y.start
            if not mask[slices].all():
                raise ValueError("rectangular source regions required")
            centre = ((x.start + x.stop - 1) / 2, (y.start + y.stop - 1) / 2)
            if (width, height) == (8, 7):
                rows.append(centre)
            elif (width, height) == (12, 12):
                io.append(centre)
            else:
                raise ValueError("unexpected source geometry")
        lower = np.array(sorted(c for c in rows if c[1] < 32), dtype=float)
        upper = np.array(sorted(c for c in rows if c[1] > 32), dtype=float)
        if (
            count != 9
            or lower.shape != (4, 2)
            or upper.shape != (4, 2)
            or io != [(31.5, 31.5)]
        ):
            raise ValueError("eight separated CCDs and central I/O required")
        return cls(lower, upper)

    @classmethod
    def from_task(cls, task):
        mask = np.zeros((64, 64), bool)
        for c in task.chiplets:
            mask[c.x : c.x + c.width, c.y : c.y + c.height] = True
        return cls.from_mask(mask)

    def layout(self, head):
        if head not in range(4):
            raise ValueError("head must be0..3")
        if head == 0:
            offsets = [-1.6, 1.6, 0, -1.6, 1.6]
        elif head == 1:
            offsets = [-2.2, 0, 2.2, 0, 0]
        elif head == 2:
            offsets = [-1.8, -0.3, 1.5, 0, 0]
        else:
            offsets = [-2.4, 0.2, 2.4, 0, 0]
        x = []
        a = []
        lo = []
        hi = []
        for kind, offset in zip(ROW_KINDS[head], offsets, strict=True):
            if kind == "io":
                centre = np.full(4, self.io_y)
                columns = np.floor((self.bottom[:, 0] + self.top[:, 0]) / 2)
            else:
                points = self.bottom if kind == "bottom" else self.top
                centre = points[:, 1]
                columns = np.floor(points[:, 0])
            if not (
                np.all(np.diff(columns) > 0) and columns[0] > 4 and columns[-1] < 59
            ):
                raise ValueError("source column centres outside inner manifold")
            x.append(np.r_[4, columns, 59])
            lo.append(np.maximum(centre - 2.5, 8.5))
            hi.append(np.minimum(centre + 2.5, 54.5))
            a.append(np.clip(centre + offset, lo[-1], hi[-1]))
        return np.array(x), np.array(a), np.array(lo), np.array(hi)


def connected_budget(score, core):
    s = np.asarray(score, dtype=float)
    d = np.asarray(core, dtype=bool).copy()
    if (
        s.shape != (64, 64)
        or d.shape != s.shape
        or not np.isfinite(s).all()
        or not 1 <= d.sum() <= 832
    ):
        raise ValueError("finite scores and <=832-cell connected core required")
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


class LockedDecoder:
    row_kinds = ROW_KINDS

    def __init__(self, geometry):
        self.geometries = (
            (geometry,) if isinstance(geometry, LockedGeometry) else tuple(geometry)
        )

    @staticmethod
    def initial(head):
        p = np.zeros(4121, dtype=np.float64)
        ratio = (WIDTH_PRIORS[head] - 0.6) / 1.6
        p[20:25] = np.log(ratio / (1 - ratio))
        return p

    def coordinates(self, parameters, heads):
        if (
            parameters.ndim != 2
            or parameters.shape[1] != 4121
            or len(heads) != len(parameters)
        ):
            raise ValueError("matching4121-vectors and head indices required")
        if not torch.isfinite(parameters).all():
            raise ValueError("nonfinite parameters")
        geometries = (
            self.geometries * len(parameters)
            if len(self.geometries) == 1
            else self.geometries
        )
        if len(geometries) != len(parameters):
            raise ValueError("geometry batch mismatch")
        layouts = [g.layout(int(h)) for g, h in zip(geometries, heads, strict=True)]
        x, a, lo, hi = [
            parameters.new_tensor(np.stack([v[k] for v in layouts])) for k in range(4)
        ]
        y = (
            (a + 2 * parameters[:, :20].reshape(-1, 5, 4).tanh())
            .maximum(lo)
            .minimum(hi)
        )
        y = torch.cat((y[:, :, :1], y, y[:, :, -1:]), 2)
        return torch.stack((x, y), -1)

    @staticmethod
    def core(coordinates):
        with torch.no_grad():
            x, y = coordinates[..., 0], coordinates[..., 1]
            columns = torch.arange(4, 60, device=x.device, dtype=x.dtype)[None, None]
            segment = (columns[:, :, None] >= x[:, :, 1:-1, None]).sum(2)
            x0 = x.gather(2, segment)
            x1 = x.gather(2, segment + 1)
            y0 = y.gather(2, segment)
            y1 = y.gather(2, segment + 1)
            rows = (y0 + (columns - x0) / (x1 - x0) * (y1 - y0)).round()
            previous = torch.cat((rows[:, :, :1], rows[:, :, :-1]), 2)
            ys = torch.arange(64, device=x.device)[None, None, None]
            path = (
                (ys >= torch.minimum(rows, previous)[..., None])
                & (ys <= torch.maximum(rows, previous)[..., None])
            ).any(1)
            core = torch.zeros(len(x), 64, 64, dtype=torch.bool, device=x.device)
            core[:, 4:60] = path
            core[:, :4, 7:57] = True
            core[:, 60:, 7:57] = True
            if (core.sum((-2, -1)) > 832).any():
                raise RuntimeError("locked core exceeds832cells")
            return core

    def fields(self, parameters, heads):
        xy = self.coordinates(parameters, heads)
        X, Y = torch.meshgrid(
            torch.arange(64, device=parameters.device, dtype=parameters.dtype),
            torch.arange(64, device=parameters.device, dtype=parameters.dtype),
            indexing="ij",
        )
        a = xy[:, :, :-1]
        v = xy[:, :, 1:] - a
        rx = X[None, None, None] - a[..., 0, None, None]
        ry = Y[None, None, None] - a[..., 1, None, None]
        vx = v[..., 0, None, None]
        vy = v[..., 1, None, None]
        t = ((rx * vx + ry * vy) / (vx.square() + vy.square())).clamp(0, 1)
        distance = (
            ((rx - t * vx).square() + (ry - t * vy).square() + 1e-12).sqrt().amin(2)
        )
        widths = 0.6 + 1.6 * parameters[:, 20:25].sigmoid()
        score = torch.logsumexp(10 * (widths[..., None, None] - distance), 1) / 10
        pixels = parameters[:, 25:].reshape(-1, 1, 64, 64)
        kernel = (
            parameters.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])[None, None] / 16
        )
        pixels = F.conv2d(F.pad(pixels, (1, 1, 1, 1), mode="reflect"), kernel)[:, 0]
        return score + 0.35 * pixels.tanh(), self.core(xy)

    def binary(self, parameters, heads):
        with torch.no_grad():
            score, core = self.fields(parameters, heads)
            maps = np.stack(
                [
                    connected_budget(s, c)
                    for s, c in zip(
                        score.cpu().numpy(), core.cpu().numpy(), strict=True
                    )
                ]
            )
            return torch.as_tensor(maps, device=parameters.device)

    def density(self, parameters, heads, *, beta, hard):
        score, core = self.fields(parameters, heads)
        if not hard:
            return _CoreProjection.apply(score, core, beta)
        smooth = _CoreProjection.apply(score, torch.zeros_like(core), beta)
        with torch.no_grad():
            maps = np.stack(
                [
                    connected_budget(s, c)
                    for s, c in zip(
                        score.cpu().numpy(), core.cpu().numpy(), strict=True
                    )
                ]
            )
            real = torch.as_tensor(
                maps, dtype=parameters.dtype, device=parameters.device
            )
        return real + (smooth - smooth.detach())
