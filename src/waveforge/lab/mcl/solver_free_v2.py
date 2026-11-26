"""Solver-free inference v2: same frozen generator and decoder, faster features,
the new LightGBM selector (selector_new.joblib). No physics is called.

v1 (solver_free_tabular.TabularSession) is unchanged and still available.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, uniform_filter

from .baselines import uniform_parallel
from .fast_channels import decode_many, flow_geometry_features
from .locked_channels import canonical_frame
from .ribbed_channels import RibbedGeometry
from .solver_free import create_session, load_solver_free, raw_features
from .solver_free_tabular import FEATURE_VERSION, choose_tabular

REPO = Path(__file__).resolve().parents[4]
CHECKPOINT = REPO / "models/solver_free_20261030/model_frozen.pt"
SELECTOR = REPO / "models/solver_free_20261030/selector_v2.joblib"


def fast_features(task, designs):
    """Vectorised, bit-identical twin of candidate_features (519-feature version)."""
    frame = canonical_frame(task)
    task = frame.task
    d = frame.to_original(np.asarray(designs, dtype=bool))
    n = len(d)
    chips = task.chiplets
    indices = sorted(
        (i for i, c in enumerate(chips) if c.width == 8),
        key=lambda i: (chips[i].y >= 32, chips[i].x),
    )
    indices += [i for i, c in enumerate(chips) if c.width == 12]
    if len(indices) != 9:
        raise ValueError("eight compute regions and one I/O required")
    power = np.asarray(task.powers_w)[:, indices] / 100
    power = power[sorted(range(3), key=lambda k: tuple(power[k]))]
    centres = np.array(
        [
            (
                (chips[i].x + (chips[i].width - 1) / 2) / 63,
                (chips[i].y + (chips[i].height - 1) / 2) / 63,
            )
            for i in indices
        ]
    )
    upstream = np.zeros((3, 9))
    for i in range(9):
        w = (centres[:, 0] < centres[i, 0]) * np.exp(
            -0.5 * ((centres[:, 1] - centres[i, 1]) / 0.2) ** 2
        )
        upstream[:, i] = power @ w

    mirror = d[:, :, ::-1]
    take = np.array([a.tobytes() < b.tobytes() for a, b in zip(d, mirror, strict=True)])
    flow_d = np.where(take[:, None, None], d, mirror)
    flow = np.concatenate(
        (
            flow_geometry_features(flow_d),
            flow_d.reshape(n, 8, 8, 8, 8).mean((2, 4)).reshape(n, 64),
            flow_d[:, 4:60].sum(1) / 56,
            flow_d[:, 4:60].sum(2) / 64,
        ),
        1,
    )
    distance = np.stack([distance_transform_edt(~x) for x in d]) / 64
    df = d.astype(float)
    blurred = [uniform_filter(df, size=(1, w, w), mode="constant") for w in (3, 7, 11)]

    region = []
    distances = np.empty((n, 9, 3))
    coverages = np.empty((n, 9))
    for node, i in enumerate(indices):
        c = chips[i]
        s = np.s_[:, c.x : c.x + c.width, c.y : c.y + c.height]
        cover = d[s].mean((1, 2))
        ds = distance[s]
        dist = np.stack((ds.min((1, 2)), ds.mean((1, 2)), ds.max((1, 2))), 1)
        extras = np.stack([b[s].mean((1, 2)) for b in blurred], 1)
        pn = power[:, node]
        inter = (
            pn[None, :, None]
            * np.stack((dist[:, 1], dist[:, 2], 1 / (cover + 0.05)), 1)[:, None, :]
        ).reshape(n, 9)
        const = np.r_[centres[node], pn, pn.max(), pn.mean()]
        region.append(
            np.concatenate(
                (
                    np.broadcast_to(const, (n, 7)),
                    cover[:, None],
                    dist,
                    extras,
                    np.broadcast_to(upstream[:, node], (n, 3)),
                    inter,
                ),
                1,
            )
        )
        distances[:, node] = dist
        coverages[:, node] = cover
    heat = []
    for p in power:
        heat += [
            (p * distances[:, :, 1]).max(1),
            (p * distances[:, :, 2]).max(1),
            (p / (coverages + 0.05)).max(1),
            (p * distances[:, :, 1]).sum(1),
        ]
    x = np.concatenate((flow, *region, np.stack(heat, 1)), 1)
    return x.astype(np.float32), flow.astype(np.float32)


class SelectorV2:
    def __init__(self, path=SELECTOR):
        bundle = joblib.load(path)
        if (
            bundle["feature_version"] != FEATURE_VERSION
            or bundle["variant"] != "temperature"
        ):
            raise ValueError("unexpected selector bundle")
        self.margin = bundle["margin"]
        self.b = {k: v.booster_ for k, v in bundle["models"].items() if k != "rank"}

    def metrics(self, x, flow):
        p = lambda k, a: self.b[k].predict(a, num_threads=1)  # noqa: E731
        base = np.array(
            [p("uniform_nominal", x[8:9])[0], p("uniform_power", x[8:9])[0]]
        )
        rat = np.stack((p("relative_nominal", x), p("relative_power", x)), 1)
        rat[8] = 0
        return np.column_stack(
            (base[None] * (1 + rat), p("reynolds", flow), p("water", x))
        )

    def choose(self, x, flow):
        m = self.metrics(x, flow)
        return choose_tabular(m, self.margin, m[:, 0]), m


class SessionV2:
    """Raw task -> one binary channel map. Generator/decoder identical to v1."""

    def __init__(self, device="cuda", checkpoint=CHECKPOINT, selector=SELECTOR):
        model = load_solver_free(checkpoint, device)
        self.v1 = create_session(
            model, device=device, cuda_graphs=str(device).startswith("cuda")
        )
        self.selector = SelectorV2(selector)

    def infer(self, task):
        t0 = time.perf_counter()
        frame = canonical_frame(task)
        raw = raw_features(frame.task)
        geometry = RibbedGeometry.from_task(frame.task)
        with torch.inference_mode():
            if hasattr(self.v1, "graphed_generator"):
                q = self.v1.graphed_generator.generate(raw)
            else:  # CPU: plain forward pass, same weights
                q = (
                    self.v1.model(torch.from_numpy(raw[None]).to(self.v1.device))[
                        "parameters"
                    ][0]
                    .cpu()
                    .numpy()
                )
        t1 = time.perf_counter()
        designs = np.concatenate(
            (decode_many(geometry, q, list(range(4)) * 2), uniform_parallel()[None])
        )
        t2 = time.perf_counter()
        x, flow = fast_features(frame.task, designs)
        index, metrics = self.selector.choose(x, flow)
        index = 8 if frame.self_symmetric else index
        t3 = time.perf_counter()
        return dict(
            binary=frame.to_original(designs[index]),
            proposal_index=int(index),
            uniform_fallback=index == 8,
            predicted_metrics=metrics[index],
            online_physical_solves=0,
            total_ms=1000 * (t3 - t0),
            neural_ms=1000 * (t1 - t0),
            decoder_ms=1000 * (t2 - t1),
            selector_ms=1000 * (t3 - t2),
        )
