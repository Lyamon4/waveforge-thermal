"""Structured learned scoring of neural channel proposals, without a PDE solve."""

import time

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, uniform_filter
from torch import nn

from .fast_channels import decode_many, flow_geometry_features
from .locked_channels import canonical_frame
from .solver_free import RawConditioner, raw_features

FEATURE_VERSION = "binary-geometry-519-v1"
BRANCH_FEATURE_VERSION = "binary-branch-690-v2"


def candidate_features(task, designs, *, version=FEATURE_VERSION):
    frame = canonical_frame(task)
    task = frame.task
    designs = frame.to_original(np.asarray(designs, dtype=bool))
    chips = task.chiplets
    indices = sorted(
        (i for i, c in enumerate(chips) if c.width == 8),
        key=lambda i: (chips[i].y >= 32, chips[i].x),
    )
    indices += [i for i, c in enumerate(chips) if c.width == 12]
    if len(indices) != 9:
        raise ValueError("eight compute regions and one I/O required")
    power = np.asarray(task.powers_w)[:, indices] / 100
    order = sorted(range(3), key=lambda k: tuple(power[k]))
    power = power[order]
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
        weights = (centres[:, 0] < centres[i, 0]) * np.exp(
            -0.5 * ((centres[:, 1] - centres[i, 1]) / 0.2) ** 2
        )
        upstream[:, i] = power @ weights
    thermal, flows = [], []
    seen = {}
    for d in designs:
        key = d.tobytes()
        if key in seen:
            previous = seen[key]
            thermal.append(thermal[previous])
            flows.append(flows[previous])
            continue
        seen[key] = len(thermal)
        mirror = d[:, ::-1]
        flow_d = d if d.tobytes() < mirror.tobytes() else mirror
        flow = np.concatenate(
            (
                flow_geometry_features(flow_d[None])[0],
                flow_d.reshape(8, 8, 8, 8).mean((1, 3)).ravel(),
                flow_d[4:60].sum(0) / 56,
                flow_d[4:60].sum(1) / 64,
            )
        )
        distance = distance_transform_edt(~d) / 64
        blurred = [
            uniform_filter(d.astype(float), size=w, mode="constant") for w in (3, 7, 11)
        ]
        region = []
        distances = []
        coverages = []
        for node, i in enumerate(indices):
            c = chips[i]
            s = np.s_[c.x : c.x + c.width, c.y : c.y + c.height]
            cover = d[s].mean()
            dist = np.array([distance[s].min(), distance[s].mean(), distance[s].max()])
            extras = np.array([b[s].mean() for b in blurred])
            interaction = (
                power[:, node, None]
                * np.array([dist[1], dist[2], 1 / (cover + 0.05)])[None]
            ).ravel()
            region.extend(
                np.r_[
                    centres[node],
                    power[:, node],
                    power[:, node].max(),
                    power[:, node].mean(),
                    cover,
                    dist,
                    extras,
                    upstream[:, node],
                    interaction,
                ]
            )
            distances.append(dist)
            coverages.append(cover)
        distances = np.asarray(distances)
        coverages = np.asarray(coverages)
        global_heat = []
        for p in power:
            global_heat.extend(
                [
                    (p * distances[:, 1]).max(),
                    (p * distances[:, 2]).max(),
                    (p / (coverages + 0.05)).max(),
                    (p * distances[:, 1]).sum(),
                ]
            )
        thermal.append(np.r_[flow, region, global_heat])
        flows.append(flow)
    x = np.asarray(thermal, np.float32)
    if version == BRANCH_FEATURE_VERSION:
        x = np.concatenate((x, branch_heat_features(task, designs)), 1)
    elif version != FEATURE_VERSION:
        raise ValueError("unknown candidate feature version")
    return x, np.asarray(flows, np.float32)


def branch_heat_features(task, designs):
    """Geometry-weighted upstream power sums; no pressure/temperature equations."""
    wet = designs[:, 4:60, 7:57]
    starts = wet & ~np.pad(wet[:, :, :-1], ((0, 0), (0, 0), (1, 0)))
    labels = np.minimum(starts.cumsum(2), 5)
    widths = []
    centres = []
    y = np.arange(7, 57)
    for branch in range(1, 6):
        m = wet & (labels == branch)
        w = m.sum(2)
        widths.append(w)
        centres.append((m * y).sum(2) / np.maximum(w, 1))
    widths = np.stack(widths, 1).astype(float)
    centres = np.pad(np.stack(centres, 1), ((0, 0), (0, 0), (4, 4)), mode="edge")
    conductance = 1 / (1 / np.maximum(widths, 0.001)).sum(2)
    power = raw_features(task)[:3].astype(float)
    order = sorted(range(3), key=lambda i: tuple(np.asarray(task.powers_w)[i]))
    power = power[order]
    indices = sorted(
        (i for i, c in enumerate(task.chiplets) if c.width == 8),
        key=lambda i: (task.chiplets[i].y >= 32, task.chiplets[i].x),
    )
    indices += [i for i, c in enumerate(task.chiplets) if c.width == 12]
    columns = []
    for reach in (2.0, 4.0, 8.0):
        distance = np.arange(64)[None, None, None, :] - centres[:, :, :, None]
        weight = np.exp(-0.5 * (distance / reach) ** 2)
        weight /= weight.sum(1, keepdims=True).clip(1e-30)
        heat = np.einsum("sxy,nbxy->nsbx", power, weight, optimize=False)
        upstream = heat.cumsum(3) - 0.5 * heat
        upstream /= conductance[:, None, :, None]
        field = np.einsum("nsbx,nbxy->nsxy", upstream, weight, optimize=False) / 10000
        for i in indices:
            chip = task.chiplets[i]
            region = field[
                :, :, chip.x : chip.x + chip.width, chip.y : chip.y + chip.height
            ]
            columns.extend((region.mean((2, 3)), region.max((2, 3))))
        columns.append(field.max((2, 3)))
    return np.concatenate(columns, 1).astype(np.float32)


class GeneratorOnlyPolicy(nn.Module):
    def __init__(self, generator=None):
        super().__init__()
        if generator is None:
            from .amortized_policy import CandidatePortfolio

            self.adapter = RawConditioner()
            self.teacher = CandidatePortfolio(tuple(range(8)))
        else:
            self.adapter = generator.adapter
            self.teacher = generator.teacher

    def forward(self, raw):
        condition, _ = self.adapter(raw)
        return dict(parameters=self.teacher(condition))


def predict_metrics(bundle, features, flow):
    models = bundle["models"]

    def predict(name, x):
        if "_packed_models" in bundle:
            from .fast_trees import predict_packed

            return predict_packed(bundle["_packed_models"][name], x)
        return models[name].predict(x[:, : models[name].n_features_in_])

    baseline = np.array(
        [predict(k, features[-1:])[0] for k in ("uniform_nominal", "uniform_power")]
    )
    ratios = np.stack(
        [predict(k, features) for k in ("relative_nominal", "relative_power")], 1
    )
    ratios[-1] = 0
    temperatures = baseline[None] * (1 + ratios)
    reynolds = predict("reynolds", flow)
    water = predict("water", features)
    return np.column_stack((temperatures, reynolds, water))


def pair_scores(bundle, features, metrics):
    if "pairwise" not in bundle["models"]:
        return metrics[:, 0]
    a, b = np.triu_indices(8, 1)
    features = features[:, : bundle["models"]["pairwise"].n_features_in_ // 2]
    mean = (features[a] + features[b]) * 0.5
    difference = features[a] - features[b]
    rows = np.concatenate(
        (np.column_stack((mean, difference)), np.column_stack((mean, -difference)))
    )
    if "_packed_models" in bundle:
        from .fast_trees import predict_packed

        values = predict_packed(bundle["_packed_models"]["pairwise"], rows)
    else:
        values = bundle["models"]["pairwise"].predict(rows)
    delta = (values[: len(a)] - values[len(a) :]) * 0.5
    scores = np.zeros(9)
    np.add.at(scores, a, delta)
    np.add.at(scores, b, -delta)
    scores[:8] = scores[:8] / 8 + metrics[:8, 0].mean()
    scores[8] = metrics[8, 0]
    blend = bundle.get("pair_blend", 1.0)
    return blend * scores + (1 - blend) * metrics[:, 0]


def choose_tabular(metrics, margin, scores=None):
    valid = (
        np.isfinite(metrics).all(1)
        & (metrics[:, 2] >= 0)
        & (metrics[:, 2] + margin < 1)
        & (metrics[:, 3] < 1)
        & (metrics[:, :2].max(1) <= 1.4)
    )
    values = metrics[:, 0] if scores is None else scores
    return int(np.where(valid, values, np.inf).argmin()) if valid.any() else 8


class TabularSession:
    def __init__(self, model, *, device="cuda"):
        from .fast_trees import pack_predictor

        self.device = torch.device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.model.selector_bundle["_packed_models"] = {
            name: pack_predictor(estimator)
            for name, estimator in model.selector_bundle["models"].items()
        }

    def infer(self, task):
        from .baselines import uniform_parallel
        from .ribbed_channels import RibbedGeometry

        started = time.perf_counter()
        frame = canonical_frame(task)
        raw = raw_features(frame.task)
        geometry = RibbedGeometry.from_task(frame.task)
        prepared = time.perf_counter()
        with torch.inference_mode():
            q = (
                self.graphed_generator.generate(raw)
                if hasattr(self, "graphed_generator")
                else (
                    self.model(torch.from_numpy(raw[None]).to(self.device))[
                        "parameters"
                    ][0]
                    .cpu()
                    .numpy()
                )
            )
        generated = time.perf_counter()
        designs = np.concatenate(
            (decode_many(geometry, q, list(range(4)) * 2), uniform_parallel()[None])
        )
        decoded = time.perf_counter()
        features, flow = candidate_features(
            frame.task, designs, version=self.model.selector_bundle["feature_version"]
        )
        metrics = predict_metrics(self.model.selector_bundle, features, flow)
        scores = pair_scores(self.model.selector_bundle, features, metrics)
        index = choose_tabular(metrics, self.model.selector_bundle["re_margin"], scores)
        index = 8 if frame.self_symmetric else index
        fallback = index == 8
        binary = frame.to_original(designs[index])
        finished = time.perf_counter()
        return dict(
            binary=binary,
            _all_parameters=q,
            parameters=np.empty(0) if fallback else q[index],
            head=-1 if fallback else index % 4,
            proposal_index=index,
            proposal_count=8,
            decoded_proposals=8,
            uniform_fallback=fallback,
            predicted_metrics=metrics[index],
            physics_checked=False,
            online_physical_solves=0,
            total_ms=1000 * (finished - started),
            preparation_ms=1000 * (prepared - started),
            neural_ms=1000 * (generated - prepared),
            decoder_ms=1000 * (decoded - generated),
            selector_ms=1000 * (finished - decoded),
        )
