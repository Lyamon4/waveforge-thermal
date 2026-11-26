"""Raw geometric inputs and learned decisions; no online physical solve."""

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .fast_channels import decode_one


def raw_features(task):
    """Cell-integrated watts, chip mask, ports and coordinates; no PDE fields."""
    powers = np.asarray(task.powers_w, dtype=np.float64)
    if powers.shape != (3, len(task.chiplets)):
        raise ValueError("three workloads and matching chiplets required")
    x = np.zeros((7, 64, 64), dtype=np.float32)
    for i, chip in enumerate(task.chiplets):
        a, b = chip.x, chip.y
        sx, sy = chip.width, chip.height
        if (
            a < 0
            or b < 0
            or a + sx > 64
            or b + sy > 64
            or x[3, a : a + sx, b : b + sy].any()
        ):
            raise ValueError("nonoverlapping chiplets inside 64x64 required")
        x[:3, a : a + sx, b : b + sy] = (powers[:, i] / (sx * sy))[:, None, None]
        x[3, a : a + sx, b : b + sy] = 1
    x[4, (0, 63), 24:40] = 1
    t = np.arange(64, dtype=np.float32) / np.float32(63)
    x[5] = t[:, None]
    x[6] = t[None, :]
    if not np.isfinite(x).all() or (powers < 0).any():
        raise ValueError("finite nonnegative powers required")
    return x


def ranking_loss(logits, temperatures, admissible, *, tolerance=0.005):
    """Soft near-optimal labels avoid forcing arbitrary ordering of tied designs."""
    if logits.shape != temperatures.shape or admissible.shape != logits.shape:
        raise ValueError("aligned candidate scores and physical labels required")
    if not admissible.any(1).all():
        raise ValueError("each training task needs an admitted candidate")
    best = temperatures.masked_fill(~admissible, torch.inf).amin(1, keepdim=True)
    regret = (temperatures - best) / (best - 25).clamp_min(1e-6)
    targets = torch.softmax(
        (-regret / tolerance).masked_fill(~admissible, -torch.inf), 1
    )
    classification = -(targets * F.log_softmax(logits, 1)).sum(1).mean()
    cost = torch.where(admissible, regret.clamp_min(0), torch.ones_like(regret))
    return classification + (torch.softmax(logits, 1) * cost).sum(1).mean()


def _block(a, b, *, stride=1):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1, stride=stride),
        nn.SiLU(),
        nn.Conv2d(b, b, 3, padding=1),
        nn.SiLU(),
    )


class RawConditioner(nn.Module):
    """Predict six learned features, never call a field or sensitivity solver."""

    def __init__(self):
        super().__init__()
        self.e0 = _block(7, 24)
        self.e1 = _block(24, 48, stride=2)
        self.e2 = _block(48, 72, stride=2)
        self.e3 = _block(72, 96, stride=2)
        self.u2 = _block(96 + 72, 72)
        self.u1 = _block(72 + 48, 48)
        self.u0 = _block(48 + 24, 24)
        self.fields = nn.Conv2d(24, 6, 1)

    def forward(self, raw):
        a = self.e0(raw)
        b = self.e1(a)
        c = self.e2(b)
        d = self.e3(c)
        up = self.u2(
            torch.cat(
                (
                    F.interpolate(
                        d, size=c.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    c,
                ),
                1,
            )
        )
        up = self.u1(
            torch.cat(
                (
                    F.interpolate(
                        up, size=b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b,
                ),
                1,
            )
        )
        up = self.u0(
            torch.cat(
                (
                    F.interpolate(
                        up, size=a.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    a,
                ),
                1,
            )
        )
        fields = self.fields(up)
        fields = torch.cat((fields[:, :3] * raw[:, 3:4], fields[:, 3:].tanh()), 1)
        context = torch.cat(
            (
                F.adaptive_avg_pool2d(d, (4, 4)).flatten(1),
                F.adaptive_avg_pool2d(raw[:, :3], (4, 8)).flatten(1),
            ),
            1,
        )
        condition = torch.cat((raw[:, :3], fields, raw[:, 3:]), 1)
        return condition, context


class RawPortfolioPolicy(nn.Module):
    """A trainable raw-input bridge and selector around immutable proposal nets."""

    def __init__(self, teacher):
        super().__init__()
        self.teacher = teacher.eval().requires_grad_(False)
        self.adapter = RawConditioner()
        self.context = nn.Sequential(nn.Linear(96 * 16 + 3 * 32, 128), nn.SiLU())
        self.selector = nn.Sequential(
            nn.Linear(128 + 29 + 32 + 9, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 6),
        )
        self.register_buffer("head_identity", torch.eye(9))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.teacher.requires_grad_(False)
        return self

    def forward(self, raw):
        condition, context = self.adapter(raw)
        q = self.teacher(condition)
        decision = self.decide(q, context)
        return dict(parameters=q, condition=condition, **decision)

    def decide(self, q, context):
        # Decision loss must not manipulate the generator to fool its own scorer.
        context = self.context(context.detach())[:, None].expand(-1, 9, -1)
        fixed = q.detach().float()
        local = F.adaptive_avg_pool2d(
            fixed[:, :, 29:].reshape(-1, 8, 64, 64), (4, 8)
        ).flatten(2)
        actions = torch.cat((fixed[:, :, :29], local), 2)
        actions = F.pad(actions, (0, 0, 0, 1))
        identity = self.head_identity[None].expand(q.shape[0], -1, -1)
        features = torch.cat((context, actions, identity), 2)
        prediction = self.selector(features)
        return dict(
            logits=prediction[:, :, 0],
            metrics=prediction[:, :, 1:5],
            validity_logit=prediction[:, :, 5],
        )


class SolverFreeSession:
    """Loaded online session. Physical checks belong to a separate audit command."""

    def __init__(self, model, *, device="cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)

    def infer(self, task):
        from .baselines import uniform_parallel
        from .locked_channels import canonical_frame
        from .ribbed_channels import RibbedGeometry

        started = time.perf_counter()
        frame = canonical_frame(task)
        raw = raw_features(frame.task)
        geometry = RibbedGeometry.from_task(frame.task)
        prepared = time.perf_counter()
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(raw[None]).to(self.device))
            index = int(prediction["logits"][0].argmax().item())
            q = prediction["parameters"][0, min(index, 7)].cpu().numpy()
            metrics = prediction["metrics"][0, index].cpu().numpy()
        generated = time.perf_counter()
        fallback = frame.self_symmetric or index == 8
        binary = uniform_parallel() if fallback else decode_one(geometry, q, index % 4)
        binary = frame.to_original(binary)
        finished = time.perf_counter()
        return dict(
            binary=binary,
            parameters=np.empty(0) if fallback else q,
            head=-1 if fallback else index % 4,
            proposal_index=index,
            proposal_count=8,
            uniform_fallback=fallback,
            predicted_metrics=metrics,
            physics_checked=False,
            online_physical_solves=0,
            total_ms=1000 * (finished - started),
            preparation_ms=1000 * (prepared - started),
            neural_ms=1000 * (generated - prepared),
            decoder_ms=1000 * (finished - generated),
        )


def load_solver_free(path, device="cuda"):
    from .amortized_policy import CandidatePortfolio

    saved = torch.load(path, map_location=device, weights_only=True)
    architecture = saved.get("architecture")
    if architecture == "GeneratorOnlyPolicy":
        import joblib

        from .solver_free_tabular import (
            BRANCH_FEATURE_VERSION,
            FEATURE_VERSION,
            GeneratorOnlyPolicy,
        )

        model = GeneratorOnlyPolicy().to(device)
        selector = Path(path).resolve().parent / saved["selector_file"]
        if selector.resolve().parent != Path(path).resolve().parent:
            raise ValueError("selector must accompany the neural checkpoint")
        if (
            hashlib.sha256(selector.read_bytes()).hexdigest()
            != saved["selector_sha256"]
        ):
            raise ValueError("selector artifact changed")
        model.selector_bundle = joblib.load(selector)
        if model.selector_bundle["feature_version"] not in (
            FEATURE_VERSION,
            BRANCH_FEATURE_VERSION,
        ):
            raise ValueError("wrong structured feature version")
    elif architecture == "RasterRawPolicy":
        from .solver_free_raster import RasterRawPolicy

        model = RasterRawPolicy(
            hydraulic_branch=saved["config"].get("hydraulic_branch", False),
            hydraulic_features=saved["config"].get("hydraulic_features", False),
        ).to(device)
    elif architecture == "RawPortfolioPolicy":
        model = RawPortfolioPolicy(CandidatePortfolio(tuple(range(8)))).to(device)
    else:
        raise ValueError("a raw-input solver-free checkpoint is required")
    model.load_state_dict(saved["model"], strict=True)
    if "inference" in saved:
        model.selection_rule = saved["inference"]["selection_rule"]
        model.re_margin = saved["inference"]["re_margin"]
    return model.eval().requires_grad_(False)


def create_session(model, device="cuda", *, cuda_graphs=False):
    if hasattr(model, "selector_bundle"):
        from .solver_free_tabular import TabularSession

        session = TabularSession(model, device=device)
    elif hasattr(model, "critic"):
        from .solver_free_raster import RasterSession

        session = RasterSession(model, device=device)
    else:
        if cuda_graphs:
            raise ValueError("graph capture requires a raster or structured selector")
        return SolverFreeSession(model, device=device)
    if cuda_graphs:
        from .solver_free_graph import GraphedGenerator

        session.graphed_generator = GraphedGenerator(session.model)
    return session
