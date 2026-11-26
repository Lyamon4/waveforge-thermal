"""Shared geometry-aware critic; every online operation is solver-free."""

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .fast_channels import decode_many, flow_geometry_features
from .solver_free import RawConditioner, _block, raw_features


def choose_predictions(prediction, *, re_margin=0.0):
    """Predicted feasibility precedes temperature; this is not a physical proof."""
    m = prediction["metrics"]
    feasible = (
        torch.isfinite(m).all(2)
        & (m[:, :, 2] >= 0)
        & (m[:, :, 2] + re_margin < 1)
        & (m[:, :, 3] < 1)
        & (m[:, :, :2].amax(2) <= 1.4)
    )
    scores = m[:, :, 0].masked_fill(~feasible, torch.inf)
    choice = scores.argmin(1)
    return torch.where(feasible.any(1), choice, torch.full_like(choice, 8))


class RasterCritic(nn.Module):
    """Equal binary designs receive equal scores, independent of proposal origin."""

    def __init__(self, *, hydraulic_branch=False, hydraulic_features=False):
        super().__init__()
        self.hydraulic_branch = hydraulic_branch
        self.hydraulic_features = hydraulic_features
        self.encoder = nn.Sequential(
            _block(8, 24),
            _block(24, 40, stride=2),
            _block(40, 64, stride=2),
            _block(64, 80, stride=2),
        )
        self.value = nn.Sequential(
            nn.Linear(80 * 16 + 80 + 9, 160),
            nn.SiLU(),
            nn.Linear(160, 64),
            nn.SiLU(),
            nn.Linear(64, 5),
        )
        if hydraulic_branch:
            self.flow = nn.Sequential(
                _block(1, 16),
                _block(16, 24, stride=2),
                _block(24, 32, stride=2),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32 * 16, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )

        if hydraulic_features:
            self.geometry_flow = nn.Sequential(
                nn.Linear(89, 96),
                nn.SiLU(),
                nn.Linear(96, 48),
                nn.SiLU(),
                nn.Linear(48, 1),
            )
            nn.init.zeros_(self.geometry_flow[-1].weight)
            nn.init.zeros_(self.geometry_flow[-1].bias)

    def forward(self, raw, designs, flow_features=None):
        batch, count = designs.shape[:2]
        d = designs.float().reshape(-1, 1, 64, 64)
        x = raw[:, None].expand(-1, count, -1, -1, -1).reshape(-1, 7, 64, 64)
        hidden = self.encoder(torch.cat((x, d), 1))
        feature = torch.cat(
            (F.adaptive_avg_pool2d(hidden, (4, 4)).flatten(1), hidden.amax((-2, -1))), 1
        )
        power = x[:, :3]
        total = power.sum((-2, -1)).clamp_min(1e-6)
        coverage = []
        for width in (1, 5, 11):
            nearby = (
                d
                if width == 1
                else F.avg_pool2d(d, width, stride=1, padding=width // 2)
            )
            coverage.append((power * nearby).sum((-2, -1)) / total)
        values = self.value(torch.cat((feature, *coverage), 1)).reshape(batch, count, 5)
        metrics = values[:, :, :4]
        valid = values[:, :, 4]
        if self.hydraulic_branch:
            re = self.flow(d).reshape(batch, count)
            if self.hydraulic_features:
                if flow_features is None:
                    raise ValueError("geometric flow descriptors are required")
                re = re + self.geometry_flow(flow_features).reshape(batch, count)
            metrics = torch.stack(
                (metrics[:, :, 0], metrics[:, :, 1], re, metrics[:, :, 3]), 2
            )
            valid = torch.minimum((1 - re) / 0.025, (1 - metrics[:, :, 3]) / 0.05)
            valid = torch.minimum(valid, (1.4 - metrics[:, :, :2].amax(2)) / 0.05)
        logits = -metrics[:, :, 0] / 0.004 + 4 * F.logsigmoid(valid)
        return dict(logits=logits, metrics=metrics, validity_logit=valid)


class RasterRawPolicy(nn.Module):
    def __init__(
        self, generator=None, *, hydraulic_branch=False, hydraulic_features=False
    ):
        super().__init__()
        if generator is None:
            from .amortized_policy import CandidatePortfolio

            self.adapter = RawConditioner()
            self.teacher = CandidatePortfolio(tuple(range(8)))
        else:
            self.adapter = generator.adapter
            self.teacher = generator.teacher
        self.adapter.requires_grad_(False).eval()
        self.teacher.requires_grad_(False).eval()
        self.critic = RasterCritic(
            hydraulic_branch=hydraulic_branch, hydraulic_features=hydraulic_features
        )

    def train(self, mode=True):
        super().train(mode)
        self.adapter.eval()
        self.teacher.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.adapter.requires_grad_(False)
        self.teacher.requires_grad_(False)
        return self

    def forward(self, raw):
        condition, _ = self.adapter(raw)
        return dict(parameters=self.teacher(condition))


class RasterSession:
    """Decode cheap proposals in a batch, score them neurally, return one map."""

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
            if hasattr(self, "graphed_generator"):
                parameters = self.graphed_generator.generate(raw)
                x = self.graphed_generator.raw
            else:
                x = torch.from_numpy(raw[None]).to(self.device)
                parameters = self.model(x)["parameters"][0].cpu().numpy()
        generated = time.perf_counter()
        designs = decode_many(geometry, parameters, list(range(4)) * 2)
        designs = np.concatenate((designs, uniform_parallel()[None]))
        decoded = time.perf_counter()
        with torch.inference_mode():
            features = None
            if self.model.critic.hydraulic_features:
                features = torch.from_numpy(flow_geometry_features(designs)[None]).to(
                    self.device
                )
            pred = self.model.critic(
                x,
                torch.from_numpy(designs[None]).to(self.device),
                flow_features=features,
            )
            index = (
                int(choose_predictions(pred, re_margin=self.model.re_margin)[0].item())
                if getattr(self.model, "selection_rule", None) == "hard_metrics"
                else int(pred["logits"][0].argmax().item())
            )
            metrics = pred["metrics"][0, index].cpu().numpy()
        fallback = frame.self_symmetric or index == 8
        index = 8 if fallback else index
        binary = frame.to_original(designs[index])
        finished = time.perf_counter()
        return dict(
            binary=binary,
            _all_parameters=parameters,
            parameters=np.empty(0) if fallback else parameters[index],
            head=-1 if fallback else index % 4,
            proposal_index=index,
            proposal_count=8,
            decoded_proposals=8,
            uniform_fallback=fallback,
            predicted_metrics=metrics,
            physics_checked=False,
            online_physical_solves=0,
            total_ms=1000 * (finished - started),
            preparation_ms=1000 * (prepared - started),
            neural_ms=1000 * ((generated - prepared) + (finished - decoded)),
            decoder_ms=1000 * (decoded - generated),
        )
