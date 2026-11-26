"""Physical-coordinate smooth trajectories with separate binary incumbents."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import uniform_parallel
from .design_gradient_cache import DesignGradientCache
from .joint_design import JointObjective
from .locked_evaluation import JointEvaluator, ranking, write
from .ribbed_channels import RibbedDecoder, RibbedGeometry
from .stage3_training import worker_init
from .stage5_physics import Stage5Solver


class RibbedEvaluator(JointEvaluator):
    def __init__(self, task, output):
        super().__init__(task, output)
        self.decoder = RibbedDecoder(RibbedGeometry.from_task(self.frame.task))

    def parameters(self, q, head):
        d = self.decoder.binary(
            torch.tensor(np.asarray(q)[None], dtype=torch.float64),
            [head],
            physical=True,
        )[0].numpy()
        return self.evaluate(d)

    def export(self, folder, row, parameters=None, head=None, **extra):
        result = super().export(folder, row, None, None, **extra)
        if parameters is not None:
            q = torch.tensor(parameters[None], dtype=torch.float64)
            coords = self.decoder.coordinates(q, [head], physical=True)
            core, wall, paths = self.decoder.masks(coords)
            # Display actual reserved paths, not possibly overlapping raw control lines.
            x = np.broadcast_to(np.arange(4, 60), (5, 56))
            xy = np.stack((x, paths[0]), -1).astype(float)
            if self.frame.reflected:
                xy[..., 1] = 63 - xy[..., 1]
            path = Path(folder) / "selected.npz"
            with np.load(path) as data:
                values = {k: data[k] for k in data.files}
            values.update(
                parameters=parameters,
                head=head,
                coordinates=xy,
                copper_rib_mask=self.frame.to_original(wall[0].numpy()),
                reserved_core=self.frame.to_original(core[0].numpy()),
            )
            np.savez_compressed(path, **values)
        result["parameter_space"] = (
            "physical offsets, half-widths, IO-x shifts, bounded pixel scores"
        )
        write(Path(folder) / "result.json", result)
        return result


class LegacyDecoder:
    """Unchanged older five-parallel-route parameterization, no chiplet anchors."""

    def binary(self, p, heads):
        from .stage5_decoder import binary

        return binary(p)

    def density(self, p, heads, *, beta, hard):
        from .stage5_decoder import density

        return density(p, beta=beta, hard=hard)[0]

    def coordinates(self, p, heads):
        from .stage5_decoder import KNOTS, waypoints

        y = waypoints(p)
        x = p.new_tensor(KNOTS)[None, None].expand_as(y)
        return torch.stack((x, y), -1)


class LegacyEvaluator(JointEvaluator):
    def __init__(self, task, output):
        super().__init__(task, output)
        self.decoder = LegacyDecoder()


def fit_uniform_start():
    """Encode the fixed hand template, with no thermal solves or task outcomes."""
    from .stage5_decoder import binary, continuous, initial_parameters

    p = torch.tensor(initial_parameters(), requires_grad=True)
    target = torch.tensor(uniform_parallel(), dtype=torch.float64)
    with torch.no_grad():
        p[8:12] = 0.12
        p[22] = 0
        p[25:].reshape(64, 64)[28:36, 34] = 6
    optimizer = torch.optim.Adam([p], lr=0.06)
    for i in range(501):
        if i % 10 == 0 and torch.equal(binary(p[None])[0], target.bool()):
            return p.detach().numpy()
        optimizer.zero_grad(set_to_none=True)
        d = continuous(p[None], beta=20)[0]
        loss = torch.nn.functional.binary_cross_entropy(
            d.clamp(1e-12, 1 - 1e-12), target
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            p[:8] = 0
            p[12:20] = 0
            p[:25].clamp_(-6, 6)
            p[25:].clamp_(-12, 12)
    raise RuntimeError("fixed Uniform template could not be encoded exactly")


def optimize_physical(evaluator, initial, head, *, kind, steps, checkpoint_every=1):
    """Never roll back parameters or shrink LR based on binary acceptance.

    The callback counts gradient objective queries. Binary acceptance is only
    an independent incumbent filter; it never controls the optimizer trajectory.
    """
    p = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    objective = JointObjective(Stage5Solver([evaluator.problem]))
    physical_cache = DesignGradientCache(objective)
    history = []
    candidates = []
    parameters = []
    started = time.perf_counter()

    def checkpoint(v):
        row = evaluator.parameters(v, head)
        candidates.append(row)
        parameters.append(v.copy())
        return row

    checkpoint(np.asarray(initial))

    def calculate(x):
        i = len(history)
        warm = steps // 2 if steps >= 100 else steps
        hard = steps >= 100 and i >= warm
        beta = 6.0 if hard or steps == 1 else 3 + 3 * i / max(warm - 1, 1)
        design = evaluator.decoder.density(
            x[None], [head], beta=beta, hard=hard, physical=True
        )[0]
        value, g, peak = physical_cache.evaluate(design, 2, cache=hard)
        gradient = torch.autograd.grad(design, x, g)[0]
        if not np.isfinite(value) or not torch.isfinite(gradient).all():
            raise RuntimeError("invalid physical trajectory gradient")
        history.append(
            dict(step=i + 1, loss=value, beta=beta, hard=hard, coarse_peak_c=peak)
        )
        if (i + 1) % 50 == 0:
            write(
                evaluator.output / "progress.json",
                dict(
                    kind=kind,
                    head=head,
                    objective_evaluations=i + 1,
                    requested=steps,
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
        return value, gradient

    if kind == "Adam":
        geometry = p[:29].detach().clone().requires_grad_(True)
        pixels = p[29:].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam(
            [{"params": [geometry], "lr": 0.12}, {"params": [pixels], "lr": 0.06}]
        )
        for i in range(steps):
            factor = (
                0.0 if steps == 1 else 0.5 * (1 + np.cos(np.pi * i / max(steps - 1, 1)))
            )
            opt.param_groups[0]["lr"] = 0.03 + 0.09 * factor
            opt.param_groups[1]["lr"] = 0.015 + 0.045 * factor
            opt.zero_grad(set_to_none=True)
            q = torch.cat((geometry, pixels))
            _, g = calculate(q)
            geometry.grad = g[:29]
            pixels.grad = g[29:]
            torch.nn.utils.clip_grad_norm_(
                [geometry, pixels], 10.0, error_if_nonfinite=True
            )
            opt.step()
            with torch.no_grad():
                geometry[:20].clamp_(-5.5, 5.5)
                geometry[20:25].clamp_(0.6, 2.2)
                geometry[25:29].clamp_(-2, 2)
                pixels.clamp_(-1, 1)
            history[-1].update(
                lr_branch=opt.param_groups[0]["lr"], lr_pixel=opt.param_groups[1]["lr"]
            )
            if (i + 1) % checkpoint_every == 0 or i + 1 == steps:
                checkpoint(torch.cat((geometry, pixels)).detach().numpy())
    elif kind == "MMA":
        import nlopt

        x = np.asarray(initial).copy()
        while len(history) < steps:
            before = len(history)
            last = x.copy()
            opt = nlopt.opt(nlopt.LD_MMA, len(x))
            lo = np.r_[
                np.full(20, -5.5),
                np.full(5, 0.6),
                np.full(4, -2.0),
                np.full(4096, -1.0),
            ]
            hi = np.r_[
                np.full(20, 5.5), np.full(5, 2.2), np.full(4, 2.0), np.full(4096, 1.0)
            ]
            opt.set_lower_bounds(lo)
            opt.set_upper_bounds(hi)
            opt.set_maxeval(steps - before)

            def function(v, grad):
                nonlocal last
                last = v.copy()
                value, g = calculate(torch.tensor(v, requires_grad=True))
                if grad.size:
                    grad[:] = g.detach().numpy()
                if len(history) % checkpoint_every == 0:
                    checkpoint(v)
                return value

            opt.set_min_objective(function)
            try:
                x = opt.optimize(x)
            except nlopt.RoundoffLimited:
                x = last
            if len(history) == before:
                raise RuntimeError("MMA consumed no evaluations")
        checkpoint(x)
    else:
        raise ValueError("unknown optimizer")
    best = min(range(len(candidates)), key=lambda i: ranking(candidates[i]))
    return dict(
        raw=candidates[best],
        parameters=parameters[best],
        head=head,
        history=history,
        checkpoints=candidates,
        objective_evaluations=len(history),
        physical_objective_solves=physical_cache.misses,
        elapsed_seconds=time.perf_counter() - started,
        distinct_binary_maps=len({r["binary_sha256"] for r in candidates}),
    )


def select_and_export(evaluator, folder, trajectories, uniform, *, method, **extra):
    best = min(trajectories, key=lambda r: ranking(r["raw"]))
    fallback = evaluator.frame.self_symmetric or ranking(uniform) < ranking(best["raw"])
    selected = uniform if fallback else best["raw"]
    return evaluator.export(
        folder,
        selected,
        None if fallback else best["parameters"],
        None if fallback else best["head"],
        method=method,
        raw=best["raw"],
        fallback_used=fallback,
        objective_evaluations=sum(t["objective_evaluations"] for t in trajectories),
        physical_objective_solves=sum(
            t.get("physical_objective_solves", 0) for t in trajectories
        ),
        elapsed_seconds=sum(t.get("elapsed_seconds", 0) for t in trajectories),
        trajectories=[
            {k: v for k, v in t.items() if k != "parameters"} for t in trajectories
        ],
        **extra,
    )


def matched_reference(task, output, *, kind):
    worker_init()
    output = Path(output)
    if (output / "result.json").exists():
        return json.loads((output / "result.json").read_text())
    evaluator = RibbedEvaluator(task, output)
    uniform = evaluator.evaluate(uniform_parallel())
    seeds = []
    for h in range(4):
        q = RibbedDecoder.to_physical(torch.tensor(RibbedDecoder.initial(h)[None]))[
            0
        ].numpy()
        seeds.append((q, h, evaluator.parameters(q, h)))
    q, h, _ = min(seeds, key=lambda v: ranking(v[2]))
    trajectory = optimize_physical(
        evaluator, q, h, kind=kind, steps=600, checkpoint_every=50
    )
    return select_and_export(
        evaluator,
        output,
        [trajectory],
        uniform,
        method="Matched-" + kind + "600",
        initial_physics_scores=4,
        budget_label="1x600",
    )


def neural_case(task, predictions, output):
    worker_init()
    output = Path(output)
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    started = time.perf_counter()
    evaluator = RibbedEvaluator(task, output)
    uniform = evaluator.evaluate(uniform_parallel())
    if not uniform["valid"]:
        raise RuntimeError("Uniform fallback invalid")
    write(output / "uniform.json", uniform)
    q = RibbedDecoder.to_physical(torch.tensor(predictions)).numpy()
    starts = [
        dict(
            raw=evaluator.parameters(p, h),
            parameters=p,
            head=h,
            history=[],
            checkpoints=[],
            objective_evaluations=0,
        )
        for h, p in enumerate(q)
    ]
    starts[0]["elapsed_seconds"] = time.perf_counter() - started
    methods = {}
    methods["R0"] = select_and_export(
        evaluator, output / "R0", starts, uniform, method="R0", initial_physics_scores=4
    )
    best = min(starts, key=lambda v: ranking(v["raw"]))
    single = optimize_physical(
        evaluator, best["parameters"], best["head"], kind="Adam", steps=25
    )
    methods["R25"] = select_and_export(
        evaluator,
        output / "R25",
        [single, *starts],
        uniform,
        method="R25-smooth-single",
    )
    top = sorted(starts, key=lambda v: ranking(v["raw"]))[:3]
    segments = [
        optimize_physical(evaluator, t["parameters"], t["head"], kind="Adam", steps=8)
        for t in top
    ]
    leader = min(segments, key=lambda v: ranking(v["raw"]))
    final = optimize_physical(
        evaluator, leader["parameters"], leader["head"], kind="Adam", steps=1
    )
    methods["MultiR25"] = select_and_export(
        evaluator,
        output / "MultiR25",
        [*starts, *segments, final],
        uniform,
        method="R25-smooth-multi",
        budget_label="3x8+1=25",
        initial_physics_scores=4,
    )
    if methods["MultiR25"]["objective_evaluations"] != 25:
        raise RuntimeError("multihead budget mismatch")
    write(output / "complete.json", methods)
    return methods
