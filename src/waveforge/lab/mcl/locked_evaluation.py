"""One jointly admitted design per method; the same decoder for all methods."""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from waveforge.experiments.audit_mcl_stage5_admission import record
from waveforge.lab.hydraulic_probe import (
    FlowResult,
    HydraulicProbe,
    refined_spec,
    replicated_design,
)

from .baselines import uniform_parallel
from .contracts import problem_from_task
from .design_gradient_cache import DesignGradientCache
from .dual_profile import record_result, solve_dual_profile
from .high_fidelity_eval import refine_problem
from .joint_design import JointObjective, joint_admission, joint_flow_penalty
from .locked_channels import LockedDecoder, LockedGeometry, canonical_frame
from .stage3_training import worker_init
from .stage5_evaluation import PUMP_W
from .stage5_physics import Stage5Solver, stage5_spec


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def ranking(row):
    return (not row["valid"], 0 if row["valid"] else row["violation"], row["peak_c"])


class JointEvaluator:
    def __init__(self, task, output, *, forward_backend="fast"):
        if forward_backend not in ("reference", "fast"):
            raise ValueError("unknown candidate evaluation backend")
        self.forward_backend = forward_backend
        self.frame = canonical_frame(task)
        self.problem = problem_from_task(self.frame.task, stage5_spec())
        self.decoder = LockedDecoder(LockedGeometry.from_task(self.frame.task))
        self.fine_spec = refined_spec(self.problem.spec)
        self.probe = HydraulicProbe(self.fine_spec)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache, self.fields, self.designs = {}, {}, {}
        self.started = time.perf_counter()

    def evaluate(self, design):
        d = np.asarray(design, dtype=bool)
        key = hashlib.sha256(d.tobytes()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        modes, fields = {}, {}
        violation = 0.0
        coarse_temperature = None
        for grid in (64, 128):
            p, tiled = refine_problem(self.problem, d, grid // 64)
            if self.forward_backend == "fast":
                nom_fields, pow_fields = solve_dual_profile(
                    tiled, p, pump_power_w=PUMP_W
                )
                nominal = record_result(tiled, p, nom_fields)
                pressure = np.sqrt(
                    PUMP_W * p.spec.pressure_drop_pa / (nominal["flow_ml_s"] * 1e-6)
                )
                power = record_result(tiled, p, pow_fields, pressure)
            else:
                nominal, nom_fields = record(tiled, p)
                pressure = np.sqrt(
                    PUMP_W * p.spec.pressure_drop_pa / (nominal["flow_ml_s"] * 1e-6)
                )
                power, pow_fields = record(tiled, p, pressure)
            if grid == 64:
                coarse_temperature = (
                    nom_fields.temperature[:, 0, p.chip_mask].ravel() / 100
                )
            modes[str(grid)] = dict(nominal=nominal, power=power)
            for row in (nominal, power):
                diag = row["diagnostics"]
                violation += max(0.0, diag["max_local_reynolds_estimate"] / 1200 - 1)
                violation += max(0.0, diag["peak_open_water_temperature_c"] / 60 - 1)
                violation += max(0.0, row["silicon_peak_c"] / 95 - 1)
                if (
                    not row["admission"]["numerical"]
                    or not row["admission"]["conservation"]
                ):
                    violation += 1e6
            if grid == 128:
                fields = dict(nominal=nom_fields, power=pow_fields)
        gate = joint_admission(modes)
        # Exactly the training objective, evaluated without an adjoint, ranks
        # search parents. Independent nominal128 temperature still selects finals.
        with torch.no_grad():
            tensor = torch.tensor(d[None], dtype=torch.float64)
            if self.forward_backend == "fast":
                f = fields["nominal"]
                flow = FlowResult(
                    *(
                        torch.from_numpy(np.asarray(v)[None])
                        for v in (f.pressure, f.qx, f.qy, f.qin, f.qout)
                    ),
                    torch.tensor(
                        [self.fine_spec.pressure_drop_pa], dtype=torch.float64
                    ),
                )
            else:
                flow = self.probe(replicated_design(tensor))
            penalty = joint_flow_penalty(flow, self.fine_spec).item()
            thermal = (
                torch.logsumexp(500 * torch.tensor(coarse_temperature), 0)
                - np.log(len(coarse_temperature))
            ).item() / 500
        tv = (
            np.abs(np.diff(d.astype(float), axis=0)).mean()
            + np.abs(np.diff(d.astype(float), axis=1)).mean()
        )
        row = dict(
            binary_sha256=key,
            valid=gate["admissible"],
            violation=violation,
            peak_c=modes["128"]["nominal"]["silicon_peak_c"],
            modes=modes,
            gate=gate,
            search_objective=thermal + 0.001 * tv + penalty,
            flow_penalty=penalty,
        )
        self.cache[key] = row
        self.designs[key] = d.copy()
        # Keep only the currently best field arrays, not thousands of 128-grid fields.
        if not self.fields or ranking(row) < ranking(
            self.cache[next(iter(self.fields))]
        ):
            self.fields = {key: fields}
        return row

    def parameters(self, p, head):
        d = self.decoder.binary(
            torch.tensor(np.asarray(p)[None], dtype=torch.float64), [head]
        )[0].numpy()
        return self.evaluate(d)

    def export(self, folder, row, parameters=None, head=None, **extra):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        key = row["binary_sha256"]
        binary = self.frame.to_original(self.designs[key])
        problem = problem_from_task(self.frame.task, stage5_spec())
        if key not in self.fields:
            p, tiled = refine_problem(problem, self.designs[key], 2)
            _, fields = record(tiled, p)
        else:
            fields = self.fields[key]["nominal"]
        coordinates = (
            np.empty((0,))
            if head is None
            else self.decoder.coordinates(
                torch.tensor(parameters[None], dtype=torch.float64), [head]
            )[0].numpy()
        )
        if self.frame.reflected and head is not None:
            coordinates = coordinates.copy()
            coordinates[..., 1] = 63 - coordinates[..., 1]
        np.savez_compressed(
            folder / "selected.npz",
            binary=binary,
            canonical_binary=self.designs[key],
            fine_binary=np.repeat(np.repeat(binary, 2, 0), 2, 1),
            parameters=np.empty((0,)) if parameters is None else parameters,
            head=-1 if head is None else head,
            coordinates=coordinates,
            temperature=self.frame.to_original(fields.temperature),
            pressure=self.frame.to_original(fields.pressure),
            qx=self.frame.to_original(fields.qx),
            qy=self.frame.to_original(fields.qy, y_velocity=True),
        )
        result = dict(
            selected=row,
            original_binary_sha256=hashlib.sha256(binary.tobytes()).hexdigest(),
            canonical_y_flip=self.frame.reflected,
            selection=(
                "lowest nominal128 silicon temperature, "
                "jointly admissible in both modes"
            ),
            **extra,
        )
        write(folder / "result.json", result)
        return result


def optimize_head(
    task, output, *, kind, head, steps=600, initial=None, evaluator_type=JointEvaluator
):
    """A full budget for each topology, not a weaker single fan reference."""
    worker_init()
    output = Path(output)
    if (output / "result.json").exists():
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        if result.get("objective_evaluations") != steps or result.get("kind") != kind:
            raise ValueError("incompatible partial/reference result")
        return result
    evaluator = evaluator_type(task, output)
    params = (
        LockedDecoder.initial(head) if initial is None else np.asarray(initial).copy()
    )
    p = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    objective = JointObjective(Stage5Solver([evaluator.problem]))
    physical_cache = DesignGradientCache(objective)
    candidates, saved_parameters, history = [], [], []
    started = time.perf_counter()

    def checkpoint(v):
        row = evaluator.parameters(v, head)
        candidates.append(row)
        saved_parameters.append(np.asarray(v).copy())
        return row

    checkpoint(params)
    uniform = evaluator.evaluate(uniform_parallel())

    def calculate(x):
        i = len(history)
        hard = kind == "R25" or i >= steps // 2
        beta = 6.0 if hard else 2 + 4 * i / max(steps // 2 - 1, 1)
        design = evaluator.decoder.density(x[None], [head], beta=beta, hard=hard)[0]
        value, design_gradient, peak = physical_cache.evaluate(
            design, 2 if hard else 1, cache=hard
        )
        gradient = torch.autograd.grad(design, x, design_gradient)[0]
        if not torch.isfinite(gradient).all() or not np.isfinite(value):
            raise RuntimeError("nonfinite optimization objective/gradient")
        history.append(
            dict(
                evaluation=i + 1,
                loss=value,
                hard=hard,
                coarse_peak_c=peak,
            )
        )
        if len(history) % 50 == 0:
            write(
                output / "progress.json",
                dict(
                    kind=kind,
                    head=head,
                    objective_evaluations=len(history),
                    requested=steps,
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
        return value, gradient

    if kind in ("Adam", "R25"):
        opt = torch.optim.Adam([p], lr=0.015)
        current, scale = candidates[0], 1.0
        for i in range(steps):
            opt.param_groups[0]["lr"] = (
                0.015 if kind == "R25" else (0.04 if i < steps // 2 else 0.015)
            ) * scale
            old = p.detach().clone()
            state = copy.deepcopy(opt.state_dict()) if kind == "R25" else None
            opt.zero_grad(set_to_none=True)
            _, p.grad = calculate(p)
            torch.nn.utils.clip_grad_norm_([p], 10.0, error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                p[:25].clamp_(-6, 6)
                p[25:].clamp_(-12, 12)
            if kind == "R25" or (i + 1) % 50 == 0 or i + 1 == steps:
                candidate = checkpoint(p.detach().numpy())
                if kind == "R25":
                    accepted = ranking(candidate) <= ranking(current)
                    history[-1]["accepted"] = accepted
                    if accepted:
                        current = candidate
                    else:
                        with torch.no_grad():
                            p.copy_(old)
                        opt.load_state_dict(state)
                        scale *= 0.5
    elif kind == "MMA":
        import nlopt

        x = params.copy()
        for stop in sorted({steps // 2, steps}):
            while len(history) < stop:
                before = len(history)
                opt = nlopt.opt(nlopt.LD_MMA, len(x))
                bounds = np.r_[np.full(25, 6.0), np.full(4096, 12.0)]
                opt.set_lower_bounds(-bounds)
                opt.set_upper_bounds(bounds)
                opt.set_maxeval(stop - before)
                last_x = x.copy()

                def function(v, grad):
                    nonlocal last_x
                    last_x = v.copy()
                    value, g = calculate(
                        torch.tensor(v, dtype=torch.float64, requires_grad=True)
                    )
                    if grad.size:
                        grad[:] = g.detach().numpy()
                    if len(history) % 50 == 0:
                        checkpoint(v)
                    return value

                opt.set_min_objective(function)
                try:
                    x = opt.optimize(x)
                except nlopt.RoundoffLimited:
                    x = last_x
                if len(history) == before:
                    raise RuntimeError("MMA consumed no evaluations")
            checkpoint(x)
    else:
        raise ValueError("unknown optimizer")
    best = min(range(len(candidates)), key=lambda i: ranking(candidates[i]))
    raw = candidates[best]
    use_fallback = evaluator.frame.self_symmetric or ranking(uniform) < ranking(raw)
    selected = uniform if use_fallback else raw
    if not selected["valid"]:
        raise RuntimeError("no jointly admissible candidate or fallback")
    return evaluator.export(
        output,
        selected,
        None if use_fallback else saved_parameters[best],
        None if use_fallback else head,
        kind=kind,
        raw=raw,
        raw_parameters=saved_parameters[best].tolist(),
        raw_head=head,
        fallback_used=use_fallback,
        objective_evaluations=len(history),
        history=history,
        checkpoints=candidates,
        elapsed_seconds=time.perf_counter() - started,
        independent_thermal_systems=4 * len(evaluator.cache),
        physical_objective_solves=physical_cache.misses,
        exact_physical_cache_hits=physical_cache.hits,
        biased_STE=True,
    )


def evaluate_neural_case(task, predictions, output, *, seed):
    """Four R0 heads, one R25 trajectory, exactly 32 directed search candidates."""
    worker_init()
    output = Path(output)
    if (output / "complete.json").exists():
        completed = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        if set(completed) != {"R0", "R25", "Search32", "Anchors"} or not all(
            value["selected"]["valid"] for value in completed.values()
        ):
            raise ValueError("incomplete or inadmissible saved neural comparison")
        return completed
    evaluator = JointEvaluator(task, output)
    start = time.perf_counter()
    uniform = evaluator.evaluate(uniform_parallel())
    if not uniform["valid"]:
        raise RuntimeError("joint fallback invalid")
    write(output / "uniform.json", uniform)
    methods = {}

    def finish(name, population, rows, objective_evaluations, **extra):
        best = min(range(len(rows)), key=lambda i: ranking(rows[i]))
        p, h = population[best]
        raw = rows[best]
        fallback = evaluator.frame.self_symmetric or ranking(uniform) < ranking(raw)
        selected = uniform if fallback else raw
        methods[name] = evaluator.export(
            output / name,
            selected,
            None if fallback else p,
            None if fallback else h,
            raw=raw,
            fallback_used=fallback,
            objective_evaluations=objective_evaluations,
            logical_candidates=len(rows),
            elapsed_seconds=time.perf_counter() - start,
            **extra,
        )
        return best

    population = [(np.asarray(p).copy(), h) for h, p in enumerate(predictions)]
    rows = [evaluator.parameters(p, h) for p, h in population]
    best = finish("R0", population, rows, 0)
    p, h = population[best]
    refinement = optimize_head(
        task, output / "R25_path", kind="R25", head=h, steps=25, initial=p
    )
    methods["R25"] = refinement
    # Save R25 at the same common path as the other methods.
    import shutil

    (output / "R25").mkdir(exist_ok=True)
    shutil.copyfile(
        output / "R25_path" / "selected.npz", output / "R25" / "selected.npz"
    )
    write(output / "R25" / "result.json", refinement)
    effective_seed = seed ^ int(evaluator.frame.task.fingerprint()[:16], 16)
    rng = np.random.default_rng(effective_seed)
    search_population = list(population)
    search_rows = list(rows)
    seen = {r["binary_sha256"] for r in rows}
    # Four predictions may coincide. The budget counts unique binary maps.
    unique = []
    unique_population = []
    for item, row in zip(search_population, search_rows, strict=True):
        if row["binary_sha256"] not in {r["binary_sha256"] for r in unique}:
            unique.append(row)
            unique_population.append(item)
    search_population, search_rows = unique_population, unique
    attempts = 0
    while len(search_rows) < 32 and attempts < 3200:
        attempts += 1
        elite = sorted(
            range(len(search_rows)), key=lambda i: search_rows[i]["search_objective"]
        )[:4]
        parent = int(rng.choice(elite))
        p, h = search_population[parent]
        p = p.copy()
        b = int(rng.integers(5))
        c = int(rng.integers(5))
        p[b * 4 : b * 4 + 4] += rng.normal(0, float(rng.choice([0.15, 0.35, 0.7])), 4)
        delta = rng.normal(0, 0.45)
        p[20 + b] += delta
        p[20 + c] -= delta
        p[:25] = np.clip(p[:25], -6, 6)
        d = evaluator.decoder.binary(torch.tensor(p[None]), [h])[0].numpy()
        key = hashlib.sha256(d.tobytes()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        search_population.append((p, h))
        search_rows.append(evaluator.evaluate(d))
    if len(search_rows) != 32:
        raise RuntimeError("search did not produce the requested 32 unique designs")
    finish(
        "Search32",
        search_population,
        search_rows,
        0,
        attempts=attempts,
        unique_search_candidates=len(seen),
        requested_candidates=32,
        seed=effective_seed,
    )
    # A separate no-network control quantifies the value of the geometry prior.
    anchors = [(LockedDecoder.initial(h), h) for h in range(4)]
    finish("Anchors", anchors, [evaluator.parameters(p, h) for p, h in anchors], 0)
    write(output / "complete.json", methods)
    return methods
