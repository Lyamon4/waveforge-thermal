"""Pre-production checks; no training and no protected workload access."""

import time
from itertools import pairwise

import numpy as np
import torch

from .baselines import uniform_parallel
from .contracts import problem_from_task
from .stage3_training import linear_report, worker_init
from .stage5_decoder import ManifoldUNet, binary, density, fields, initial_parameters
from .stage5_evaluation import evaluate
from .stage5_physics import Stage5Solver, stage5_spec
from .stage5_reference import solve_reference
from .vascular import connectivity


def qualify(task):
    worker_init()
    start = time.perf_counter()
    problem = problem_from_task(task, stage5_spec())
    rng = np.random.default_rng(2026101809)
    geometry = []
    p0 = initial_parameters()
    examples = [
        p0,
        np.r_[
            np.tile([-20.0, 20.0, -20.0, 20.0], 5),
            np.array([-20, 20, -20, 20, -20]),
            np.zeros(4096),
        ],
    ]
    examples += [rng.normal(0, 5, 4121) for _ in range(200)]
    for p in examples:
        tensor = torch.tensor(p[None], dtype=torch.float64)
        _, core, _ = fields(tensor)
        d = binary(tensor)[0].numpy()
        geometry.append(
            dict(
                core_cells=int(core.sum()),
                cells=int(d.sum()),
                connected=connectivity(d)["all_cells_port_connected"],
            )
        )
    forwards = []
    max_difference = 0.0
    max_residual = 0.0
    for name, d in [
        ("Uniform", uniform_parallel()),
        ("Manifold-initial", binary(torch.tensor(p0[None]))[0].numpy()),
    ]:
        row, ref = evaluate(d, problem)
        solver = Stage5Solver([problem])
        out = solver(torch.tensor(d[None], dtype=torch.float64))
        max_difference = max(
            max_difference,
            float(np.max(abs(out.temperature[0].numpy() - ref.temperature))),
        )
        max_residual = max(max_residual, linear_report(solver)["max_linear_residual"])
        # Net flow in five intervals at the middle of the package, not a CFD claim.
        bins = (0, 14, 26, 38, 50, 64)
        row["midplane_net_flow_by_y_band_ml_s"] = [
            float(ref.qx[31, a:b].sum() * 1e6) for a, b in pairwise(bins)
        ]
        row["name"] = name
        forwards.append(row)
    d = rng.uniform(0.2, 0.8, (64, 64))
    variable = torch.tensor(d[None], dtype=torch.float64, requires_grad=True)
    solver = Stage5Solver([problem])
    result = solver(variable)
    active = result.temperature[0, :, 0, solver.chip[0]]
    thermal = torch.logsumexp(5 * active.flatten(), 0) / 5
    grad = torch.autograd.grad(thermal, variable, retain_graph=True)[0][0].numpy()
    flowgrad = torch.autograd.grad(result.flow_ml_s.sum(), variable)[0][0].numpy()
    max_residual = max(max_residual, linear_report(solver)["max_linear_residual"])
    base = solve_reference(d, problem)
    branch = (base.qx, base.qy, base.qout)
    if any(np.any(q == 0) for q in branch):
        raise RuntimeError("ambiguous zero-flow branch")
    shift = float((5 * base.temperature[:, 0, problem.chip_mask]).max())
    checks = []
    for _ in range(8):
        v = rng.normal(size=d.shape)
        v /= np.linalg.norm(v)
        cs = solve_reference(d + 1j * 1e-20 * v, problem, _frozen_flows=branch)
        value = (
            np.log(np.exp(5 * cs.temperature[:, 0, problem.chip_mask] - shift).sum())
            + shift
        ) / 5
        gt = value.imag / 1e-20
        at = float((grad * v).sum())
        gf = cs.flow_ml_s.imag / 1e-20
        af = float((flowgrad * v).sum())
        checks.append(
            dict(
                thermal_error=abs(gt - at) / max(abs(gt), abs(at), 1e-30),
                flow_error=abs(gf - af) / max(abs(gf), abs(af), 1e-30),
            )
        )
    passed = (
        max_difference < 1e-9
        and max_residual < 1e-12
        and all(r["accepted"] for r in forwards)
        and all(
            r["core_cells"] <= 760 and r["cells"] == 1024 and r["connected"]
            for r in geometry
        )
        and all(max(r.values()) < 1e-6 for r in checks)
    )
    return dict(
        stage="5",
        passed=bool(passed),
        max_forward_difference_k=max_difference,
        max_linear_residual=max_residual,
        geometry_cases=len(geometry),
        max_core_cells=max(r["core_cells"] for r in geometry),
        gradient_checks=checks,
        forward_checks=forwards,
        elapsed_seconds=time.perf_counter() - start,
        physical_validation=False,
        closed_test_accessed=False,
        gradient_scope="physical adjoint; binary readout uses a biased STE",
    )


def benchmark_neural():
    """Measure actual network+CPU decoder backward, including device transfers."""
    worker_init()
    timings = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        torch.manual_seed(912)
        model = ManifoldUNet().to(device)
        x = torch.zeros(4, 13, 64, 64)
        samples = []
        for index in range(8):
            start = time.perf_counter()
            model.zero_grad(set_to_none=True)
            p = model(x.to(device)).cpu()
            used, _ = density(p.flatten(0, 1).double(), beta=4.0, hard=False)
            (used * torch.linspace(0, 1, 64)[None, None, :]).mean().backward()
            if device == "cuda":
                torch.cuda.synchronize()
            if index >= 2:
                samples.append(time.perf_counter() - start)
        timings[device] = float(np.median(samples))
        del model
    selected = min(timings, key=timings.get)
    return dict(
        seconds_per_neural_decoder_step=timings,
        selected_neural_device=selected,
        physics_device="cpu",
        note="Sparse LU physics stays on CPU; no GPU-utilization claim.",
    )
