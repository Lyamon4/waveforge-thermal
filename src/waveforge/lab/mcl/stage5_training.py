"""Fresh Stage5 train-only campaign using qualified persistent physics workers."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import digest, uniform_parallel
from .contracts import problem_from_task
from .stage3_training import (
    TrainingStream,
    bridged_losses,
    linear_report,
    physical_loss,
    worker_init,
    write_json,
)
from .stage4_refinement import count_solves
from .stage5_decoder import ManifoldUNet, density, diversity_loss, logit_penalty
from .stage5_physics import Stage5Solver, stage5_spec
from .stage31 import learning_rate, training_level

CONFIG = {
    "steps": 800,
    "batch": 4,
    "train_seed": 2026101807,
    "model_seed": 2026101806,
    "learning_rate_start": 0.0005,
    "learning_rate_end": 0.0001,
    "softmin_temperature": 0.008,
    "mean_head_weight": 0.03,
    "diversity_weight": 0.02,
    "logit_l2_weight": 1e-4,
    "gradient_clip": 1.0,
}


def train(pool, output, protected, *, steps=800, batch=4, device="cpu"):
    if steps < 3 or batch < 1:
        raise ValueError("training needs >=3 steps and positive batch")
    config = dict(
        CONFIG, steps=steps, batch=batch, neural_device=device, physics_device="cpu"
    )
    output = Path(output)
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(config["model_seed"])
    model = ManifoldUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate_start"])
    stream = TrainingStream(config["train_seed"], protected)
    max_residual = 0.0
    condition_counts = count_solves([])
    objective_counts = count_solves([])
    with (
        (output / "training_tasks.jsonl").open("x", encoding="utf-8") as tasks_log,
        (output / "training_history.jsonl").open("x", encoding="utf-8") as history,
    ):
        for step in range(steps):
            tasks = [stream.next() for _ in range(batch)]
            for task in tasks:
                tasks_log.write(
                    json.dumps(
                        dict(
                            step=step + 1,
                            sha256=task.fingerprint(),
                            task=task.canonical(),
                        )
                    )
                    + "\n"
                )
            tasks_log.flush()
            conditions = list(pool.map(condition_worker, tasks))
            inputs = torch.from_numpy(np.stack([c[0] for c in conditions]))
            lr = learning_rate(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            parameters = model(inputs.to(device)).cpu()
            level = training_level(step, steps)
            used, _smooth = density(
                parameters.flatten(0, 1).double(),
                beta=(2.0, 4.0, 8.0)[level],
                hard=level == 2,
            )
            used = used.reshape(batch, 4, 64, 64)
            futures = [
                pool.submit(physics_worker, t, d, level)
                for t, d in zip(tasks, used.detach().numpy(), strict=True)
            ]
            results = [f.result() for f in futures]
            for _, meta in conditions:
                for key, value in meta["solver_counts"].items():
                    condition_counts[key] += value
            for _, _, meta in results:
                for key, value in meta["solver_counts"].items():
                    objective_counts[key] += value
            losses = bridged_losses(
                used,
                np.stack([r[0] for r in results]),
                np.stack([r[1] for r in results]),
            )
            tau = config["softmin_temperature"]
            softmin = -tau * (torch.logsumexp(-losses / tau, 1) - np.log(4))
            diversity = diversity_loss(parameters.double())
            regularizer = logit_penalty(parameters)
            loss = (
                softmin.mean()
                + config["mean_head_weight"] * losses.mean()
                + config["diversity_weight"] * diversity
                + config["logit_l2_weight"] * regularizer
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            max_residual = max(
                max_residual,
                *(c[1]["max_linear_residual"] for c in conditions),
                *(r[2]["max_linear_residual"] for r in results),
            )
            row = dict(
                step=step + 1,
                phase=level,
                hard_forward=level == 2,
                learning_rate=lr,
                loss=loss.item(),
                head_losses=losses.detach().mean(0).tolist(),
                logit_l2=regularizer.item(),
                diversity=diversity.item(),
                grad_norm=float(grad_norm),
                max_linear_residual=max_residual,
                unique_training_tasks=len(stream.seen),
                rejected_training_samples=stream.rejected,
                density_budget_max_error=float(
                    (used.detach().sum((-2, -1)) - 1024).abs().max()
                ),
                elapsed_seconds=time.perf_counter() - start,
            )
            history.write(json.dumps(row, allow_nan=False) + "\n")
            history.flush()
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
                write_json(output / "progress.json", row, exclusive=False)
                print(
                    f"train {step + 1}/{steps} phase={level} loss={loss.item():.5f} "
                    f"elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            if (step + 1) % 100 == 0:
                torch.save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        step=step + 1,
                        config=config,
                        torch_rng=torch.get_rng_state(),
                        train_rng=stream.rng.bit_generator.state,
                    ),
                    output / f"training_step_{step + 1:04d}.pt",
                )
    model.eval()
    model.requires_grad_(False)
    frozen = output / "model_frozen.pt"
    torch.save(dict(model=model.state_dict(), steps=steps, config=config), frozen)
    report = dict(
        optimizer_updates=steps,
        unique_training_tasks=len(stream.seen),
        head_physical_objectives=steps * batch * 4,
        baseline_conditioning_tasks=steps * batch,
        conditioning_solver_counts=condition_counts,
        objective_solver_counts=objective_counts,
        sampling_attempted=stream.attempted,
        sampling_rejected=stream.rejected,
        model_sha256=digest(frozen),
        training_tasks_sha256=digest(output / "training_tasks.jsonl"),
        elapsed_seconds=time.perf_counter() - start,
        max_linear_residual=max_residual,
        development_used_for_training_or_checkpoint_selection=False,
    )
    write_json(output / "freeze.json", report)
    return model, report


def condition_worker(task):
    worker_init()
    problem = problem_from_task(task, stage5_spec())
    solver = Stage5Solver([problem])
    design = torch.tensor(
        uniform_parallel()[None], dtype=torch.float64, requires_grad=True
    )
    result = solver(design)
    theta = result.temperature[0, :, 0]
    gradients = []
    for scenario in range(3):
        active = theta[scenario, solver.chip[0]] / 100
        value = (torch.logsumexp(250 * active, 0) - np.log(active.numel())) / 250
        gradient = torch.autograd.grad(value, design, retain_graph=scenario < 2)[0][0]
        rms = gradient.square().mean().sqrt().clamp_min(1e-12)
        gradients.append((gradient / rms).clamp(-5, 5) / 5)
    ports = torch.zeros(64, 64, dtype=torch.float64)
    ports[[0, -1], problem.spec.port_start : problem.spec.port_stop] = 1
    x, y = torch.meshgrid(torch.arange(64) / 63, torch.arange(64) / 63, indexing="ij")
    channels = (
        torch.cat(
            (
                solver.power[0],
                theta.detach() / 150,
                torch.stack(gradients),
                solver.chip.double(),
                ports[None],
                x[None],
                y[None],
            ),
            0,
        )
        .float()
        .numpy()
    )
    if not np.isfinite(channels).all():
        raise RuntimeError("nonfinite conditioning")
    return channels, dict(
        linear_report(solver), solver_counts=count_solves(solver.linear_diagnostics)
    )


def physics_worker(task, densities, level):
    worker_init()
    solver = Stage5Solver([problem_from_task(task, stage5_spec())])
    losses, gradients, peaks, reports = [], [], [], []
    for item in densities:
        design = torch.tensor(item, dtype=torch.float64, requires_grad=True)
        loss, result = physical_loss(design, solver, level)
        gradient = torch.autograd.grad(loss, design)[0]
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise RuntimeError("nonfinite physical training loss/gradient")
        losses.append(loss.item())
        gradients.append(gradient.detach().numpy())
        peaks.append(result.scenario_peaks.detach().max().item())
        reports.append(
            dict(
                linear_report(solver),
                solver_counts=count_solves(solver.linear_diagnostics),
            )
        )
    return (
        np.array(losses),
        np.stack(gradients),
        {
            "max_linear_residual": max(r["max_linear_residual"] for r in reports),
            "forward_peaks_k": peaks,
            "linear_solves": sum(r["linear_solves"] for r in reports),
            "solver_counts": {
                key: sum(r["solver_counts"][key] for r in reports)
                for key in count_solves([])
            },
        },
    )
