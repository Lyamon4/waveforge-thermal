"""Fixed conditioning and train-only physical gradients for Stage 3."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from waveforge.lab.contracts import Chiplet, ChipletTask, content_hash
from waveforge.lab.epyc_tasks import GeometryIndex, sample_epyc_task

from .baselines import digest, uniform_parallel
from .contracts import problem_from_task, stage12_spec
from .differentiable import MCLSolver
from .stage3 import VascularUNet, continuous, diversity_loss

NORMALIZATION = {
    "layout": "channel,x,y; 64x64",
    "baseline": "frozen Stage2 Uniform-Parallel",
    "channels": [
        "q1_W_per_cell/1W",
        "q2_W_per_cell/1W",
        "q3_W_per_cell/1W",
        "theta1_Si_K/150K",
        "theta2_Si_K/150K",
        "theta3_Si_K/150K",
        "sensitivity1",
        "sensitivity2",
        "sensitivity3",
        "chip_mask",
        "boundary_port_mask",
        "X/63",
        "Y/63",
    ],
    "sensitivity": {
        "objective": "log-mean-exp(250*theta_Si/100)/250 per scenario",
        "normalization": "clip(gradient/max(RMS(gradient),1e-12),-5,5)/5",
        "core": "no core masking in baseline sensitivity",
    },
}


def write_json(path, data, *, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def worker_init():
    torch.set_num_threads(1)


def task_from_dict(row):
    return ChipletTask(tuple(Chiplet(**c) for c in row["chiplets"]), row["powers_w"])


def protected_geometries(registry, extra_tasks=()):
    """Only read the sealed index and its manifest, never split workload files."""
    registry = Path(registry)
    manifest = json.loads((registry / "manifest.json").read_text(encoding="utf-8"))
    geometry_path = registry / "geometry_exclusions.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if (
        geometry.get("schema") != "epyc-v2-geometry-index-v1"
        or content_hash(geometry) != manifest["geometry_sha256"]
    ):
        raise ValueError("geometry-only exclusion index seal mismatch")
    rows = list(geometry["retrospective_v1"])
    for split, expected_count in manifest["counts"].items():
        entries = geometry["v2"][split]
        if len(entries) != expected_count:
            raise ValueError("protected geometry count mismatch")
        rows.extend(entries)
    tasks = [
        ChipletTask(tuple(Chiplet(**c) for c in row), ((0.0,) * len(row),))
        for row in rows
    ]
    tasks.extend(extra_tasks)
    return tuple(tasks), {
        "geometry_file_sha256": digest(geometry_path),
        "geometry_content_sha256": content_hash(geometry),
        "registry_manifest_sha256": digest(registry / "manifest.json"),
        "protected_count": len(tasks),
        "near_duplicate_radius_cells": 2,
        "closed_workloads_read": False,
        "proof": "No protected geometry match implies no protected task hash match.",
    }


class TrainingStream:
    def __init__(self, seed, protected):
        self.rng = np.random.default_rng(seed)
        self.protected = GeometryIndex(tuple(protected))
        self.seen = set()
        self.rejected = 0
        self.attempted = 0

    def next(self):
        for _ in range(100000):
            task = sample_epyc_task(self.rng)
            self.attempted += 1
            fingerprint = task.fingerprint()
            if fingerprint in self.seen or self.protected.contains(task):
                self.rejected += 1
                continue
            self.seen.add(fingerprint)
            return task
        raise RuntimeError("train-only sampling exhausted; exclusions not relaxed")


def linear_report(solver):
    rows = solver.linear_diagnostics
    if not rows or not all(r["accepted"] for r in rows):
        raise RuntimeError("unqualified forward/adjoint solve")
    return {
        "max_linear_residual": max(
            max(r["relative_residual"], r["compensated_relative_residual"])
            for r in rows
        ),
        "linear_solves": len(rows),
    }


def condition_worker(task):
    worker_init()
    problem = problem_from_task(task, stage12_spec())
    solver = MCLSolver([problem])
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
    return channels, linear_report(solver)


def physical_loss(design, solver, level):
    result = solver(design[None])
    active = result.temperature[0, :, 0, solver.chip[0]].flatten() / 100
    alpha = (100.0, 250.0, 500.0)[level]
    thermal = (torch.logsumexp(alpha * active, 0) - np.log(active.numel())) / alpha
    tv = (design[1:] - design[:-1]).abs().mean() + (
        design[:, 1:] - design[:, :-1]
    ).abs().mean()
    loss = (
        thermal + 0.001 * tv + (0.0, 0.01, 0.02)[level] * (design * (1 - design)).mean()
    )
    return loss, result


def physics_worker(task, densities, level):
    worker_init()
    solver = MCLSolver([problem_from_task(task, stage12_spec())])
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
        reports.append(linear_report(solver))
    return (
        np.array(losses),
        np.stack(gradients),
        {
            "max_linear_residual": max(r["max_linear_residual"] for r in reports),
            "continuous_peaks_k": peaks,
            "linear_solves": sum(r["linear_solves"] for r in reports),
        },
    )


def bridged_losses(densities, losses, gradients):
    """Exact vector-Jacobian bridge from persistent CPU workers into the network."""
    values = torch.as_tensor(losses, dtype=densities.dtype, device=densities.device)
    grads = torch.as_tensor(gradients, dtype=densities.dtype, device=densities.device)
    return values + ((densities - densities.detach()) * grads).sum((-2, -1))


def train_model(pool, output, protected, config):
    output = Path(output)
    start = time.perf_counter()
    torch.manual_seed(config["model_seed"])
    torch.set_num_threads(1)
    model = VascularUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    stream = TrainingStream(config["train_seed"], protected)
    steps, batch = config["steps"], config["batch"]
    max_residual = 0.0
    with (
        (output / "training_tasks.jsonl").open("x", encoding="utf-8") as tasks_log,
        (output / "training_history.jsonl").open("x", encoding="utf-8") as history,
    ):
        for step in range(steps):
            tasks = [stream.next() for _ in range(batch)]
            for task in tasks:
                tasks_log.write(
                    json.dumps(
                        {
                            "step": step + 1,
                            "sha256": task.fingerprint(),
                            "task": task.canonical(),
                        }
                    )
                    + "\n"
                )
            tasks_log.flush()
            conditioned = list(pool.map(condition_worker, tasks))
            inputs = torch.from_numpy(np.stack([r[0] for r in conditioned]))
            optimizer.zero_grad(set_to_none=True)
            parameters = model(inputs)
            level = min(2, step * 3 // steps)
            density = continuous(
                parameters.flatten(0, 1).double(), (2.0, 4.0, 8.0)[level]
            ).reshape(batch, 4, 64, 64)
            futures = [
                pool.submit(physics_worker, task, data, level)
                for task, data in zip(tasks, density.detach().numpy(), strict=True)
            ]
            results = [future.result() for future in futures]
            losses = bridged_losses(
                density,
                np.stack([r[0] for r in results]),
                np.stack([r[1] for r in results]),
            )
            temperature = config["softmin_temperature"]
            softmin = -temperature * (
                torch.logsumexp(-losses / temperature, 1) - np.log(4)
            )
            diversity = diversity_loss(parameters.double())
            loss = (
                softmin.mean()
                + config["mean_head_weight"] * losses.mean()
                + (config["diversity_weight"] * diversity)
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            max_residual = max(
                max_residual,
                *(r[1]["max_linear_residual"] for r in conditioned),
                *(r[2]["max_linear_residual"] for r in results),
            )
            row = {
                "step": step + 1,
                "phase": level,
                "loss": loss.item(),
                "head_losses": losses.detach().mean(0).tolist(),
                "diversity": diversity.item(),
                "grad_norm": float(grad_norm),
                "max_linear_residual": max_residual,
                "unique_training_tasks": len(stream.seen),
                "rejected_training_samples": stream.rejected,
                "elapsed_seconds": time.perf_counter() - start,
            }
            history.write(json.dumps(row, allow_nan=False) + "\n")
            history.flush()
            if (step + 1) % 5 == 0 or step == 0 or step + 1 == steps:
                write_json(output / "progress.json", row, exclusive=False)
                print(
                    f"train {step + 1}/{steps}: loss={loss.item():.5f}, "
                    f"{row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            if (step + 1) % 100 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step + 1,
                        "torch_rng": torch.get_rng_state(),
                        "train_rng": stream.rng.bit_generator.state,
                    },
                    output / f"training_step_{step + 1:04d}.pt",
                )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen = output / "model_frozen.pt"
    torch.save({"model": model.state_dict(), "steps": steps, "config": config}, frozen)
    report = {
        "optimizer_updates": steps,
        "unique_training_tasks": len(stream.seen),
        "head_physical_objectives": steps * batch * 4,
        "baseline_conditioning_solves": steps * batch,
        "sampling_attempted": stream.attempted,
        "sampling_rejected": stream.rejected,
        "model_sha256": digest(frozen),
        "training_tasks_sha256": digest(tasks_log.name),
        "elapsed_seconds": time.perf_counter() - start,
        "max_linear_residual": max_residual,
        "development_used_for_training_or_checkpoint_selection": False,
    }
    write_json(output / "freeze.json", report)
    return model, report
