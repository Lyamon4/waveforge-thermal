"""Fresh Stage3.1 train-only campaign using qualified persistent physics workers."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import digest
from .stage3 import VascularUNet, diversity_loss
from .stage3_training import (
    TrainingStream,
    bridged_losses,
    condition_worker,
    physics_worker,
    write_json,
)
from .stage31 import density, learning_rate, logit_penalty, training_level

CONFIG = {
    "steps": 800,
    "batch": 4,
    "train_seed": 2026101803,
    "model_seed": 2026101802,
    "learning_rate_start": 0.0005,
    "learning_rate_end": 0.0001,
    "softmin_temperature": 0.008,
    "mean_head_weight": 0.03,
    "diversity_weight": 0.02,
    "logit_l2_weight": 1e-4,
    "gradient_clip": 1.0,
}


def train(pool, output, protected, *, steps=800, batch=4):
    if steps < 3 or batch < 1:
        raise ValueError("training needs >=3 steps and positive batch")
    config = dict(CONFIG, steps=steps, batch=batch)
    output = Path(output)
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(config["model_seed"])
    model = VascularUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate_start"])
    stream = TrainingStream(config["train_seed"], protected)
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
            parameters = model(inputs)
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
        baseline_conditioning_solves=steps * batch,
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
