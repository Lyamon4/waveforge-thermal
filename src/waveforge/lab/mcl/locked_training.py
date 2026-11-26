"""Local GPU network training with qualified FP64 physics in persistent workers."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .adaptive_manifold import contrast_pair, reflected_task
from .baselines import digest
from .contracts import problem_from_task
from .joint_design import JointObjective, LockedUNet, head_loss
from .locked_channels import LockedDecoder, LockedGeometry, canonical_frame
from .locked_evaluation import write
from .stage3_training import TrainingStream, bridged_losses, linear_report, worker_init
from .stage5_physics import Stage5Solver, stage5_spec
from .stage5_training import condition_worker


def physics_worker(task, designs, level):
    worker_init()
    objective = JointObjective(Stage5Solver([problem_from_task(task, stage5_spec())]))
    values, gradients, residuals = [], [], []
    for d in designs:
        x = torch.tensor(d, dtype=torch.float64, requires_grad=True)
        value, _ = objective(x, level)
        gradient = torch.autograd.grad(value, x)[0]
        if not torch.isfinite(gradient).all() or not torch.isfinite(value):
            raise RuntimeError("nonfinite physical gradient/loss")
        residuals.append(linear_report(objective.solver)["max_linear_residual"])
        if not all(v["accepted"] for v in objective.probe.linear_diagnostics):
            raise RuntimeError("unqualified fine pressure adjoint")
        residuals.extend(
            max(v["relative_residual"], v["compensated_relative_residual"])
            for v in objective.probe.linear_diagnostics
        )
        values.append(value.item())
        gradients.append(gradient.detach().numpy())
    return np.asarray(values), np.stack(gradients), max(residuals)


def train(
    pool,
    output,
    protected,
    source,
    *,
    steps=300,
    seed=202610295401,
    device="cuda",
    model_type=LockedUNet,
    geometry_type=LockedGeometry,
    decoder_type=LockedDecoder,
    architecture="LockedUNet",
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = model_type().to(device)
    # Retain only generic feature extraction; routing actions have changed meaning.
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    inherited = {
        k: v
        for k, v in checkpoint["model"].items()
        if not k.startswith(("geometry.", "pixels.", "anchors", "width_prior"))
    }
    missing, unexpected = model.load_state_dict(inherited, strict=False)
    if unexpected or set(missing) != {
        "width_prior",
        "geometry.weight",
        "geometry.bias",
        "pixels.weight",
        "pixels.bias",
    }:
        raise RuntimeError(f"incompatible source backbone: {missing}, {unexpected}")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    stream = TrainingStream(
        seed + 1, tuple(protected) + tuple(reflected_task(t) for t in protected)
    )
    rng = np.random.default_rng(seed + 2)
    resume = output / "latest.pt"
    start_step = 0
    elapsed_before = 0.0
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        stream.rng.bit_generator.state = state["stream_rng"]
        stream.seen = set(state["seen"])
        rng.bit_generator.state = state["pair_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        start_step = state["step"]
        elapsed_before = state["elapsed_seconds"]
        for name in ("history.jsonl", "tasks.jsonl"):
            path = output / name
            lines = (
                path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            )
            committed = [
                line for line in lines if json.loads(line)["step"] <= start_step
            ]
            path.write_text(
                "".join(line + "\n" for line in committed), encoding="utf-8"
            )
    started = time.perf_counter()
    with (
        (output / "history.jsonl").open("a", encoding="utf-8") as log,
        (output / "tasks.jsonl").open("a", encoding="utf-8") as task_log,
    ):
        for step in range(start_step, steps):
            first = stream.next()
            pair = contrast_pair(first, rng)
            raw_tasks = [*pair, stream.next(), stream.next()]
            tasks = [canonical_frame(t).task for t in raw_tasks]
            if any(stream.protected.contains(t) for t in tasks):
                raise RuntimeError(
                    "canonical training geometry overlaps protected geometry"
                )
            for task in tasks:
                task_log.write(
                    json.dumps(
                        dict(
                            step=step + 1,
                            sha256=task.fingerprint(),
                            task=task.canonical(),
                        )
                    )
                    + "\n"
                )
            task_log.flush()
            conditions = list(pool.map(condition_worker, tasks))
            inputs = torch.from_numpy(np.stack([c[0] for c in conditions])).to(device)
            optimizer.zero_grad(set_to_none=True)
            parameters = model(inputs).cpu().double()
            geometries = (
                [LockedGeometry.from_mask(c[0][9] > 0.5) for c in conditions]
                if geometry_type is LockedGeometry
                else [geometry_type.from_task(task) for task in tasks]
            )
            decoder = decoder_type([g for g in geometries for _ in range(4)])
            hard = step >= steps // 2
            beta = 6.0 if hard else 2 + 4 * step / max(steps // 2 - 1, 1)
            designs = decoder.density(
                parameters.flatten(0, 1), list(range(4)) * 4, beta=beta, hard=hard
            ).reshape(4, 4, 64, 64)
            futures = [
                pool.submit(physics_worker, t, d, 2 if hard else 1)
                for t, d in zip(tasks, designs.detach().numpy(), strict=True)
            ]
            results = [f.result() for f in futures]
            losses = bridged_losses(
                designs,
                np.stack([r[0] for r in results]),
                np.stack([r[1] for r in results]),
            )
            loss = head_loss(losses)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            lr = 5e-4 + (1e-4 - 5e-4) * step / max(steps - 1, 1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            elapsed = elapsed_before + time.perf_counter() - started
            row = dict(
                step=step + 1,
                hard=hard,
                beta=beta,
                loss=loss.item(),
                head_losses=losses.detach().mean(0).tolist(),
                gradient_norm=float(norm),
                max_linear_residual=max(r[2] for r in results),
                elapsed_seconds=elapsed,
            )
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            write(output / "progress.json", row)
            if (step + 1) % 10 == 0 or step == 0:
                print(
                    f"TRAIN {step + 1}/{steps} loss={loss.item():.5f} "
                    f"hard={hard} seconds={elapsed:.1f}",
                    flush=True,
                )
            if (step + 1) % 10 == 0:
                state = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step + 1,
                    stream_rng=stream.rng.bit_generator.state,
                    pair_rng=rng.bit_generator.state,
                    seen=list(stream.seen),
                    torch_rng=torch.get_rng_state(),
                    elapsed_seconds=elapsed,
                )
                torch.save(state, output / "latest.tmp")
                (output / "latest.tmp").replace(resume)
            if (step + 1) % 100 == 0:
                torch.save(
                    dict(model=model.state_dict(), step=step + 1),
                    output / f"model_{step + 1:04d}.pt",
                )
    model.eval().requires_grad_(False)
    torch.save(
        dict(
            model=model.state_dict(),
            steps=steps,
            architecture=architecture,
            seed=seed,
            source_checkpoint=str(source),
            source_sha256=digest(source),
            physics_profile="unchanged-stage5",
            canonical_y_flip=True,
            decoder=decoder_type.__name__,
            batch_size=4,
        ),
        output / "model_frozen.pt",
    )
    write(
        output / "complete.json",
        dict(
            steps=steps,
            task_exposures=steps * 4,
            elapsed_seconds=elapsed_before + time.perf_counter() - started,
            model_sha256=digest(output / "model_frozen.pt"),
            closed_test_opened=False,
        ),
    )
    return model
