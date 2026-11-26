"""Optional physical fine tuning after supervised refinement distillation."""

import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .adaptive_manifold import contrast_pair, reflected_task
from .amortized import distillation_loss
from .amortized_policy import load_policy
from .amortized_training import _arrays, _save_model, _selection_key, _Validation
from .baselines import digest
from .contracts import problem_from_task
from .joint_design import JointObjective, head_loss
from .locked_channels import canonical_frame
from .locked_evaluation import write
from .ribbed_channels import RibbedDecoder, RibbedGeometry
from .stage3_training import TrainingStream, bridged_losses
from .stage5_physics import Stage5Solver, stage5_spec
from .stage5_training import condition_worker


def physics_labels(task, designs):
    objective = JointObjective(Stage5Solver([problem_from_task(task, stage5_spec())]))
    values, gradients, residual = [], [], 0.0
    for design in designs:
        x = torch.tensor(design, dtype=torch.float64, requires_grad=True)
        loss, _ = objective(x, 2)
        gradient = torch.autograd.grad(loss, x)[0]
        records = (
            objective.solver.linear_diagnostics + objective.probe.linear_diagnostics
        )
        if not records or not all(row["accepted"] for row in records):
            raise RuntimeError("unqualified physical fine-tuning solve")
        if not torch.isfinite(gradient).all() or not torch.isfinite(loss):
            raise RuntimeError("nonfinite physical fine-tuning derivative")
        residual = max(
            residual,
            *(
                max(row["relative_residual"], row["compensated_relative_residual"])
                for row in records
            ),
        )
        values.append(loss.item())
        gradients.append(gradient.numpy())
    return np.array(values), np.stack(gradients), residual


def polish(root, groups, config, protected, *, steps=150, device="cuda"):
    root = Path(root)
    source = root / "training/model_frozen.pt"
    output = root / "polish"
    output.mkdir(exist_ok=True)
    source_sha = digest(source)
    specification = dict(
        config,
        source_sha256=source_sha,
        physical_updates=steps,
        physical_batch=4,
        objective="unchanged JointObjective level2 and head_loss",
        distillation_replay_weight=0.01,
    )
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != specification:
            raise ValueError("different physical fine-tuning configuration")
    else:
        write(output / "config.json", specification)
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    if (output / "stopped.json").exists():
        return json.loads((output / "stopped.json").read_text())
    model = load_policy(source, device).train().requires_grad_(True)
    training, _ = _arrays(root, "train", groups["train"])
    validation, metadata = _arrays(root, "validation", groups["validation"])
    monitor = _Validation(output, groups["validation"], validation, metadata, device)
    original_summary, _ = monitor.evaluate(model)
    best_summary, best_step = original_summary, 0
    _save_model(output / "model_best.pt", model, 0, specification, best_summary)
    replay_x = torch.from_numpy(training["condition"]).to(device)
    replay_y = torch.from_numpy(training["targets"].astype(np.float32)).to(device)
    replay_w = torch.from_numpy(training["weights"]).to(device)
    body, heads = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (
                heads if name.startswith(("geometry.", "pixels.", "delta_")) else body
            ).append(parameter)
    optimizer = torch.optim.Adam(
        [dict(params=body, lr=1e-5), dict(params=heads, lr=5e-5)]
    )
    forbidden = tuple(protected) + tuple(groups["validation"])
    stream = TrainingStream(
        config["seed"] + 901, forbidden + tuple(reflected_task(t) for t in forbidden)
    )
    excluded = {t.fingerprint() for t in groups["train"] + groups["validation"]}
    stream.seen.update(excluded)
    pair_rng = np.random.default_rng(config["seed"] + 902)
    replay_rng = torch.Generator().manual_seed(config["seed"] + 903)
    start = time.perf_counter()
    maximum_residual = 0.0
    with (
        ThreadPoolExecutor(max_workers=3) as pool,
        (output / "history.jsonl").open("w", encoding="utf-8") as history,
        (output / "tasks.jsonl").open("w", encoding="utf-8") as tasks_log,
    ):
        for step in range(steps):
            first = stream.next()
            tasks = [
                canonical_frame(t).task
                for t in (*contrast_pair(first, pair_rng), stream.next(), stream.next())
            ]
            if any(
                stream.protected.contains(t) or t.fingerprint() in excluded
                for t in tasks
            ):
                raise RuntimeError("fresh physical training hit a protected task")
            for task in tasks:
                tasks_log.write(
                    json.dumps(
                        dict(
                            step=step + 1,
                            task=task.canonical(),
                            sha256=task.fingerprint(),
                        )
                    )
                    + "\n"
                )
            tasks_log.flush()
            conditions = [condition_worker(t)[0] for t in tasks]
            model.train().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            parameters = model(
                torch.from_numpy(np.stack(conditions)).to(device)
            ).double()
            geometries = [RibbedGeometry.from_task(t) for t in tasks]
            decoder = RibbedDecoder([g for g in geometries for _ in range(4)])
            designs = decoder.density(
                parameters.flatten(0, 1),
                list(range(4)) * 4,
                beta=6.0,
                hard=True,
                physical=getattr(model, "prediction_space", "raw") == "physical",
            ).reshape(4, 4, 64, 64)
            arrays = designs.detach().cpu().numpy()
            futures = [
                pool.submit(physics_labels, task, d)
                for task, d in zip(tasks, arrays, strict=True)
            ]
            indices = torch.randint(len(replay_x), (8,), generator=replay_rng).to(
                device
            )
            replay = distillation_loss(
                model(replay_x[indices]),
                replay_y[indices],
                replay_w[indices],
                prediction_space=getattr(model, "prediction_space", "raw"),
            )
            results = [future.result() for future in futures]
            losses = bridged_losses(
                designs,
                np.stack([v[0] for v in results]),
                np.stack([v[1] for v in results]),
            )
            loss = head_loss(losses) + 0.01 * replay
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            rate = 0.2 + 0.4 * (1 + math.cos(math.pi * step / max(steps - 1, 1)))
            for group, initial in zip(
                optimizer.param_groups, (1e-5, 5e-5), strict=True
            ):
                group["lr"] = initial * rate
            optimizer.step()
            maximum_residual = max(maximum_residual, *(v[2] for v in results))
            record = dict(
                step=step + 1,
                loss=loss.item(),
                gradient_norm=float(norm),
                maximum_residual=maximum_residual,
                elapsed_seconds=time.perf_counter() - start,
            )
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
            write(root / "progress.json", dict(phase="physical_fine_tuning", **record))
            if step == 0 or (step + 1) % 10 == 0:
                print(f"POLISH {step + 1}/{steps} loss={loss.item():.5f}", flush=True)
            if (step + 1) % 25 == 0 or step + 1 == steps:
                summary, _ = monitor.evaluate(model)
                write(output / f"validation_{step + 1:04d}.json", summary)
                print(
                    f"POLISH_VALIDATION {step + 1} "
                    f"gap={summary['median_gap_to_teacher_percent']:.3f}%",
                    flush=True,
                )
                if _selection_key(summary, step + 1) < _selection_key(
                    best_summary, best_step
                ):
                    best_summary, best_step = summary, step + 1
                    _save_model(
                        output / "model_best.pt",
                        model,
                        best_step,
                        specification,
                        best_summary,
                    )
                    write(
                        output / "best.json", dict(step=best_step, summary=best_summary)
                    )
    model = load_policy(output / "model_best.pt", device)
    summary, selected = monitor.evaluate(model)
    monitor.export(selected, output / "validation_designs")
    shutil.copy2(output / "model_best.pt", output / "model_frozen.pt")
    if digest(source) != source_sha:
        raise RuntimeError("warm-start checkpoint changed")
    result = dict(
        physical_updates=steps,
        selected_step=best_step,
        validation=summary,
        starting_validation=original_summary,
        source_unchanged=True,
        maximum_residual=maximum_residual,
        model_sha256=digest(output / "model_frozen.pt"),
        closed_test_opened=False,
        elapsed_seconds=time.perf_counter() - start,
    )
    write(output / "complete.json", result)
    return result
