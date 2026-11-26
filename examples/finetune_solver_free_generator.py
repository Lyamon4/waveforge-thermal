"""Improve raw conditioning from new train-only layouts, preserving old weights."""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from waveforge.lab.mcl.adaptive_manifold import reflected_task
from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.locked_channels import canonical_frame
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.solver_free import raw_features
from waveforge.lab.mcl.solver_free_data import arrays
from waveforge.lab.mcl.solver_free_training import (
    field_loss,
    load_model,
    parameter_loss,
    save_model,
)
from waveforge.lab.mcl.stage3_training import (
    TrainingStream,
    protected_geometries,
    task_from_dict,
    worker_init,
)
from waveforge.lab.mcl.stage5_training import condition_worker


def main(args):
    worker_init()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    groups = {
        k: [task_from_dict(t) for t in v]
        for k, v in json.loads((args.run / "tasks.json").read_text()).items()
    }
    known = [t for v in groups.values() for t in v]
    for path in args.exclude:
        saved = json.loads(path.read_text())
        for group in saved.values() if isinstance(saved, dict) else [saved]:
            if isinstance(group, list):
                known.extend(
                    task_from_dict(t)
                    for t in group
                    if isinstance(t, dict) and "chiplets" in t
                )
    protected, hygiene = protected_geometries(args.registry, known)
    stream = TrainingStream(
        2026103087, protected + tuple(reflected_task(t) for t in protected)
    )
    tasks = [canonical_frame(stream.next()).task for _ in range(args.extra)]
    write(
        args.output / "tasks.json",
        dict(
            original_train=[t.canonical() for t in groups["train"]],
            extra_train=[t.canonical() for t in tasks],
            validation=[t.canonical() for t in groups["validation"]],
        ),
    )
    model = load_model(args.run / "generator/model_frozen.pt", "cuda")
    original = digest(args.run / "generator/model_frozen.pt")
    directory = args.output / "extra_data"
    directory.mkdir()

    def collect(index):
        task = tasks[index]
        condition, _ = condition_worker(task)
        with torch.inference_mode():
            q = (
                model.teacher(torch.from_numpy(condition[None]).cuda())[0]
                .float()
                .cpu()
                .numpy()
            )
        values = dict(raw=raw_features(task), condition=condition, parameters=q)
        np.savez_compressed(directory / f"case_{index:04d}.npz", **values)
        return index, values

    extra = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for done, future in enumerate(
            as_completed([pool.submit(collect, i) for i in range(len(tasks))]), 1
        ):
            index, values = future.result()
            extra[index] = values
            if done % 64 == 0:
                print(
                    f"EXTRA {done}/{len(tasks)} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    base = arrays(args.run, "train", groups["train"])
    val = arrays(args.run, "validation", groups["validation"])
    data = {
        k: np.concatenate((base[k], np.stack([r[k] for r in extra])))
        for k in ("raw", "condition", "parameters")
    }
    x = torch.from_numpy(data["raw"]).cuda()
    y = torch.from_numpy(data["condition"][:, 3:9]).cuda()
    q = torch.from_numpy(data["parameters"].astype(np.float32)).cuda()
    vx = torch.from_numpy(val["raw"]).cuda()
    vy = torch.from_numpy(val["condition"][:, 3:9]).cuda()
    vq = torch.from_numpy(val["parameters"].astype(np.float32)).cuda()
    model.adapter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(), lr=2e-4, weight_decay=1e-5
    )
    rng = torch.Generator().manual_seed(2026103088)
    config = dict(
        source_model_sha256=original,
        geometry_hygiene=hygiene,
        extra_training_layouts=len(tasks),
        original_training_layouts=len(groups["train"]),
        updates=args.updates,
        field_weight=0.1,
        parameter_weight=1.0,
        physics_unchanged=True,
        heldout_audit_tasks_excluded=True,
    )
    write(args.output / "config.json", config)
    best = None
    history = []
    for step in range(args.updates):
        index = torch.randint(len(x), (16,), generator=rng).cuda()
        optimizer.param_groups[0]["lr"] = 2e-5 + 1.8e-4 * 0.5 * (
            1 + math.cos(math.pi * step / max(args.updates - 1, 1))
        )
        optimizer.zero_grad(set_to_none=True)
        condition, _ = model.adapter(x[index])
        predicted = model.teacher(condition)
        loss = 0.1 * field_loss(
            condition[:, 3:9], y[index], x[index, 3:4]
        ) + parameter_loss(predicted, q[index])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.adapter.parameters(), 1, error_if_nonfinite=True
        )
        optimizer.step()
        if (step + 1) % 500 == 0:
            scores = []
            with torch.inference_mode():
                for i in range(len(vx)):
                    condition, _ = model.adapter(vx[i : i + 1])
                    p = parameter_loss(model.teacher(condition), vq[i : i + 1])
                    f = field_loss(condition[:, 3:9], vy[i : i + 1], vx[i : i + 1, 3:4])
                    scores.append((float(p), float(f)))
            p, f = np.mean(scores, axis=0)
            record = dict(step=step + 1, parameter_loss=float(p), field_loss=float(f))
            history.append(record)
            save_model(
                args.output / f"step_{step + 1:04d}.pt",
                model,
                config,
                validation=record,
            )
            if best is None or p < best["parameter_loss"]:
                best = record
                save_model(
                    args.output / "model_frozen.pt", model, config, validation=record
                )
            write(args.output / "history.json", history)
            print(
                f"REFIT {step + 1} parameter_loss={p:.6f} field_loss={f:.6f}",
                flush=True,
            )
    if digest(args.run / "generator/model_frozen.pt") != original:
        raise RuntimeError("old generator checkpoint changed")
    write(
        args.output / "complete.json",
        dict(
            selected=best,
            elapsed_seconds=time.perf_counter() - started,
            old_checkpoint_unchanged=True,
            model_sha256=digest(args.output / "model_frozen.pt"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--exclude", type=Path, nargs="+", required=True)
    p.add_argument("--extra", type=int, default=1024)
    p.add_argument("--updates", type=int, default=4000)
    main(p.parse_args())
