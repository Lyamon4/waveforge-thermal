"""Train a geometry-aware selector on already qualified frozen-generator labels."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from waveforge.lab.mcl.baselines import digest, uniform_parallel
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.solver_free import ranking_loss
from waveforge.lab.mcl.solver_free_raster import RasterRawPolicy
from waveforge.lab.mcl.solver_free_training import (
    _generator_hash,
    load_model,
    quality_summary,
)


def dataset(root, split, device, augmentation=None):
    tasks = json.loads((root / "tasks.json").read_text())[split]
    values = []
    for i in range(len(tasks)):
        with np.load(root / f"generated_{split}/case_{i:03d}.npz") as saved:
            row = {k: saved[k] for k in ("metrics", "valid", "peaks", "reference")}
            row["binary"] = np.concatenate(
                (saved["binaries"], uniform_parallel()[None])
            )
        with np.load(root / f"data/{split}_{i:03d}.npz") as original:
            row["raw"] = original["raw"]
        values.append(row)
    data = {k: np.stack([r[k] for r in values]) for k in values[0]}
    if augmentation is not None:
        for path in sorted(augmentation.glob("geometry_*.npz")):
            with np.load(path) as saved:
                good = saved["valid"].any(1)
                for key in data:
                    data[key] = np.concatenate((data[key], saved[key][good]))
    return {k: torch.from_numpy(v).to(device) for k, v in data.items()}


def main(args):
    torch.set_num_threads(1)
    torch.manual_seed(2026103041)
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.run / "generator/model_frozen.pt"
    model = RasterRawPolicy(
        load_model(source, args.device), hydraulic_branch=args.hydraulic_branch
    ).to(args.device)
    before = _generator_hash(model)
    train = dataset(args.run, "train", args.device, args.augmentation)
    val = dataset(args.run, "validation", args.device)
    config = dict(
        updates=args.updates,
        batch=8,
        learning_rate=3e-4,
        seed=2026103041,
        input="raw task plus actual binary candidate",
        generator_sha256=before,
        temperature_scale=0.02,
        other_metric_scale=0.05,
        ranking_weight=0.5,
        quality_checks="unchanged full64/128 dual-regime offline admission",
        training_only_reflection_augmentation=True,
        hydraulic_branch=args.hydraulic_branch,
        augmented_training_cases=len(train["raw"]),
        augmentation=str(args.augmentation) if args.augmentation else None,
    )
    write(args.output / "config.json", config)
    optimizer = torch.optim.AdamW(
        model.critic.parameters(), lr=3e-4, weight_decay=0.001
    )
    rng = torch.Generator().manual_seed(2026103042)
    best = None
    started = time.perf_counter()
    with (args.output / "history.jsonl").open("w", encoding="utf-8") as history:
        for step in range(args.updates):
            model.train().requires_grad_(True)
            index = torch.randint(len(train["raw"]), (8,), generator=rng).to(
                args.device
            )
            raw = train["raw"][index].clone()
            binary = train["binary"][index]
            if torch.rand((), generator=rng).item() < 0.5:
                raw[:, :5] = raw[:, :5].flip(-1)
                binary = binary.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            pred = model.critic(raw, binary)
            scale = raw.new_tensor([0.02, 0.02, 0.05, 0.05])
            regression = F.smooth_l1_loss(
                (pred["metrics"] - train["metrics"][index]) / scale,
                torch.zeros_like(pred["metrics"]),
            )
            loss = (
                regression
                + 0.5
                * ranking_loss(
                    pred["logits"],
                    train["peaks"][index],
                    train["valid"][index],
                    tolerance=0.002,
                )
                + 0.3
                * F.binary_cross_entropy_with_logits(
                    pred["validity_logit"], train["valid"][index].float()
                )
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.critic.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            if (step + 1) % 100 == 0:
                history.write(
                    json.dumps(dict(step=step + 1, loss=float(loss.detach()))) + "\n"
                )
                print(
                    f"RASTER {step + 1}/{args.updates} loss={loss.item():.4f}",
                    flush=True,
                )
            if (step + 1) % 200 == 0 or step + 1 == args.updates:
                model.eval()
                choices = []
                with torch.inference_mode():
                    for first in range(len(val["raw"])):
                        result = model.critic(
                            val["raw"][first : first + 1],
                            val["binary"][first : first + 1],
                        )
                        choices.extend(result["logits"].argmax(1).cpu().tolist())
                ids = np.array(choices)
                peaks = val["peaks"].cpu().numpy()
                valid = val["valid"].cpu().numpy()
                summary = quality_summary(
                    peaks[np.arange(len(ids)), ids],
                    valid[np.arange(len(ids)), ids],
                    val["reference"].cpu().numpy(),
                )
                key = (
                    summary["invalid_count"],
                    not summary["near_quality_target_passed"],
                    summary["median_gap_percent"],
                    summary["p90_gap_percent"],
                )
                row = dict(step=step + 1, summary=summary)
                write(args.output / f"validation_{step + 1:04d}.json", row)
                if best is None or key < best["key"]:
                    best = dict(key=key, **row)
                    torch.save(
                        dict(
                            model={
                                k: v.detach().cpu()
                                for k, v in model.state_dict().items()
                            },
                            architecture="RasterRawPolicy",
                            config=config,
                            online_physical_solves=0,
                            teacher_free=False,
                            validation=row,
                        ),
                        args.output / "model_best.pt",
                    )
                print(
                    f"RASTER VALIDATE {step + 1} "
                    f"gap={summary['median_gap_percent']:.3f}% "
                    f"p90={summary['p90_gap_percent']:.3f}% "
                    f"invalid={summary['invalid_count']}",
                    flush=True,
                )
    if _generator_hash(model) != before:
        raise RuntimeError("critic training changed the generator")
    result = dict(
        selected={k: v for k, v in best.items() if k != "key"},
        updates=args.updates,
        generator_unchanged=True,
        elapsed_seconds=time.perf_counter() - started,
        model_sha256=digest(args.output / "model_best.pt"),
    )
    write(args.output / "complete.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--augmentation", type=Path)
    p.add_argument("--hydraulic-branch", action="store_true")
    main(p.parse_args())
