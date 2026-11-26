"""Balanced geometry-aware selection with explicit predicted admission margins."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from train_raster_selector import dataset

from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.fast_channels import flow_geometry_features
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.solver_free import ranking_loss
from waveforge.lab.mcl.solver_free_raster import RasterRawPolicy, choose_predictions
from waveforge.lab.mcl.solver_free_training import (
    _generator_hash,
    load_model,
    quality_summary,
)


def load_data(root, split, device, augmentation=None):
    data = dataset(root, split, "cpu", augmentation)
    cache = {}

    def features(binary):
        result = []
        for d in binary.reshape(-1, 64, 64):
            key = d.tobytes()
            if key not in cache:
                cache[key] = flow_geometry_features(d[None])[0]
            result.append(cache[key])
        return torch.from_numpy(np.stack(result).reshape(len(binary), 9, 89))

    binary = data["binary"].numpy()
    data["flow_features"] = features(binary)
    data["flow_features_flip"] = features(binary[:, :, :, ::-1])
    return {k: v.to(device) for k, v in data.items()}


def selection_key(summary):
    return (
        summary["invalid_count"],
        not summary["near_quality_target_passed"],
        summary["median_gap_percent"],
        summary["p90_gap_percent"],
    )


def main(args):
    torch.set_num_threads(1)
    torch.manual_seed(2026103061)
    args.output.mkdir(parents=True, exist_ok=False)
    model = RasterRawPolicy(
        load_model(args.run / "generator/model_frozen.pt", args.device),
        hydraulic_branch=True,
        hydraulic_features=True,
    ).to(args.device)
    if args.warm_start:
        saved = torch.load(args.warm_start, map_location=args.device, weights_only=True)
        state = {
            k.removeprefix("critic."): v
            for k, v in saved["model"].items()
            if k.startswith("critic.")
        }
        missing = model.critic.load_state_dict(state, strict=False)
        if missing.unexpected_keys or any(
            not k.startswith("geometry_flow.") for k in missing.missing_keys
        ):
            raise ValueError("incompatible critic warm start")
    before = _generator_hash(model)
    train = load_data(args.run, "train", args.device, args.augmentation)
    val = load_data(args.run, "validation", args.device)
    base_count = len(json.loads((args.run / "tasks.json").read_text())["train"])
    margins = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    config = dict(
        seed=2026103061,
        updates=args.updates,
        batch=8,
        hydraulic_branch=True,
        hydraulic_features=True,
        augmentation=str(args.augmentation),
        training_cases=len(train["raw"]),
        original_training_geometries=base_count,
        balanced_original_augmented=True,
        cosine_lr_start=3e-4,
        cosine_lr_end=3e-5,
        calibration_margins=margins,
        generator_sha256=before,
        warm_start_sha256=digest(args.warm_start) if args.warm_start else None,
    )
    write(args.output / "config.json", config)
    optimizer = torch.optim.AdamW(
        model.critic.parameters(), lr=3e-4, weight_decay=0.001
    )
    rng = torch.Generator().manual_seed(2026103062)
    best = None
    started = time.perf_counter()
    with (args.output / "history.jsonl").open("w", encoding="utf-8") as history:
        for step in range(args.updates):
            model.train().requires_grad_(True)
            i = torch.cat(
                (
                    torch.randint(base_count, (4,), generator=rng),
                    torch.randint(base_count, len(train["raw"]), (4,), generator=rng),
                )
            ).to(args.device)
            optimizer.param_groups[0]["lr"] = 3e-4 * (
                0.1 + 0.45 * (1 + math.cos(math.pi * step / max(args.updates - 1, 1)))
            )
            raw = train["raw"][i].clone()
            binary = train["binary"][i]
            features = train["flow_features"][i]
            if torch.rand((), generator=rng).item() < 0.5:
                raw[:, :5] = raw[:, :5].flip(-1)
                binary = binary.flip(-1)
                features = train["flow_features_flip"][i]
            optimizer.zero_grad(set_to_none=True)
            pred = model.critic(raw, binary, features)
            scale = raw.new_tensor([0.02, 0.02, 0.025, 0.05])
            regression = F.smooth_l1_loss(
                (pred["metrics"] - train["metrics"][i]) / scale,
                torch.zeros_like(pred["metrics"]),
            )
            loss = regression + 0.5 * ranking_loss(
                pred["logits"], train["peaks"][i], train["valid"][i], tolerance=0.002
            )
            loss = loss + 0.3 * F.binary_cross_entropy_with_logits(
                pred["validity_logit"], train["valid"][i].float()
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
            if (step + 1) % 200 == 0 or step + 1 == args.updates:
                model.eval()
                predictions = []
                with torch.inference_mode():
                    for j in range(len(val["raw"])):
                        p = model.critic(
                            val["raw"][j : j + 1],
                            val["binary"][j : j + 1],
                            val["flow_features"][j : j + 1],
                        )
                        predictions.append(p["metrics"].cpu())
                metrics = torch.cat(predictions)
                peaks = val["peaks"].cpu().numpy()
                valid = val["valid"].cpu().numpy()
                trials = []
                for margin in margins:
                    ids = choose_predictions(
                        dict(metrics=metrics), re_margin=margin
                    ).numpy()
                    summary = quality_summary(
                        peaks[np.arange(len(ids)), ids],
                        valid[np.arange(len(ids)), ids],
                        val["reference"].cpu().numpy(),
                    )
                    trials.append((selection_key(summary), margin, summary))
                key, margin, summary = min(trials, key=lambda r: r[0])
                row = dict(step=step + 1, margin=margin, summary=summary)
                write(args.output / f"validation_{step + 1:04d}.json", row)
                state = dict(
                    model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    architecture="RasterRawPolicy",
                    config=config,
                    online_physical_solves=0,
                    teacher_free=False,
                    validation=row,
                    inference=dict(selection_rule="hard_metrics", re_margin=margin),
                )
                torch.save(state, args.output / f"model_{step + 1:04d}.pt")
                if best is None or key < best["key"]:
                    best = dict(key=key, **row)
                    torch.save(state, args.output / "model_best.pt")
                print(
                    f"CALIBRATED {step + 1}/{args.updates} "
                    f"gap={summary['median_gap_percent']:.3f}% "
                    f"p90={summary['p90_gap_percent']:.3f}% "
                    f"invalid={summary['invalid_count']} margin={margin}",
                    flush=True,
                )
    if _generator_hash(model) != before:
        raise RuntimeError("critic changed the generator")
    write(
        args.output / "complete.json",
        dict(
            selected={k: v for k, v in best.items() if k != "key"},
            updates=args.updates,
            generator_unchanged=True,
            elapsed_seconds=time.perf_counter() - started,
            model_sha256=digest(args.output / "model_best.pt"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--augmentation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warm-start", type=Path)
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
