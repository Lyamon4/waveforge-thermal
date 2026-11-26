"""Learn structured candidate values; the neural generator stays unchanged."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from waveforge.lab.contracts import ChipletTask
from waveforge.lab.mcl.baselines import digest, uniform_parallel
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.solver_free_tabular import (
    FEATURE_VERSION,
    GeneratorOnlyPolicy,
    candidate_features,
    choose_tabular,
    predict_metrics,
)
from waveforge.lab.mcl.solver_free_training import load_model, quality_summary
from waveforge.lab.mcl.stage3_training import task_from_dict


def features_for_split(args, split):
    path = args.output / f"features_{split}.npz"
    if path.exists():
        with np.load(path) as saved:
            return {k: saved[k] for k in saved.files}
    tasks = [
        task_from_dict(t)
        for t in json.loads((args.run / "tasks.json").read_text())[split]
    ]
    values = []

    def add(task, designs, metrics, valid, reference):
        x, f = candidate_features(
            task, designs, version=getattr(args, "feature_version", FEATURE_VERSION)
        )
        keys = []
        for d in designs:
            a, b = d.tobytes(), d[:, ::-1].tobytes()
            keys.append(hashlib.sha256(min(a, b)).hexdigest())
        values.append(
            dict(
                x=x,
                flow=f,
                metrics=metrics,
                valid=valid,
                reference=reference,
                keys=np.array(keys, dtype="S64"),
            )
        )

    for i, task in enumerate(tasks):
        with np.load(args.run / f"generated_{split}/case_{i:03d}.npz") as saved:
            designs = np.concatenate((saved["binaries"], uniform_parallel()[None]))
            add(task, designs, saved["metrics"], saved["valid"], saved["reference"])
    if split == "train":
        for path in sorted(args.augmentation.glob("geometry_*.npz")):
            index = int(path.stem.split("_")[-1])
            task = tasks[index]
            with np.load(path) as saved:
                for k in range(1, len(saved["raw"])):
                    raw = saved["raw"][k]
                    powers = np.array(
                        [
                            [
                                raw[s, c.x : c.x + c.width, c.y : c.y + c.height].sum(
                                    dtype=np.float64
                                )
                                for c in task.chiplets
                            ]
                            for s in range(3)
                        ]
                    )
                    changed = ChipletTask(task.chiplets, powers)
                    add(
                        changed,
                        saved["binary"][k],
                        saved["metrics"][k],
                        saved["valid"][k],
                        saved["reference"][k],
                    )
            print(f"FEATURES {path.stem} examples={len(values)}", flush=True)
    arrays = {k: np.stack([r[k] for r in values]) for k in values[0]}
    np.savez_compressed(args.output / f"features_{split}.npz", **arrays)
    return arrays


def key(summary):
    return (
        summary["invalid_count"],
        not summary["near_quality_target_passed"],
        summary["median_gap_percent"],
        summary["p90_gap_percent"],
    )


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    train = features_for_split(args, "train")
    val = features_for_split(args, "validation")
    cfg = dict(
        feature_version=FEATURE_VERSION,
        iterations=args.iterations,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        learning_rate=0.05,
        l2_regularization=0.1,
        early_stopping=False,
        seed=2026103071,
        original_training_geometries=256,
        training_workload_cases=len(train["x"]),
        physics_unchanged=True,
        online_physical_solves=0,
        generator_checkpoint_sha256=digest(args.run / "generator/model_frozen.pt"),
        balanced_geometry_weights=args.balance_geometries,
        calibration_margins=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    )
    write(args.output / "config.json", cfg)
    models = {}

    def fit(name, x, y, weight=None):
        m = HistGradientBoostingRegressor(
            max_iter=args.iterations,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            learning_rate=0.05,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=2026103071,
        )
        m.fit(x, y, sample_weight=weight)
        models[name] = m
        print(f"FIT {name} rows={len(y)}", flush=True)

    x = train["x"].reshape(-1, train["x"].shape[-1])
    weights = np.ones(len(train["x"]))
    if args.balance_geometries:
        weights[:256] = (len(weights) - 256) / 256
    with threadpool_limits(limits=6):
        fit("uniform_nominal", train["x"][:, 8], train["metrics"][:, 8, 0], weights)
        fit("uniform_power", train["x"][:, 8], train["metrics"][:, 8, 1], weights)
        for name, index in (("relative_nominal", 0), ("relative_power", 1)):
            y = (
                train["metrics"][:, :, index] / train["metrics"][:, 8:9, index] - 1
            ).ravel()
            fit(name, x, y, np.repeat(weights, 9))
        fit("water", x, train["metrics"][:, :, 3].ravel(), np.repeat(weights, 9))
        _, unique = np.unique(train["keys"].ravel(), return_index=True)
        fit(
            "reynolds",
            train["flow"].reshape(-1, train["flow"].shape[-1])[unique],
            train["metrics"][:, :, 2].ravel()[unique],
        )
    predictions = np.stack(
        [
            predict_metrics(dict(models=models), a, b)
            for a, b in zip(val["x"], val["flow"], strict=True)
        ]
    )
    candidates = []
    peaks = np.stack(
        [
            np.load(args.run / f"generated_validation/case_{i:03d}.npz")["peaks"]
            for i in range(len(val["x"]))
        ]
    )
    for margin in cfg["calibration_margins"]:
        ids = np.array([choose_tabular(p, margin) for p in predictions])
        summary = quality_summary(
            peaks[np.arange(len(ids)), ids],
            val["valid"][np.arange(len(ids)), ids],
            val["reference"],
        )
        candidates.append((key(summary), margin, summary))
        print(
            f"TABULAR margin={margin} gap={summary['median_gap_percent']:.3f}% "
            f"p90={summary['p90_gap_percent']:.3f}% invalid={summary['invalid_count']}",
            flush=True,
        )
    _, margin, summary = min(candidates, key=lambda r: r[0])
    bundle = dict(
        models=models, re_margin=margin, feature_version=FEATURE_VERSION, config=cfg
    )
    joblib.dump(bundle, args.output / "selector.joblib", compress=3)
    model = GeneratorOnlyPolicy(load_model(args.run / "generator/model_frozen.pt"))
    torch.save(
        dict(
            model=model.state_dict(),
            architecture="GeneratorOnlyPolicy",
            selector_file="selector.joblib",
            selector_sha256=digest(args.output / "selector.joblib"),
            config=cfg,
            online_physical_solves=0,
            teacher_free=False,
            validation=summary,
        ),
        args.output / "model_frozen.pt",
    )
    write(
        args.output / "complete.json",
        dict(
            validation=summary,
            re_margin=margin,
            elapsed_seconds=time.perf_counter() - started,
            model_sha256=digest(args.output / "model_frozen.pt"),
            flow_mae=float(
                np.abs(predictions[:, :, 2] - val["metrics"][:, :, 2]).mean()
            ),
            flow_max_abs_error=float(
                np.abs(predictions[:, :, 2] - val["metrics"][:, :, 2]).max()
            ),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--augmentation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--balance-geometries", action="store_true")
    main(p.parse_args())
