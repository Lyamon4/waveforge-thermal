"""Add upstream branch power features; keep generator and physics fixed."""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from train_solver_free_tabular import features_for_split, key

from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.solver_free_tabular import (
    BRANCH_FEATURE_VERSION,
    choose_tabular,
    pair_scores,
    predict_metrics,
)
from waveforge.lab.mcl.solver_free_training import quality_summary


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    args.feature_version = BRANCH_FEATURE_VERSION
    train = features_for_split(args, "train")
    val = features_for_split(args, "validation")
    bundle = joblib.load(args.base / "selector.joblib")
    bundle["feature_version"] = BRANCH_FEATURE_VERSION
    models = bundle["models"]
    weights = np.ones(len(train["x"]))
    weights[:256] = (len(weights) - 256) / 256
    x = train["x"].reshape(-1, 690)
    with threadpool_limits(limits=6):
        for name, i in (("relative_nominal", 0), ("relative_power", 1)):
            model = HistGradientBoostingRegressor(
                max_iter=args.iterations,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                learning_rate=0.05,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=2026103075,
            )
            y = (train["metrics"][:, :, i] / train["metrics"][:, 8:9, i] - 1).ravel()
            model.fit(x, y, sample_weight=np.repeat(weights, 9))
            models[name] = model
            print(f"FIT {name} rows={len(y)}", flush=True)
    predictions = np.stack(
        [
            predict_metrics(bundle, x, f)
            for x, f in zip(val["x"], val["flow"], strict=True)
        ]
    )
    peaks = np.stack(
        [
            np.load(args.run / f"generated_validation/case_{i:03d}.npz")["peaks"]
            for i in range(len(val["x"]))
        ]
    )
    trials = []
    for blend in (0.0, 0.5, 1.0):
        bundle["pair_blend"] = blend
        scores = np.stack(
            [
                pair_scores(bundle, x, m)
                for x, m in zip(val["x"], predictions, strict=True)
            ]
        )
        for margin in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
            ids = np.array(
                [
                    choose_tabular(m, margin, s)
                    for m, s in zip(predictions, scores, strict=True)
                ]
            )
            summary = quality_summary(
                peaks[np.arange(len(ids)), ids],
                val["valid"][np.arange(len(ids)), ids],
                val["reference"],
            )
            trials.append(dict(blend=blend, margin=margin, validation=summary))
            print(
                f"BRANCH blend={blend} margin={margin} "
                f"gap={summary['median_gap_percent']:.4f}% "
                f"p90={summary['p90_gap_percent']:.4f}% "
                f"invalid={summary['invalid_count']}",
                flush=True,
            )
    best = min(trials, key=lambda r: key(r["validation"]))
    bundle.update(pair_blend=best["blend"], re_margin=best["margin"])
    bundle["config"] = dict(
        feature_version=BRANCH_FEATURE_VERSION,
        iterations=args.iterations,
        online_physical_solves=0,
        physics_unchanged=True,
        balanced_geometry_weights=True,
        base_model_sha256=digest(args.base / "model_frozen.pt"),
    )
    joblib.dump(bundle, args.output / "selector.joblib", compress=3)
    payload = torch.load(
        args.base / "model_frozen.pt", map_location="cpu", weights_only=False
    )
    payload.update(
        selector_sha256=digest(args.output / "selector.joblib"),
        config=bundle["config"],
        validation=best["validation"],
    )
    torch.save(payload, args.output / "model_frozen.pt")
    write(
        args.output / "complete.json",
        dict(
            selected=best,
            trials=trials,
            elapsed_seconds=time.perf_counter() - started,
            model_sha256=digest(args.output / "model_frozen.pt"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--augmentation", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=700)
    main(p.parse_args())
