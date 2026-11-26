"""Learn signed differences between learned candidates; reuse fixed value models."""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.solver_free_tabular import (
    choose_tabular,
    pair_scores,
    predict_metrics,
)
from waveforge.lab.mcl.solver_free_training import quality_summary


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bundle = joblib.load(args.source / "selector.joblib")
    with np.load(args.source / "features_train.npz") as saved:
        x = saved["x"]
        metrics = saved["metrics"]
        valid = saved["valid"]
    a, b = np.triu_indices(8, 1)
    mean = (x[:, a] + x[:, b]) * 0.5
    diff = x[:, a] - x[:, b]
    rng = np.random.default_rng(2026103081)
    sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=diff.shape[:2])
    pair_x = np.concatenate((mean, diff * sign[:, :, None]), 2).reshape(
        -1, 2 * x.shape[-1]
    )
    y = ((metrics[:, a, 0] - metrics[:, b, 0]) * sign).ravel()
    best = np.where(valid, metrics[:, :, 0], np.inf).min(1)
    near = (metrics[:, :, 0] - best[:, None]) <= 0.005 * best[:, None]
    weights = np.where(near[:, a] | near[:, b], 3.0, 0.5)
    weights[:256] *= (len(x) - 256) / 256
    cfg = dict(
        seed=2026103081,
        iterations=args.iterations,
        learning_rate=0.035,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        balanced_geometry_weights=True,
        ranking="antisymmetric pairwise differences, mean score over eight proposals",
        source_bundle_sha256=digest(args.source / "selector.joblib"),
        training_pairs=len(y),
        original_training_geometries=256,
    )
    write(args.output / "config.json", cfg)
    model = HistGradientBoostingRegressor(
        max_iter=args.iterations,
        learning_rate=0.035,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=2026103081,
    )
    with threadpool_limits(limits=6):
        model.fit(pair_x, y, sample_weight=weights.ravel())
    bundle["models"]["pairwise"] = model
    bundle["pairwise_config"] = cfg
    with np.load(args.source / "features_validation.npz") as saved:
        val = {k: saved[k] for k in saved.files}
    true = np.stack(
        [
            np.load(args.run / f"generated_validation/case_{i:03d}.npz")["peaks"]
            for i in range(len(val["x"]))
        ]
    )
    predictions = [
        predict_metrics(bundle, a, b)
        for a, b in zip(val["x"], val["flow"], strict=True)
    ]
    scores = [
        pair_scores(bundle, a, m) for a, m in zip(val["x"], predictions, strict=True)
    ]
    choices = []
    for margin in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]:
        ids = np.array(
            [
                choose_tabular(m, margin, s)
                for m, s in zip(predictions, scores, strict=True)
            ]
        )
        summary = quality_summary(
            true[np.arange(len(ids)), ids],
            val["valid"][np.arange(len(ids)), ids],
            val["reference"],
        )
        key = (
            summary["invalid_count"],
            not summary["near_quality_target_passed"],
            summary["median_gap_percent"],
            summary["p90_gap_percent"],
        )
        choices.append((key, margin, summary))
        print(
            f"PAIRS margin={margin} gap={summary['median_gap_percent']:.3f}% "
            f"p90={summary['p90_gap_percent']:.3f}% invalid={summary['invalid_count']}",
            flush=True,
        )
    _, margin, summary = min(choices, key=lambda r: r[0])
    bundle["re_margin"] = margin
    joblib.dump(bundle, args.output / "selector.joblib", compress=3)
    state = torch.load(
        args.source / "model_frozen.pt", map_location="cpu", weights_only=True
    )
    state.update(
        selector_sha256=digest(args.output / "selector.joblib"),
        validation=summary,
        pairwise_training=cfg,
    )
    torch.save(state, args.output / "model_frozen.pt")
    write(
        args.output / "complete.json",
        dict(
            validation=summary,
            re_margin=margin,
            elapsed_seconds=time.perf_counter() - started,
            model_sha256=digest(args.output / "model_frozen.pt"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=900)
    main(p.parse_args())
