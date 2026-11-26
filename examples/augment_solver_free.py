"""Expand only training workloads using the unchanged fixed-design thermal model."""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from waveforge.lab.contracts import ChipletTask
from waveforge.lab.mcl.baselines import uniform_parallel
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.solver_free import raw_features
from waveforge.lab.mcl.solver_free_augmentation import response_labels
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init


def main(args):
    worker_init()
    os.environ["WAVEFORGE_SPARSE_BACKEND"] = "cudss"
    args.output.mkdir(parents=True, exist_ok=True)
    groups = json.loads((args.run / "tasks.json").read_text())
    tasks = [task_from_dict(t) for t in groups["train"]]
    rng = np.random.default_rng(2026103051)
    indices = rng.permutation(len(tasks))[: args.geometries].tolist()
    config = dict(
        seed=2026103051,
        training_indices=indices,
        workloads_per_geometry=args.loads,
        physics="unchanged, separate operator for each geometry and design",
        validation_augmented=False,
        official_tests_opened=False,
    )
    if (args.output / "config.json").exists() and json.loads(
        (args.output / "config.json").read_text()
    ) != config:
        raise ValueError("augmentation settings differ")
    write(args.output / "config.json", config)

    def augment(index):
        path = args.output / f"geometry_{index:03d}.npz"
        if path.exists():
            return
        task = tasks[index]
        rnd = np.random.default_rng(2026103051 + index + 1)
        p = np.repeat(np.array(task.powers_w)[None], args.loads, axis=0)
        ccd = [i for i, c in enumerate(task.chiplets) if c.width == 8]
        for k in range(1, args.loads):
            for scenario in range(3):
                for _ in range(10000):
                    load = rnd.dirichlet(np.full(8, 2.0)) * 310
                    if load.max() <= 80:
                        break
                p[k, scenario, ccd] = load
            # Mix in the procedural family's uniform and row-biased modes.
            if k % 2 == 0:
                p[k, 0, ccd] = 38.75
                lower = [i for i in ccd if task.chiplets[i].y < 32]
                upper = [i for i in ccd if task.chiplets[i].y >= 32]
                p[k, 2, lower] = 58.125
                p[k, 2, upper] = 19.375
        raw = np.stack([raw_features(ChipletTask(task.chiplets, v)) for v in p])
        with np.load(args.run / f"generated_train/case_{index:03d}.npz") as original:
            designs = np.concatenate((original["binaries"], uniform_parallel()[None]))
            expected_metrics = original["metrics"].copy()
            expected_valid = original["valid"].copy()
        metrics, valid = [], []
        known = {}
        for j, d in enumerate(designs):
            key = hashlib.sha256(d.tobytes()).hexdigest()
            if key not in known:
                known[key] = response_labels(task, d, p)
            m, v = known[key]
            if (
                not np.allclose(m[0], expected_metrics[j], rtol=0, atol=1e-6)
                or v[0] != expected_valid[j]
            ):
                raise RuntimeError(
                    "superposition disagrees with the original physical labels"
                )
            metrics.append(m)
            valid.append(v)
        metrics = np.stack(metrics, axis=1).astype(np.float32)
        valid = np.stack(valid, axis=1)
        peaks = 25 + 50 * metrics[:, :, 0].astype(np.float64)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                raw=raw,
                binary=np.repeat(designs[None], len(p), axis=0),
                metrics=metrics,
                valid=valid,
                peaks=peaks,
                reference=np.min(np.where(valid, peaks, np.inf), axis=1),
            )
        temporary.replace(path)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(augment, i) for i in indices]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            print(
                f"AUGMENT {completed}/{len(indices)} "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
            write(
                args.output / "progress.json",
                dict(
                    completed=completed,
                    total=len(indices),
                    elapsed_seconds=time.perf_counter() - start,
                ),
            )
    write(
        args.output / "complete.json",
        dict(
            geometries=len(indices),
            workload_examples=len(indices) * args.loads,
            elapsed_seconds=time.perf_counter() - start,
            input_generation_and_final_physics_unchanged=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometries", type=int, default=64)
    p.add_argument("--loads", type=int, default=32)
    main(p.parse_args())
