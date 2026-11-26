"""Train a frozen four-head replacement for the unchanged 25-query refinement."""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from waveforge.lab.mcl.amortized import collect_teacher, fresh_task_groups
from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.ribbed_channels import RibbedUNet
from waveforge.lab.mcl.stage3_training import (
    protected_geometries,
    task_from_dict,
    worker_init,
)
from waveforge.lab.mcl.stage5_physics import stage5_spec
from waveforge.lab.mcl.stage5_training import condition_worker

ROOT = Path(__file__).resolve().parents[1]


def prepare(args):
    if args.teacher.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("source weights must be outside the new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks_path = ROOT / "results/stage55_20261029/inputs/evaluation_tasks.json"
    opened = [task_from_dict(t) for t in json.loads(tasks_path.read_text())["tasks"]]
    protected, hygiene = protected_geometries(args.registry, opened)
    excluded = set()
    for path in args.exclude_task_log:
        with path.open(encoding="utf-8") as handle:
            excluded.update(
                json.loads(line)["sha256"] for line in handle if line.strip()
            )
    configuration = dict(
        method="head-aligned distillation of 3x8+1 refinement",
        teacher_free=False,
        source_sha256=digest(args.teacher),
        seed=args.seed,
        train_tasks=args.train_tasks,
        validation_tasks=args.validation_tasks,
        updates=args.updates,
        batch_size=args.batch_size,
        physics=asdict(stage5_spec()),
        tim_conductivity_w_m_k=40.0,
        acceptance="unchanged joint64/128 nominal and equal pump power",
        optimizer_queries_at_inference=0,
        candidate_heads=4,
        head_learning_rate=1e-4,
        backbone_learning_rate=2e-5,
        selection="median gap to teacher, p90 gap, fallback count, earlier update",
        quality_target=dict(
            median_gap_to_teacher_percent_at_most=0.5,
            p90_gap_to_teacher_percent_at_most=2.0,
        ),
        geometry_hygiene=hygiene,
        previous_train_hashes_excluded=len(excluded),
        source_exclusion_logs_sha256=[digest(p) for p in args.exclude_task_log],
        closed_test_opened=False,
    )
    if args.architecture == "residual":
        configuration.update(
            architecture="ResidualRibbedUNet",
            prediction_space="physical",
            frozen_source_body=True,
        )
    if args.loss == "topology":
        configuration.update(
            training_loss="binary_topology",
            topology_beta=6.0,
            parameter_loss_weight=0.25,
        )
    if args.warm_start:
        configuration.update(
            warm_start_checkpoint=str(args.warm_start.resolve()),
            warm_start_sha256=digest(args.warm_start),
        )
    configuration = json.loads(json.dumps(configuration))
    config_path = args.output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError("output already contains a different experiment")
        manifest = json.loads((args.output / "tasks.json").read_text())
        groups = {
            split: [task_from_dict(t) for t in manifest[split]]
            for split in ("train", "validation")
        }
    else:
        groups = fresh_task_groups(
            args.seed,
            protected,
            train_count=args.train_tasks,
            validation_count=args.validation_tasks,
            excluded_hashes=excluded,
        )
        write(config_path, configuration)
        write(
            args.output / "tasks.json",
            {split: [t.canonical() for t in tasks] for split, tasks in groups.items()},
        )
    source = args.output / "source_policy.pt"
    if not source.exists():
        shutil.copy2(args.teacher, source)
    if digest(source) != configuration["source_sha256"]:
        raise ValueError("immutable source copy mismatch")
    if args.reuse_data:
        previous = json.loads((args.reuse_data / "config.json").read_text())
        for key in (
            "source_sha256",
            "seed",
            "train_tasks",
            "validation_tasks",
            "physics",
        ):
            if previous[key] != configuration[key]:
                raise ValueError("reused teacher dataset is incompatible")
        if json.loads((args.reuse_data / "tasks.json").read_text()) != {
            split: [t.canonical() for t in tasks] for split, tasks in groups.items()
        }:
            raise ValueError(
                "teacher dataset has different train/validation membership"
            )
        data = args.output / "data"
        data.mkdir(exist_ok=True)
        for path in (args.reuse_data / "data").iterdir():
            if path.suffix in (".npz", ".json"):
                destination = data / path.name
                if not destination.exists():
                    shutil.copy2(path, destination)
        if not (args.output / "inputs.npz").exists():
            shutil.copy2(args.reuse_data / "inputs.npz", args.output / "inputs.npz")
    return groups, configuration


def collect(args, groups):
    model = RibbedUNet().to(args.device)
    model.load_state_dict(
        torch.load(args.teacher, map_location=args.device, weights_only=True)["model"],
        strict=True,
    )
    model.eval().requires_grad_(False)
    data = args.output / "data"
    data.mkdir(exist_ok=True)
    entries = [
        (split, i, task)
        for split in ("validation", "train")
        for i, task in enumerate(groups[split])
    ]
    prepared = args.output / "inputs.npz"
    if prepared.exists():
        with np.load(prepared) as saved:
            conditions, predictions = saved["conditions"], saved["predictions"]
        if conditions.shape != (len(entries), 13, 64, 64):
            raise ValueError("conditioning cache does not match task manifest")
    else:
        conditions, predictions = [], []
        for index, (_, _, task) in enumerate(entries):
            condition, _ = condition_worker(task)
            with torch.inference_mode():
                prediction = model(torch.from_numpy(condition[None]).to(args.device))[0]
            conditions.append(condition)
            predictions.append(prediction.cpu().numpy())
            if index % 8 == 0:
                print(f"INPUTS {index + 1}/{len(entries)}", flush=True)
        conditions, predictions = np.stack(conditions), np.stack(predictions)
        temporary = prepared.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, conditions=conditions, predictions=predictions)
        temporary.replace(prepared)
    started = time.perf_counter()
    done, gains = 0, []
    with ThreadPoolExecutor(max_workers=args.teacher_jobs) as pool:
        pending = {}
        for index, (split, number, task) in enumerate(entries):
            path = data / f"{split}_{number:03d}"
            if path.with_suffix(".json").exists() and path.with_suffix(".npz").exists():
                meta = json.loads(path.with_suffix(".json").read_text())
                if meta["task_sha256"] != task.fingerprint():
                    raise ValueError("cached example task mismatch")
                done += 1
                gains.append(meta["teacher_gain_percent"])
                continue
            future = pool.submit(
                collect_teacher,
                task,
                conditions[index],
                predictions[index],
                path,
                trajectory_workers=args.trajectory_workers,
            )
            pending[future] = (split, number)
        for future in as_completed(pending):
            meta = future.result()
            split, number = pending[future]
            done += 1
            gains.append(meta["teacher_gain_percent"])
            progress = dict(
                phase="teacher_examples",
                completed=done,
                total=len(entries),
                elapsed_seconds=time.perf_counter() - started,
                median_teacher_gain_percent=float(np.median(gains)),
                latest=f"{split}_{number:03d}",
                closed_test_opened=False,
            )
            write(args.output / "progress.json", progress)
            print(
                f"TEACHER {done}/{len(entries)} {split}_{number:03d} "
                f"gain={meta['teacher_gain_percent']:.3f}% "
                f"elapsed={progress['elapsed_seconds']:.1f}s",
                flush=True,
            )
    del model
    return len(entries)


def main(args):
    worker_init()
    os.environ["WAVEFORGE_SPARSE_BACKEND"] = args.sparse_backend
    groups, config = prepare(args)
    if args.phase in ("all", "collect"):
        collect(args, groups)
    if args.phase in ("all", "train"):
        from waveforge.lab.mcl.amortized_training import train

        train(args.output, args.teacher, groups, config, device=args.device)
    if args.phase == "polish":
        from waveforge.lab.mcl.amortized_polish import polish

        opened = json.loads(
            (ROOT / "results/stage55_20261029/inputs/evaluation_tasks.json").read_text()
        )
        protected, _ = protected_geometries(
            args.registry, tuple(task_from_dict(t) for t in opened["tasks"])
        )
        polish(
            args.output,
            groups,
            config,
            protected,
            steps=args.polish_updates,
            device=args.device,
        )
    if digest(args.teacher) != config["source_sha256"]:
        raise RuntimeError("original weights changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument(
        "--teacher", type=Path, default=ROOT / "models/manifold_unet_ribbed.pt"
    )
    parser.add_argument("--exclude-task-log", nargs="*", type=Path, default=[])
    parser.add_argument("--train-tasks", type=int, default=64)
    parser.add_argument("--validation-tasks", type=int, default=8)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026103001)
    parser.add_argument("--teacher-jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--trajectory-workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--sparse-backend", choices=("scipy", "cudss"), default="cudss")
    parser.add_argument(
        "--phase", choices=("all", "collect", "train", "polish"), default="all"
    )
    parser.add_argument("--polish-updates", type=int, default=150)
    parser.add_argument(
        "--architecture", choices=("plain", "residual"), default="plain"
    )
    parser.add_argument("--reuse-data", type=Path)
    parser.add_argument(
        "--loss", choices=("parameter", "topology"), default="parameter"
    )
    parser.add_argument("--warm-start", type=Path)
    main(parser.parse_args())
