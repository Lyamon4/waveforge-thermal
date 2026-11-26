"""Independently label new frozen raw proposals in the unchanged physical model."""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.locked_evaluation import write
from waveforge.lab.mcl.solver_free_training import (
    label_generator,
    load_model,
    quality_summary,
)
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init


def main(args):
    worker_init()
    os.environ["WAVEFORGE_SPARSE_BACKEND"] = "cudss"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "generator").mkdir()
    shutil.copy2(args.checkpoint, args.output / "generator/model_frozen.pt")
    shutil.copy2(args.run / "tasks.json", args.output / "tasks.json")
    groups = {
        k: [task_from_dict(t) for t in v]
        for k, v in json.loads((args.run / "tasks.json").read_text()).items()
    }
    model = load_model(args.checkpoint, "cuda")
    config = dict(
        model_sha256=digest(args.checkpoint),
        source_data=str(args.run.resolve()),
        physics_unchanged=True,
    )
    write(args.output / "config.json", config)
    for split in ("validation", "train"):
        labels = label_generator(
            args.run,
            model,
            groups[split],
            split,
            args.output / f"generated_{split}",
            device="cuda",
        )
        if split == "validation":
            ids = np.where(labels["valid"], labels["peaks"], np.inf).argmin(1)
            summary = quality_summary(
                labels["peaks"][np.arange(len(ids)), ids],
                labels["valid"][np.arange(len(ids)), ids],
                labels["reference"],
            )
            write(args.output / "generator/physical_validation.json", summary)
            print("REFIT ORACLE " + json.dumps(summary), flush=True)
    write(args.output / "labels_complete.json", config)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
