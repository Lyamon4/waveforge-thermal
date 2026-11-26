"""Offline training for millisecond raw-task to single-channel-map inference."""

import argparse
import os
from pathlib import Path

from waveforge.lab.mcl.solver_free_data import collect_dataset, prepare_dataset

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    os.environ["WAVEFORGE_SPARSE_BACKEND"] = "cudss"
    groups, config = prepare_dataset(
        args.output,
        args.teacher,
        args.registry,
        train_count=args.train_tasks,
        validation_count=args.validation_tasks,
        previous=args.previous,
    )
    if args.phase in ("collect", "all"):
        collect_dataset(args.output, args.teacher, groups, device=args.device)
    if args.phase in ("train", "all"):
        from waveforge.lab.mcl.solver_free_training import train

        train(args.output, args.teacher, groups, config, device=args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--teacher", type=Path, default=ROOT / "models/manifold_unet_amortized_r0.pt"
    )
    parser.add_argument(
        "--previous", type=Path, default=ROOT.parent / "WaveForge_AMORTIZED_20261030"
    )
    parser.add_argument("--train-tasks", type=int, default=256)
    parser.add_argument("--validation-tasks", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", choices=("collect", "train", "all"), default="all")
    main(parser.parse_args())
