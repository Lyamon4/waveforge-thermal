"""Evaluate the published ManifoldUNet on one open development case.

Install the package first: python -m pip install -e .
Then: python examples/evaluate_pretrained_mcl.py --output artifacts/example
This runs four-head selection and25 refinement steps; it does not train weights.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init
from waveforge.lab.mcl.stage5_decoder import ManifoldUNet
from waveforge.lab.mcl.stage5_training import condition_worker
from waveforge.lab.mcl.stage51_evaluation import neural_task


def main(args):
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = Path(args.checkpoint or root / "models/manifold_unet.pt")
    tasks_path = Path(args.tasks or root / "data/development_tasks.json")
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"]
    if not 0 <= args.task < len(tasks):
        raise ValueError("task index is outside the supplied open-development set")
    worker_init()
    model = ManifoldUNet().to(args.device)
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=True
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    task = task_from_dict(tasks[args.task])
    condition, _counts = condition_worker(task)
    with torch.inference_mode():
        parameters = (
            model(torch.from_numpy(condition[None]).to(args.device))[0].cpu().numpy()
        )
    r0, r25 = neural_task(task, parameters, Path(args.output))
    result = dict(
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        task=args.task,
        model_training_updates=0,
        R0_nominal_128_silicon_c=r0["selected"]["fine"]["silicon_peak_c"],
        R25_nominal_128_silicon_c=r25["selected"]["fine"]["silicon_peak_c"],
        R0_admissible_on_both_grids=r0["selected_admissible_both"],
        R25_admissible_on_both_grids=r25["selected_admissible_both"],
        scope=(
            "Raw nominal designs on one open case; "
            "other operating regimes require their own audit."
        ),
    )
    (Path(args.output) / "example_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--tasks")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
