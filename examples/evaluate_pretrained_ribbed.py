"""Run frozen four-head prediction and optional 25-step channel refinement."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from waveforge.lab.mcl.baselines import uniform_parallel
from waveforge.lab.mcl.locked_channels import canonical_frame
from waveforge.lab.mcl.locked_evaluation import ranking
from waveforge.lab.mcl.ribbed_channels import RibbedDecoder, RibbedUNet
from waveforge.lab.mcl.ribbed_evaluation import (
    RibbedEvaluator,
    optimize_physical,
    select_and_export,
)
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init
from waveforge.lab.mcl.stage5_training import condition_worker


def main(args):
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint or root / "models/manifold_unet_ribbed.pt"
    tasks_path = args.tasks or (
        root / "results/stage55_20261029/inputs/evaluation_tasks.json"
    )
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"]
    if not 0 <= args.task < len(tasks):
        raise ValueError("task index is outside the supplied open-development set")
    worker_init()
    task = task_from_dict(tasks[args.task])
    condition, _counts = condition_worker(canonical_frame(task).task)
    model = RibbedUNet().to(args.device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=args.device, weights_only=True)["model"],
        strict=True,
    )
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        raw = model(torch.from_numpy(condition[None]).to(args.device))[0]
        parameters = RibbedDecoder.to_physical(raw.cpu().double()).numpy()

    evaluator = RibbedEvaluator(task, args.output)
    uniform = evaluator.evaluate(uniform_parallel())
    if not uniform["valid"]:
        raise RuntimeError("Uniform fallback fails the unchanged joint admission")
    starts = [
        dict(
            raw=evaluator.parameters(p, h),
            parameters=p,
            head=h,
            history=[],
            checkpoints=[],
            objective_evaluations=0,
        )
        for h, p in enumerate(parameters)
    ]
    trajectories = list(starts)
    if args.mode == "r25":
        best = min(starts, key=lambda t: ranking(t["raw"]))
        trajectories.append(
            optimize_physical(
                evaluator, best["parameters"], best["head"], kind="Adam", steps=25
            )
        )
    elif args.mode == "multi25":
        top = sorted(starts, key=lambda t: ranking(t["raw"]))[:3]
        segments = [
            optimize_physical(
                evaluator, t["parameters"], t["head"], kind="Adam", steps=8
            )
            for t in top
        ]
        leader = min(segments, key=lambda t: ranking(t["raw"]))
        trajectories.extend(segments)
        trajectories.append(
            optimize_physical(
                evaluator, leader["parameters"], leader["head"], kind="Adam", steps=1
            )
        )
    result = select_and_export(
        evaluator,
        args.output,
        trajectories,
        uniform,
        method=args.mode,
        initial_physics_scores=4,
    )
    expected_queries = 0 if args.mode == "r0" else 25
    if result["objective_evaluations"] != expected_queries:
        raise RuntimeError("unexpected refinement budget")
    print(
        json.dumps(
            dict(
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                task=args.task,
                mode=args.mode,
                model_weight_updates=0,
                refinement_objective_queries=expected_queries,
                jointly_admissible=result["selected"]["valid"],
                fallback_used=result["fallback_used"],
                nominal_c=result["selected"]["modes"]["128"]["nominal"][
                    "silicon_peak_c"
                ],
                equal_pump_power_c=result["selected"]["modes"]["128"]["power"][
                    "silicon_peak_c"
                ],
                output=str(args.output),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("r0", "r25", "multi25"), default="r0")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
