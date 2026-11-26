"""Run frozen four-head prediction and optional 25-step channel refinement."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from waveforge.lab.mcl.amortized_policy import load_policy
from waveforge.lab.mcl.baselines import uniform_parallel
from waveforge.lab.mcl.locked_channels import canonical_frame
from waveforge.lab.mcl.locked_evaluation import ranking, write
from waveforge.lab.mcl.ribbed_channels import RibbedDecoder
from waveforge.lab.mcl.ribbed_evaluation import (
    RibbedEvaluator,
    optimize_physical,
    refine_independent_starts,
    select_and_export,
)
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init
from waveforge.lab.mcl.stage5_training import condition_worker


def main(args):
    started = time.perf_counter()
    if getattr(args, "sparse_backend", None) is not None:
        os.environ["WAVEFORGE_SPARSE_BACKEND"] = args.sparse_backend
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
    model = load_policy(checkpoint, args.device)
    with torch.inference_mode():
        raw = model(torch.from_numpy(condition[None]).to(args.device))[0]
        parameters = (
            raw.cpu().double().numpy()
            if getattr(model, "prediction_space", "raw") == "physical"
            else RibbedDecoder.to_physical(raw.cpu().double()).numpy()
        )

    evaluator = RibbedEvaluator(
        task,
        args.output,
        forward_backend=getattr(args, "evaluation_backend", "fast"),
    )
    uniform = evaluator.evaluate(uniform_parallel())
    if not uniform["valid"]:
        raise RuntimeError("Uniform fallback fails the unchanged joint admission")
    proposal_heads = getattr(model, "proposal_heads", tuple(range(len(parameters))))
    starts = [
        dict(
            raw=evaluator.parameters(p, h),
            parameters=p,
            head=h,
            history=[],
            checkpoints=[],
            objective_evaluations=0,
        )
        for h, p in zip(proposal_heads, parameters, strict=True)
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
        segments = refine_independent_starts(
            evaluator, top, steps=8, workers=getattr(args, "workers", 1)
        )
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
        initial_physics_scores=len(parameters),
    )
    expected_queries = 0 if args.mode == "r0" else 25
    if result["objective_evaluations"] != expected_queries:
        raise RuntimeError("unexpected refinement budget")
    result["request_elapsed_seconds"] = time.perf_counter() - started
    result["request_elapsed_scope"] = (
        "after imports; includes preparation, loads and export"
    )
    write(args.output / "result.json", result)
    print(
        json.dumps(
            dict(
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                task=args.task,
                mode=args.mode,
                model_weight_updates=0,
                neural_candidates=len(parameters),
                refinement_objective_queries=expected_queries,
                request_elapsed_seconds=result["request_elapsed_seconds"],
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
    parser.add_argument("--sparse-backend", choices=("scipy", "cudss"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--evaluation-backend", choices=("fast", "reference"), default="fast"
    )
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
