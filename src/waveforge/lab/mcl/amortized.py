"""Head-aligned self-imitation of qualified short refinement trajectories.

This training uses optimized targets. It is separate from the earlier
teacher-free experiment and changes neither the decoder nor physical operators.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .adaptive_manifold import contrast_pair, reflected_task
from .locked_channels import canonical_frame
from .locked_evaluation import ranking
from .ribbed_channels import N_PARAMETERS, MaskedBudget, RibbedDecoder
from .stage3_training import TrainingStream


def fresh_task_groups(
    seed, protected, *, train_count, validation_count, excluded_hashes=()
):
    """Reserve validation first; exclude both orientations from training."""
    if min(train_count, validation_count) < 1:
        raise ValueError("positive train and validation task counts required")
    seen = set(excluded_hashes)

    def sample(count, forbidden, offset):
        forbidden = tuple(forbidden)
        stream = TrainingStream(
            seed + offset, forbidden + tuple(reflected_task(t) for t in forbidden)
        )
        stream.seen.update(seen)
        rng = np.random.default_rng(seed + offset + 1)
        tasks = []
        while len(tasks) < count:
            first = stream.next()
            # Same balanced workload families as the source model: an opposite
            # row pair and two ordinary procedural tasks. No outcome sampling.
            batch = (*contrast_pair(first, rng), stream.next(), stream.next())
            for original in batch:
                task = canonical_frame(original).task
                key = task.fingerprint()
                if key in seen or stream.protected.contains(task):
                    continue
                seen.add(key)
                tasks.append(task)
                if len(tasks) == count:
                    break
        return tasks

    validation = sample(validation_count, protected, 100)
    train = sample(train_count, (*protected, *validation), 200)
    return dict(train=train, validation=validation)


def select_head_targets(initial, baseline_rows, trajectories):
    """Retain each head's best admissible incumbent; never relabel a head."""
    initial = np.asarray(initial, dtype=np.float64)
    if initial.shape != (4, N_PARAMETERS) or not np.isfinite(initial).all():
        raise ValueError("four finite physical parameter vectors required")
    if len(baseline_rows) != 4:
        raise ValueError("four baseline head records required")
    targets = initial.copy()
    best = list(baseline_rows)
    for trajectory in trajectories:
        head = trajectory["head"]
        parameters = np.asarray(trajectory["parameters"], dtype=np.float64)
        if head not in range(4) or parameters.shape != (N_PARAMETERS,):
            raise ValueError("invalid trajectory head or parameter shape")
        if not np.isfinite(parameters).all():
            raise ValueError("nonfinite teacher parameters")
        candidate = trajectory["raw"]
        if candidate["valid"] and ranking(candidate) < ranking(best[head]):
            targets[head] = parameters
            best[head] = candidate
    weights = np.array([0.25 if row["valid"] else 0 for row in baseline_rows])
    for head, (before, after) in enumerate(zip(baseline_rows, best, strict=True)):
        if after is not before:
            gain = (
                100
                * (before["peak_c"] - after["peak_c"])
                / max(before["peak_c"] - 25, 1e-9)
            )
            weights[head] = 2 if not before["valid"] else 1 + np.clip(gain / 5, 0, 1)
    return targets, weights


def distillation_loss(raw, targets, weights, *, prediction_space="raw"):
    """Physical-coordinate regression, without inverse tanh at exact bounds."""
    if (
        raw.ndim != 3
        or raw.shape[1:] != (4, N_PARAMETERS)
        or targets.shape != raw.shape
        or weights.shape != raw.shape[:2]
    ):
        raise ValueError("aligned [batch,4,4125] targets and head weights required")
    if not all(torch.isfinite(x).all() for x in (raw, targets, weights)):
        raise ValueError("finite training operands required")
    if (weights < 0).any():
        raise ValueError("nonnegative head weights required")
    if prediction_space == "raw":
        physical = RibbedDecoder.to_physical(raw.flatten(0, 1)).reshape_as(raw)
    elif prediction_space == "physical":
        physical = raw
    else:
        raise ValueError("unknown prediction space")
    # Fixed scales approximate one short trajectory's displacement. Each group
    # gets equal weight, so 4096 pixel values cannot bury 29 routing controls.
    losses = []
    for start, stop, scale in (
        (0, 20, 0.5),
        (20, 25, 0.1),
        (25, 29, 0.25),
        (29, N_PARAMETERS, 0.1),
    ):
        error = (physical[..., start:stop] - targets[..., start:stop]) / scale
        losses.append(
            F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean(-1)
        )
    per_head = torch.stack(losses).mean(0)
    return (weights * per_head).sum() / weights.sum().clamp_min(1e-30)


def topology_proxy(prediction, geometries, *, prediction_space="raw", beta=6.0):
    """Training surrogate only: connected binary inference is unchanged."""
    if len(geometries) != len(prediction):
        raise ValueError("one geometry per training task required")
    decoder = RibbedDecoder([g for g in geometries for _ in range(4)])
    score, core, wall = decoder.fields(
        prediction.flatten(0, 1).double(),
        list(range(4)) * len(prediction),
        physical=prediction_space == "physical",
    )
    # The same header-only surrogate already used by the qualified hard STE.
    header = torch.zeros_like(core)
    header[:, :4, 7:57] = True
    header[:, 60:, 7:57] = True
    return MaskedBudget.apply(score, header, wall, beta).reshape(-1, 4, 64, 64)


def topology_distillation_loss(
    prediction, targets, weights, geometries, binary, *, prediction_space="raw"
):
    proxy = topology_proxy(prediction, geometries, prediction_space=prediction_space)
    if binary.shape != proxy.shape:
        raise ValueError("four aligned binary target maps per task required")
    entropy = F.binary_cross_entropy(
        proxy.clamp(1e-8, 1 - 1e-8), binary.to(proxy), reduction="none"
    ).mean((-2, -1))
    shape_loss = (weights * entropy).sum() / weights.sum().clamp_min(1e-30)
    return shape_loss + 0.25 * distillation_loss(
        prediction, targets, weights, prediction_space=prediction_space
    )


def collect_teacher(task, condition, raw, output, *, trajectory_workers=3):
    """One unchanged 3x8+1 teacher, retaining targets for all four heads."""
    import time
    from pathlib import Path

    from .baselines import uniform_parallel
    from .locked_evaluation import write
    from .ribbed_evaluation import (
        RibbedEvaluator,
        optimize_physical,
        refine_independent_starts,
    )

    output = Path(output)
    started = time.perf_counter()
    evaluator = RibbedEvaluator(task, output.parent / "work" / output.name)
    uniform = evaluator.evaluate(uniform_parallel())
    if not uniform["valid"]:
        raise RuntimeError("training task fails unchanged Uniform admission")
    initial = RibbedDecoder.to_physical(torch.from_numpy(raw).double()).numpy()
    starts = [
        dict(parameters=p, head=h, raw=evaluator.parameters(p, h))
        for h, p in enumerate(initial)
    ]
    top = sorted(starts, key=lambda value: ranking(value["raw"]))[:3]
    trajectories = refine_independent_starts(
        evaluator, top, steps=8, workers=trajectory_workers
    )
    leader = min(trajectories, key=lambda value: ranking(value["raw"]))
    trajectories.append(
        optimize_physical(
            evaluator, leader["parameters"], leader["head"], kind="Adam", steps=1
        )
    )
    if sum(t["objective_evaluations"] for t in trajectories) != 25:
        raise RuntimeError("teacher refinement budget changed")
    original_rows = [v["raw"] for v in starts]
    targets, weights = select_head_targets(initial, original_rows, trajectories)
    target_rows = [evaluator.parameters(q, h) for h, q in enumerate(targets)]
    baseline = min([uniform, *original_rows], key=ranking)
    teacher = min([uniform, *(t["raw"] for t in starts + trajectories)], key=ranking)
    if evaluator.frame.self_symmetric:
        baseline = teacher = uniform
    if not teacher["valid"] or ranking(teacher) > ranking(baseline):
        raise RuntimeError("teacher lost the original qualified incumbent")
    for h, row in enumerate(target_rows):
        if weights[h] > 0 and not row["valid"]:
            raise RuntimeError("invalid channel target entered supervised training")
    arrays = dict(
        condition=np.asarray(condition, dtype=np.float32),
        raw=np.asarray(raw),
        initial=initial,
        targets=targets,
        weights=weights.astype(np.float32),
        initial_binary=np.stack(
            [evaluator.designs[r["binary_sha256"]] for r in original_rows]
        ),
        target_binary=np.stack(
            [evaluator.designs[r["binary_sha256"]] for r in target_rows]
        ),
    )
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output.with_suffix(".npz"))
    meta = dict(
        task=task.canonical(),
        task_sha256=task.fingerprint(),
        uniform=uniform,
        initial_heads=original_rows,
        target_heads=target_rows,
        baseline=baseline,
        teacher=teacher,
        teacher_objective_queries=25,
        improved_heads=int((weights > 0.25).sum()),
        eligible_heads=int((weights > 0).sum()),
        teacher_gain_percent=100
        * (baseline["peak_c"] - teacher["peak_c"])
        / (baseline["peak_c"] - 25),
        elapsed_seconds=time.perf_counter() - started,
    )
    write(output.with_suffix(".json"), meta)
    return meta
