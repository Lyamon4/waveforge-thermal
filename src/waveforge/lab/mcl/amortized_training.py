"""GPU self-distillation with independent, unchanged binary physics selection."""

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .amortized import distillation_loss, topology_distillation_loss
from .baselines import digest, uniform_parallel
from .locked_evaluation import ranking, write
from .ribbed_channels import RibbedDecoder, RibbedGeometry, RibbedUNet
from .ribbed_evaluation import RibbedEvaluator


def validation_summary(actual, teachers, original):
    selected = [r["selected"] for r in actual]
    peaks = np.array([r["peak_c"] for r in selected], dtype=float)
    teacher = np.array([r["peak_c"] for r in teachers], dtype=float)
    source = np.array([r["peak_c"] for r in original], dtype=float)
    if not len(peaks) or np.any(teacher <= 25) or np.any(source <= 25):
        raise ValueError("positive nonempty baseline overheating required")
    gaps = 100 * (peaks - teacher) / (teacher - 25)
    gains = 100 * (source - peaks) / (source - 25)
    result = dict(
        median_gap_to_teacher_percent=float(np.median(gaps)),
        p90_gap_to_teacher_percent=float(np.quantile(gaps, 0.9)),
        worst_gap_to_teacher_percent=float(gaps.max()),
        median_gain_vs_original_r0_percent=float(np.median(gains)),
        gaps_to_teacher_percent=gaps.tolist(),
        gains_vs_original_r0_percent=gains.tolist(),
        fallback_count=sum(r["fallback_used"] for r in actual),
        jointly_admissible=all(r["valid"] for r in selected),
    )
    if all("modes" in r for r in selected + teachers):
        powered = np.array(
            [r["modes"]["128"]["power"]["silicon_peak_c"] for r in selected]
        )
        ref = np.array([r["modes"]["128"]["power"]["silicon_peak_c"] for r in teachers])
        result["median_equal_power_gap_to_teacher_percent"] = float(
            np.median(100 * (powered - ref) / (ref - 25))
        )
    return result


def _selection_key(summary, step):
    return (
        summary["median_gap_to_teacher_percent"],
        summary["p90_gap_to_teacher_percent"],
        summary["fallback_count"],
        step,
    )


def _arrays(root, split, tasks):
    examples, metadata = [], []
    for i, task in enumerate(tasks):
        path = root / "data" / f"{split}_{i:03d}"
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["task_sha256"] != task.fingerprint():
            raise ValueError("dataset entry has wrong geometry or workload")
        with np.load(path.with_suffix(".npz")) as saved:
            examples.append({key: saved[key] for key in saved.files})
        metadata.append(meta)
    arrays = {key: np.stack([r[key] for r in examples]) for key in examples[0]}
    return arrays, metadata


class _Validation:
    def __init__(self, root, tasks, arrays, metadata, device):
        self.metadata = metadata
        self.inputs = torch.from_numpy(arrays["condition"]).to(device)
        self.evaluators = []
        for i, (task, meta) in enumerate(zip(tasks, metadata, strict=True)):
            evaluator = RibbedEvaluator(task, root / "validation" / f"case_{i:03d}")
            known = [(meta["uniform"], uniform_parallel())]
            for field, array in (
                ("initial_heads", "initial_binary"),
                ("target_heads", "target_binary"),
            ):
                known.extend(zip(meta[field], arrays[array][i], strict=True))
            # Exact memoization of already qualified designs in the same task.
            # No teacher parameters or scores enter the network prediction.
            for row, binary in known:
                key = row["binary_sha256"]
                evaluator.cache[key] = row
                evaluator.designs[key] = binary.copy()
            self.evaluators.append(evaluator)

    def score(self, physical, proposal_heads=None):
        records = []
        for q, evaluator, meta in zip(
            physical, self.evaluators, self.metadata, strict=True
        ):
            identities = (
                tuple(range(len(q))) if proposal_heads is None else proposal_heads
            )
            heads = [
                evaluator.parameters(p, h) for h, p in zip(identities, q, strict=True)
            ]
            head = min(range(len(heads)), key=lambda h: ranking(heads[h]))
            uniform = meta["uniform"]
            fallback = evaluator.frame.self_symmetric or ranking(uniform) < ranking(
                heads[head]
            )
            records.append(
                dict(
                    selected=uniform if fallback else heads[head],
                    fallback_used=fallback,
                    head=identities[head],
                    parameters=q[head].copy(),
                )
            )
        summary = validation_summary(
            records,
            [r["teacher"] for r in self.metadata],
            [r["baseline"] for r in self.metadata],
        )
        if not summary["jointly_admissible"]:
            raise RuntimeError("student escaped the inherited admission rules")
        return summary, records

    def evaluate(self, model):
        model.eval()
        with torch.inference_mode():
            # Deployment uses batch one. Keep its numerical path for selection.
            raw = torch.cat([model(x[None]) for x in self.inputs])
            q = (
                raw.cpu().double()
                if getattr(model, "prediction_space", "raw") == "physical"
                else RibbedDecoder.to_physical(raw.flatten(0, 1).cpu().double())
            )
        return self.score(
            q.reshape(-1, raw.shape[1], 4125).numpy(),
            getattr(model, "proposal_heads", None),
        )

    def export(self, records, root):
        for i, (record, evaluator) in enumerate(
            zip(records, self.evaluators, strict=True)
        ):
            fallback = record["fallback_used"]
            evaluator.export(
                root / f"case_{i:03d}",
                record["selected"],
                None if fallback else record["parameters"],
                None if fallback else record["head"],
                method="distilled-R0",
                objective_evaluations=0,
                fallback_used=fallback,
            )


def _save_model(path, model, step, config, summary):
    temporary = path.with_suffix(".tmp")
    torch.save(
        dict(
            model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            architecture=type(model).__name__,
            prediction_space=getattr(model, "prediction_space", "raw"),
            selected_step=step,
            training_method="self-distilled short refinement",
            teacher_free=False,
            source_sha256=config["source_sha256"],
            validation=summary,
            physics_profile="unchanged-stage5",
            decoder="RibbedDecoder",
            refinement_objective_queries_at_inference=0,
        ),
        temporary,
    )
    temporary.replace(path)


def train(root, source, groups, config, *, device="cuda"):
    root = Path(root)
    output = root / "training"
    output.mkdir(exist_ok=True)
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"] + 300)
    training, _ = _arrays(root, "train", groups["train"])
    validation, meta = _arrays(root, "validation", groups["validation"])
    source_state = torch.load(source, map_location=device, weights_only=True)["model"]
    if config.get("architecture") == "ResidualRibbedUNet":
        from .amortized_policy import ResidualRibbedUNet

        model = ResidualRibbedUNet().to(device)
        model.source.load_state_dict(source_state)
        difference = np.abs(training["targets"] - training["initial"])
        scale = np.empty(4125)
        for start, stop, floor in (
            (0, 20, 0.25),
            (20, 25, 0.1),
            (25, 29, 0.25),
            (29, 4125, 0.1),
        ):
            scale[start:stop] = max(
                float(difference[..., start:stop].max()) * 1.15, floor
            )
        model.correction_scale.copy_(torch.from_numpy(scale).to(device))
    else:
        model = RibbedUNet().to(device)
        model.load_state_dict(source_state)
        if config.get("warm_start_checkpoint"):
            warm = Path(config["warm_start_checkpoint"])
            if digest(warm) != config["warm_start_sha256"]:
                raise ValueError("warm-start weights changed")
            model.load_state_dict(
                torch.load(warm, map_location=device, weights_only=True)["model"]
            )
    monitor = _Validation(root, groups["validation"], validation, meta, device)
    inputs = torch.from_numpy(training["condition"]).to(device)
    targets = torch.from_numpy(training["targets"].astype(np.float32)).to(device)
    weights = torch.from_numpy(training["weights"]).to(device)
    binary = torch.from_numpy(training["target_binary"]).to(device)
    geometries = [RibbedGeometry.from_task(t) for t in groups["train"]]
    eligible = torch.nonzero(weights.sum(1) > 0).flatten()
    if not len(eligible):
        raise RuntimeError("no admissible teacher examples to learn from")
    body, heads = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (
                heads if name.startswith(("geometry.", "pixels.", "delta_")) else body
            ).append(parameter)
    optimizer = torch.optim.Adam(
        [
            dict(params=body, lr=config["backbone_learning_rate"]),
            dict(params=heads, lr=config["head_learning_rate"]),
        ]
    )
    rng = torch.Generator().manual_seed(config["seed"] + 400)
    start_step, elapsed_before = 0, 0.0
    latest = output / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["sampling_rng"].cpu())
        start_step, elapsed_before = saved["step"], saved["elapsed_seconds"]
    started = time.perf_counter()
    baseline = [
        dict(
            selected=m["baseline"],
            fallback_used=(
                m["baseline"]["binary_sha256"] == m["uniform"]["binary_sha256"]
            ),
        )
        for m in meta
    ]
    original_summary = validation_summary(
        baseline, [m["teacher"] for m in meta], [m["baseline"] for m in meta]
    )
    best_step = 0
    best_summary, _ = monitor.evaluate(model)
    if (output / "best.json").exists():
        best = json.loads((output / "best.json").read_text())
        best_step, best_summary = best["step"], best["summary"]
    else:
        _save_model(output / "model_best.pt", model, 0, config, best_summary)
        write(output / "best.json", dict(step=0, summary=best_summary))
    # A fixed-offset comparator tests whether a global average update suffices.
    delta = (training["targets"] - training["initial"]).mean(0)
    np.savez_compressed(output / "mean_correction.npz", delta=delta)
    corrected = torch.from_numpy(validation["initial"] + delta[None])
    for q in corrected.flatten(0, 1):
        RibbedDecoder.clamp_physical(q)
    correction_summary, _ = monitor.score(corrected.numpy())
    write(output / "fixed_correction_validation.json", correction_summary)
    updates = config["updates"]
    batch = config["batch_size"]
    history_path = output / "history.jsonl"
    if history_path.exists():
        lines = [
            line
            for line in history_path.read_text().splitlines()
            if json.loads(line)["step"] <= start_step
        ]
        history_path.write_text("".join(line + "\n" for line in lines))
    with history_path.open("a", encoding="utf-8") as history:
        for step in range(start_step, updates):
            model.train().requires_grad_(True)
            index = eligible[
                torch.randint(len(eligible), (batch,), generator=rng).to(device)
            ]
            factor = 0.2 + 0.4 * (1 + math.cos(math.pi * step / max(updates - 1, 1)))
            for group, rate in zip(
                optimizer.param_groups,
                (config["backbone_learning_rate"], config["head_learning_rate"]),
                strict=True,
            ):
                group["lr"] = rate * factor
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs[index])
            space = getattr(model, "prediction_space", "raw")
            if config.get("training_loss") == "binary_topology":
                selected_geometries = [geometries[i] for i in index.cpu().tolist()]
                loss = topology_distillation_loss(
                    prediction,
                    targets[index],
                    weights[index],
                    selected_geometries,
                    binary[index],
                    prediction_space=space,
                )
            else:
                loss = distillation_loss(
                    prediction, targets[index], weights[index], prediction_space=space
                )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            row = dict(
                step=step + 1,
                loss=loss.item(),
                gradient_norm=float(norm),
                elapsed_seconds=elapsed_before + time.perf_counter() - started,
            )
            history.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 0 or (step + 1) % 50 == 0:
                history.flush()
                write(root / "progress.json", dict(phase="training", **row))
                print(
                    f"DISTILL {step + 1}/{updates} loss={loss.item():.6f}", flush=True
                )
            if (step + 1) % 200 == 0 or step + 1 == updates:
                summary, _ = monitor.evaluate(model)
                write(output / f"validation_{step + 1:04d}.json", summary)
                _save_model(
                    output / f"model_{step + 1:04d}.pt",
                    model,
                    step + 1,
                    config,
                    summary,
                )
                print(
                    f"VALIDATE {step + 1} "
                    f"gap={summary['median_gap_to_teacher_percent']:.3f}% "
                    f"gain={summary['median_gain_vs_original_r0_percent']:.3f}%",
                    flush=True,
                )
                if _selection_key(summary, step + 1) < _selection_key(
                    best_summary, best_step
                ):
                    best_step, best_summary = step + 1, summary
                    _save_model(
                        output / "model_best.pt", model, best_step, config, best_summary
                    )
                    write(
                        output / "best.json", dict(step=best_step, summary=best_summary)
                    )
            if (step + 1) % 100 == 0 or step + 1 == updates:
                temporary = latest.with_suffix(".tmp")
                torch.save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        sampling_rng=rng.get_state(),
                        step=step + 1,
                        elapsed_seconds=elapsed_before + time.perf_counter() - started,
                    ),
                    temporary,
                )
                temporary.replace(latest)
    model.load_state_dict(
        torch.load(output / "model_best.pt", map_location=device, weights_only=True)[
            "model"
        ]
    )
    model.eval().requires_grad_(False)
    if hasattr(model, "source") and not all(
        torch.equal(v, source_state[k]) for k, v in model.source.state_dict().items()
    ):
        raise RuntimeError("frozen source tensors changed")
    final_summary, records = monitor.evaluate(model)
    monitor.export(records, output / "validation_designs")
    shutil.copy2(output / "model_best.pt", output / "model_frozen.pt")
    if digest(source) != config["source_sha256"]:
        raise RuntimeError("source weights changed during training")
    complete = dict(
        selected_step=best_step,
        training_updates=updates,
        unique_training_tasks=len(groups["train"]),
        task_presentations=updates * batch,
        validation_tasks=len(groups["validation"]),
        teacher_free=False,
        original_r0=original_summary,
        student_r0=final_summary,
        fixed_correction=correction_summary,
        refinement_objective_queries_at_inference=0,
        model_sha256=digest(output / "model_frozen.pt"),
        source_unchanged=True,
        closed_test_opened=False,
        elapsed_seconds=elapsed_before + time.perf_counter() - started,
    )
    write(output / "complete.json", complete)
    write(
        root / "progress.json",
        dict(
            phase="complete",
            selected_step=best_step,
            median_gap=final_summary["median_gap_to_teacher_percent"],
        ),
    )
    return complete
