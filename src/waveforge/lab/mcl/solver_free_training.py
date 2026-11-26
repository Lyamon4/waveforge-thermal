"""Offline raw-feature distillation, followed by selection on actual candidates."""

import hashlib
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .amortized import distillation_loss
from .amortized_policy import load_policy
from .baselines import digest, uniform_parallel
from .locked_evaluation import write
from .solver_free import RawPortfolioPolicy, ranking_loss
from .solver_free_data import arrays, label_candidates
from .stage3_training import worker_init


def quality_summary(peaks, valid, reference):
    peaks = np.asarray(peaks, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if peaks.shape != reference.shape or np.any(reference <= 25):
        raise ValueError("aligned positive temperature-rise references required")
    gaps = 100 * (peaks - reference) / (reference - 25)
    result = dict(
        median_gap_percent=float(np.median(gaps)),
        p90_gap_percent=float(np.quantile(gaps, 0.9)),
        worst_gap_percent=float(gaps.max()),
        invalid_count=int((~np.asarray(valid, dtype=bool)).sum()),
        gaps_percent=gaps.tolist(),
    )
    result["near_quality_target_passed"] = (
        result["invalid_count"] == 0
        and result["median_gap_percent"] <= 0.5
        and result["p90_gap_percent"] <= 2
    )
    result["no_median_loss_observed"] = (
        result["invalid_count"] == 0 and result["median_gap_percent"] <= 0
    )
    return result


def _key(summary):
    return (
        summary["invalid_count"],
        summary["median_gap_percent"],
        summary["p90_gap_percent"],
    )


def _generator_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(("adapter.", "teacher.")):
            h.update(name.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_model(path, model, config, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        dict(
            model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            architecture="RawPortfolioPolicy",
            config=config,
            online_physical_solves=0,
            teacher_free=False,
            input_channels=7,
            output="one binary design via exact decoder",
            **extra,
        ),
        temporary,
    )
    temporary.replace(path)


def load_model(path, device="cpu"):
    from .amortized_policy import CandidatePortfolio

    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["architecture"] != "RawPortfolioPolicy":
        raise ValueError("wrong raw-policy architecture")
    model = RawPortfolioPolicy(CandidatePortfolio(tuple(range(8)))).to(device)
    model.load_state_dict(saved["model"], strict=True)
    return model.eval().requires_grad_(False)


def field_loss(prediction, target, mask):
    thermal = (((prediction[:, :3] - target[:, :3]) / 0.1).square() * mask).sum()
    thermal = thermal / (3 * mask.sum()).clamp_min(1)
    sensitivity = ((prediction[:, 3:] - target[:, 3:]) / 0.15).square().mean()
    return thermal + sensitivity


def parameter_loss(prediction, target):
    return distillation_loss(
        prediction.reshape(-1, 4, 4125),
        target.reshape(-1, 4, 4125),
        prediction.new_ones((len(prediction) * 2, 4)),
        prediction_space="physical",
    )


def train_generator(
    root, teacher, groups, config, *, device="cuda", updates=1600, batch=8
):
    root = Path(root)
    output = root / "generator"
    output.mkdir(exist_ok=True)
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    torch.manual_seed(config["seed"] + 20)
    train = arrays(root, "train", groups["train"])
    validation = arrays(root, "validation", groups["validation"])
    model = RawPortfolioPolicy(load_policy(teacher, device)).to(device)
    model.train().requires_grad_(True)
    x = torch.from_numpy(train["raw"]).to(device)
    y = torch.from_numpy(train["condition"][:, 3:9]).to(device)
    q = torch.from_numpy(train["parameters"].astype(np.float32)).to(device)
    vx = torch.from_numpy(validation["raw"]).to(device)
    vy = torch.from_numpy(validation["condition"][:, 3:9]).to(device)
    vq = torch.from_numpy(validation["parameters"].astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(), lr=4e-4, weight_decay=1e-5
    )
    rng = torch.Generator().manual_seed(config["seed"] + 21)
    rows = []
    started = time.perf_counter()
    latest = output / "latest.pt"
    first = 0
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"].cpu())
        first = saved["step"]
        rows = (
            json.loads((output / "checkpoints.json").read_text())
            if (output / "checkpoints.json").exists()
            else []
        )
    with (output / "history.jsonl").open("a", encoding="utf-8") as history:
        for step in range(first, updates):
            model.train()
            index = torch.randint(len(x), (batch,), generator=rng).to(device)
            lr = 4e-4 * (
                0.2 + 0.4 * (1 + math.cos(math.pi * step / max(updates - 1, 1)))
            )
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if step < updates // 2:
                predicted, _ = model.adapter(x[index])
                loss = field_loss(predicted[:, 3:9], y[index], x[index, 3:4])
            else:
                predicted = model(x[index])
                loss = field_loss(
                    predicted["condition"][:, 3:9], y[index], x[index, 3:4]
                ) + 0.5 * parameter_loss(predicted["parameters"], q[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.adapter.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            record = dict(
                step=step + 1,
                loss=float(loss.detach()),
                elapsed_seconds=time.perf_counter() - started,
            )
            history.write(json.dumps(record) + "\n")
            if (step + 1) % 100 == 0:
                history.flush()
                write(
                    root / "progress.json", dict(phase="raw_input_training", **record)
                )
                print(f"RAW {step + 1}/{updates} loss={record['loss']:.5f}", flush=True)
            if (step + 1) % 400 == 0 or step + 1 == updates:
                model.eval()
                losses = []
                with torch.inference_mode():
                    for i in range(len(vx)):
                        predicted = model(vx[i : i + 1])
                        f = field_loss(
                            predicted["condition"][:, 3:9],
                            vy[i : i + 1],
                            vx[i : i + 1, 3:4],
                        )
                        p = parameter_loss(predicted["parameters"], vq[i : i + 1])
                        losses.append((float(f), float(p)))
                score = np.mean(losses, axis=0)
                row = dict(
                    step=step + 1,
                    field_loss=float(score[0]),
                    parameter_loss=float(score[1]),
                )
                rows.append(row)
                write(output / "checkpoints.json", rows)
                save_model(
                    output / f"step_{step + 1:04d}.pt", model, config, validation=row
                )
                print(
                    f"RAW VALIDATE {step + 1} fields={score[0]:.4f} "
                    f"parameters={score[1]:.4f}",
                    flush=True,
                )
                temporary = latest.with_suffix(".tmp")
                torch.save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        rng=rng.get_state(),
                        step=step + 1,
                    ),
                    temporary,
                )
                temporary.replace(latest)
    shortlist = sorted(rows, key=lambda r: r["parameter_loss"])[:2]
    qualified = []
    for row in shortlist:
        model = load_model(output / f"step_{row['step']:04d}.pt", device)
        labels = label_generator(
            root,
            model,
            groups["validation"],
            "validation",
            output / f"qualification_{row['step']:04d}",
            device=device,
        )
        indices = np.where(labels["valid"], labels["peaks"], np.inf).argmin(1)
        actual = labels["peaks"][np.arange(len(indices)), indices]
        summary = quality_summary(
            actual, np.ones(len(actual), bool), labels["reference"]
        )
        qualified.append(dict(step=row["step"], summary=summary))
    selected = min(qualified, key=lambda r: _key(r["summary"]))
    shutil.copy2(output / f"step_{selected['step']:04d}.pt", output / "model_frozen.pt")
    complete = dict(
        selected=selected,
        qualification=qualified,
        updates=updates,
        unique_training_tasks=len(x),
        elapsed_seconds=time.perf_counter() - started,
        teacher_unchanged=digest(teacher) == config["teacher_sha256"],
    )
    write(output / "complete.json", complete)
    return complete


def label_generator(root, model, tasks, split, output, *, device="cuda", workers=2):
    """Only offline: label the exact candidates the frozen raw generator emits."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = _generator_hash(model)
    model.eval()
    contexts, parameters = [], []
    source = arrays(root, split, tasks)
    with torch.inference_mode():
        for raw in source["raw"]:
            condition, context = model.adapter(torch.from_numpy(raw[None]).to(device))
            q = model.teacher(condition)
            contexts.append(context[0].cpu().numpy())
            parameters.append(q[0].cpu().numpy())
    started = time.perf_counter()

    def work(index):
        path = output / f"case_{index:03d}"
        if path.with_suffix(".json").exists():
            meta = json.loads(path.with_suffix(".json").read_text())
            if (
                meta["generator_sha256"] != fingerprint
                or meta["task_sha256"] != tasks[index].fingerprint()
            ):
                raise ValueError("stale generated-candidate labels")
            with np.load(path.with_suffix(".npz")) as saved:
                return {k: saved[k] for k in saved.files}
        previous = json.loads((root / "data" / f"{split}_{index:03d}.json").read_text())
        known = (
            previous["candidates"],
            np.concatenate((source["binaries"][index], uniform_parallel()[None])),
        )
        labels, rows, best = label_candidates(
            tasks[index], parameters[index], output / "work" / path.stem, known=known
        )
        labels.update(
            context=contexts[index],
            parameters=parameters[index],
            reference=np.array(previous["reference"]["peak_c"]),
        )
        with path.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **labels)
        path.with_suffix(".npz.tmp").replace(path.with_suffix(".npz"))
        write(
            path.with_suffix(".json"),
            dict(
                generator_sha256=fingerprint,
                task_sha256=tasks[index].fingerprint(),
                candidates=rows,
                best_index=best,
            ),
        )
        return labels

    values = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, i): i for i in range(len(tasks))}
        for completed, future in enumerate(as_completed(futures), 1):
            values[futures[future]] = future.result()
            if completed % 8 == 0 or completed == len(tasks):
                print(f"GENERATED LABELS {split} {completed}/{len(tasks)}", flush=True)
                write(
                    root / "progress.json",
                    dict(
                        phase=f"label_generated_{split}",
                        completed=completed,
                        total=len(tasks),
                        elapsed_seconds=time.perf_counter() - started,
                    ),
                )
    return {k: np.stack([r[k] for r in values]) for k in values[0]}


def train_selector(
    root, teacher, groups, config, *, device="cuda", updates=3000, batch=32
):
    root = Path(root)
    output = root / "selector"
    output.mkdir(exist_ok=True)
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    model = load_model(root / "generator/model_frozen.pt", device)
    generator_sha = _generator_hash(model)
    training = label_generator(
        root, model, groups["train"], "train", root / "generated_train", device=device
    )
    validation = label_generator(
        root,
        model,
        groups["validation"],
        "validation",
        root / "generated_validation",
        device=device,
    )
    names = ("context", "parameters", "metrics", "valid", "peaks")
    train = {k: torch.from_numpy(training[k]).to(device) for k in names}
    val = {k: torch.from_numpy(validation[k]).to(device) for k in names}
    model.requires_grad_(False)
    for module in (model.context, model.selector):
        module.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-4, weight_decay=0.01)
    rng = torch.Generator().manual_seed(config["seed"] + 31)
    best = None
    started = time.perf_counter()
    with (output / "history.jsonl").open("w", encoding="utf-8") as history:
        for step in range(updates):
            model.train()
            i = torch.randint(len(train["context"]), (batch,), generator=rng).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model.decide(train["parameters"][i], train["context"][i])
            loss = (
                ranking_loss(
                    pred["logits"],
                    train["peaks"][i],
                    train["valid"][i],
                    tolerance=0.002,
                )
                + 0.5 * F.smooth_l1_loss(pred["metrics"], train["metrics"][i])
                + 0.3
                * F.binary_cross_entropy_with_logits(
                    pred["validity_logit"], train["valid"][i].float()
                )
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % 50 == 0:
                history.write(
                    json.dumps(dict(step=step + 1, loss=float(loss.detach()))) + "\n"
                )
            if (step + 1) % 200 == 0 or step + 1 == updates:
                model.eval()
                with torch.inference_mode():
                    p = model.decide(val["parameters"], val["context"])
                    indices = p["logits"].argmax(1).cpu().numpy()
                summary = quality_summary(
                    validation["peaks"][np.arange(len(indices)), indices],
                    validation["valid"][np.arange(len(indices)), indices],
                    validation["reference"],
                )
                row = dict(step=step + 1, summary=summary, loss=float(loss.detach()))
                write(output / f"validation_{step + 1:04d}.json", row)
                if best is None or _key(summary) < _key(best["summary"]):
                    best = row
                    save_model(
                        output / "model_best.pt",
                        model,
                        config,
                        validation=row,
                        generator_sha256=generator_sha,
                    )
                print(
                    f"SELECT {step + 1}/{updates} "
                    f"gap={summary['median_gap_percent']:.3f}% "
                    f"invalid={summary['invalid_count']}",
                    flush=True,
                )
                write(root / "progress.json", dict(phase="selector_training", **row))
    if _generator_hash(model) != generator_sha:
        raise RuntimeError("selector changed the frozen generator")
    shutil.copy2(output / "model_best.pt", root / "model_frozen.pt")
    complete = dict(
        selected=best,
        updates=updates,
        elapsed_seconds=time.perf_counter() - started,
        generator_unchanged=True,
        teacher_unchanged=digest(teacher) == config["teacher_sha256"],
        online_physical_solves=0,
        model_sha256=digest(root / "model_frozen.pt"),
    )
    write(output / "complete.json", complete)
    write(root / "complete.json", complete)
    return complete


def train(root, teacher, groups, config, *, device="cuda"):
    worker_init()
    settings = dict(
        generator_updates=1600,
        generator_batch=8,
        generator_lr=4e-4,
        raw_field_scales=(0.1, 0.15),
        parameter_loss_weight=0.5,
        selector_updates=3000,
        selector_batch=32,
        selector_lr=5e-4,
        ranking_tolerance=0.002,
        metric_loss_weight=0.5,
        validity_loss_weight=0.3,
        teacher_sha256=config["teacher_sha256"],
    )
    settings = json.loads(json.dumps(settings))
    path = Path(root) / "training_config.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("training settings changed in an existing experiment")
    if not path.exists():
        write(path, settings)
    train_generator(root, teacher, groups, config, device=device)
    result = train_selector(root, teacher, groups, config, device=device)
    if digest(teacher) != config["teacher_sha256"]:
        raise RuntimeError("source weights changed")
    return result
