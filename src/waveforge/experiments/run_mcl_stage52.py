"""Frozen Stage5 weights + unchanged Stage51 selector/refinement; sealed controls."""

import argparse
import copy
import csv
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from waveforge.experiments.audit_mcl_stage5_admission import record, summarize
from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.contracts import problem_from_task
from waveforge.lab.mcl.high_fidelity_eval import refine_problem
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init, write_json
from waveforge.lab.mcl.stage4_inventory import seal_files, verify_seal
from waveforge.lab.mcl.stage5_admission import system
from waveforge.lab.mcl.stage5_evaluation import REGIMES, pressure_for
from waveforge.lab.mcl.stage5_physics import stage5_spec
from waveforge.lab.mcl.stage5_training import condition_worker
from waveforge.lab.mcl.stage51 import (
    FLOW_CONFIG,
    apply_selection_gate,
    load_model,
)
from waveforge.lab.mcl.stage51_evaluation import neural_task

OLD_SHA = "95701d5448a5ad6845ef2851ed0bf3bd11399cbb6908e9de5ff492cbd068b344"
CLASSICAL = ("Manifold-Adam", "Manifold-MMA")
NEURAL = ("Manifold-UNet-R0", "Manifold-UNet-R25-C")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_control(manifest, baselines, task_hashes):
    expected_profile = json.loads(json.dumps(asdict(stage5_spec())))
    if (
        manifest.get("stage") != "5.1"
        or manifest.get("physical_profile") != expected_profile
        or manifest.get("flow_penalty") != FLOW_CONFIG
        or manifest.get("development_task_sha256") != task_hashes
    ):
        raise ValueError("Incompatible cached physics, objective or task order")
    expected = {(i, m) for i in range(len(task_hashes)) for m in CLASSICAL}
    actual = [(r["task"], r["method"]) for r in baselines]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing or duplicated classical run")
    if any(
        r.get("evaluations") != 600
        or r.get("requested_evaluations") != 600
        or r.get("terminated_early") is not False
        for r in baselines
    ):
        raise ValueError("Cached classical budget is not complete600")


def verify_cached_control_sources(root, manifest):
    """Keep both numerical paths and shared neural/decoder dependencies sealed."""
    paths = list((root / "src/waveforge/lab/mcl").glob("*.py"))
    paths += [
        root / relative
        for relative in (
            "src/waveforge/experiments/audit_mcl_stage5_admission.py",
            "src/waveforge/ml/mt3_unet.py",
            "src/waveforge/design/parameterization.py",
            "src/waveforge/design/batched_parameterization.py",
            "src/waveforge/design/binary_readout.py",
            "src/waveforge/runtime.py",
            "src/waveforge/lab/contracts.py",
        )
    ]
    seal = {str(p): manifest["source_sha256"][str(p)] for p in paths}
    verify_seal(seal)
    return seal


def row_key(row):
    return row["task"], row["grid"], row["regime"], row["method"]


def merge_rows(reference_rows, neural_rows):
    original = {row_key(r): r for r in reference_rows}
    new = {row_key(r): r for r in neural_rows}
    wanted = {k for k, r in original.items() if r["method"] in NEURAL}
    if (
        len(original) != len(reference_rows)
        or len(new) != len(neural_rows)
        or set(new) != wanted
    ):
        raise ValueError("Missing, duplicate or unexpected audit row")
    return [copy.deepcopy(new.get(row_key(r), r)) for r in reference_rows]


def paired_table(rows, grid=128, regime="nominal"):
    lookup = {
        (r["task"], r["method"]): r
        for r in rows
        if r["grid"] == grid and r["regime"] == regime
    }
    table = []
    for i in sorted({k[0] for k in lookup}):
        selected = [lookup[i, m] for m in (*CLASSICAL, *NEURAL)]
        available = all(r["system"] is not None for r in selected)
        base = (
            min(r["system"]["system_peak_rise_k"] for r in selected[:2])
            if available
            else None
        )
        row = dict(
            task=i,
            comparison_available=available,
            best_classical_system_c=base + 25 if available else None,
        )
        for name, r in zip(("r0", "r25"), selected[2:], strict=True):
            value = r["system"]["system_peak_rise_k"] if available else None
            row.update(
                {
                    f"{name}_raw_c": r["silicon_peak_c"],
                    f"{name}_system_c": value + 25 if available else None,
                    f"{name}_gain_percent": 100 * (base - value) / base
                    if available
                    else None,
                    f"{name}_raw_admissible": r["admission"]["admissible"],
                    f"{name}_fallback": r["system"]["fallback_used"]
                    if available
                    else None,
                    f"{name}_re": r["diagnostics"]["max_local_reynolds_estimate"],
                }
            )
        table.append(row)
    return table


def audit_neural(task, index, chosen, grid, output, reference_rows):
    """Only new neural maps are solved. Uniform/classical rows are reused unchanged."""
    worker_init()
    problem = problem_from_task(task, stage5_spec())
    rows = []
    selected = [r for r in chosen if r["task"] == index]
    if {r["method"] for r in selected} != set(NEURAL) or len(selected) != 2:
        raise ValueError("Exactly two selected neural designs required")
    for item in selected:
        with np.load(item["path"], allow_pickle=False) as data:
            problem_grid, design = refine_problem(
                problem, data["binary"].copy(), grid // 64
            )
        nominal, fields = record(design, problem_grid)
        np.savez_compressed(
            Path(output) / f"dev_{index:02d}_{grid}_{item['method']}.npz",
            binary=design,
            temperature=fields.temperature,
            pressure=fields.pressure,
            qx=fields.qx,
            qy=fields.qy,
        )
        for regime in REGIMES:
            raw = (
                nominal
                if regime == "nominal"
                else record(
                    design, problem_grid, pressure_for(nominal["flow_ml_s"], regime)
                )[0]
            )
            uniform = next(
                r
                for r in reference_rows
                if r["task"] == index
                and r["grid"] == grid
                and r["regime"] == regime
                and r["method"] == "Uniform-Parallel"
            )
            available = uniform["admission"]["admissible"]
            rows.append(
                dict(
                    raw,
                    task=index,
                    grid=grid,
                    regime=regime,
                    method=item["method"],
                    comparison_available=available,
                    system=system(raw, uniform) if available else None,
                )
            )
    # The shared admission gate looks up Uniform when nominal selection failed.
    # Give it private copies without inserting duplicate controls into the audit.
    uniform_rows = [
        copy.deepcopy(r)
        for r in reference_rows
        if r["task"] == index
        and r["grid"] == grid
        and r["method"] == "Uniform-Parallel"
    ]
    apply_selection_gate(
        [*uniform_rows, *rows],
        {r["method"]: r["admissible_both"] for r in selected},
    )
    write_json(Path(output) / f"dev_{index:02d}_{grid}.json", rows)
    return rows


def write_report(output, rows):
    output = Path(output)
    summarize(output, rows)
    path = output / "ИТОГ_RU.md"
    body = path.read_text(encoding="utf-8")
    intro = (
        "# Stage 5.2 — старые веса с новым отбором и доработкой\n\n"
        "Сеть не переобучалась. Проверены все4 головы на64/128, затем только "
        "лучшая доработана25 шагами по неизменной процедуре Stage5.1. "
        "Adam/MMA600 и общий запасной Uniform взяты из проверенных сохранённых "
        "результатов Stage5.1. Нового штрафа за расход нет; штраф Re одинаковый.\n\n"
        "Физика и пороги прежние. Сетка128 участвует в отборе, поэтому это "
        "открытый development-контроль на8 задачах, а не независимый закрытый тест "
        "или подтверждение качества реального водоблока. Проценты считаются "
        "от перегрева над входной водой25°C.\n\n"
    )
    table = paired_table(rows)
    lines = [
        "## Главное сравнение на128×128, номинальный режим",
        "",
        "| Задача | Лучший Adam/MMA, °C | U-Net (наша модель) +R25, °C | Выигрыш |",
        "|---|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['task']} | {r['best_classical_system_c']:.3f} | "
            f"{r['r25_system_c']:.3f} | {r['r25_gain_percent']:+.3f}% |"
        )
    path.write_text(
        intro + "\n".join(lines) + "\n\n" + body[body.index("## Сетка") :],
        encoding="utf-8",
    )
    with (output / "paired_128_nominal.csv").open(
        "x", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def run(args):
    worker_init()
    if not torch.cuda.is_available():
        raise RuntimeError("Requested local CUDA GPU unavailable")
    root = Path(__file__).resolve().parents[3]
    source, output = Path(args.reference).resolve(), Path(args.output).resolve()
    receipt = read(source / "completion.json")
    if not receipt.get("completed") or receipt.get("smoke_only"):
        raise ValueError("Reference campaign incomplete")
    old_seal = {str(source / name): sha for name, sha in receipt["files"].items()}
    verify_seal(old_seal)
    manifest = read(source / "manifest.json")
    baselines = read(source / "baseline_results.json")
    tasks = tuple(task_from_dict(t) for t in read(args.selection)["tasks"][:8])
    validate_control(manifest, baselines, [t.fingerprint() for t in tasks])
    model_path = source / "old_model_frozen.pt"
    if digest(model_path) != OLD_SHA:
        raise ValueError("Original Stage5 model hash mismatch")
    unchanged = verify_cached_control_sources(root, manifest)
    paths = [
        *root.glob("src/waveforge/**/*.py"),
        root / "tests/test_mcl_stage52.py",
        root / "src/waveforge/experiments/run_mcl_stage52.py",
        Path(args.selection),
    ]
    seal = seal_files(paths)
    # Protect all inputs actually used, including the receipt itself.
    old_seal.update(seal_files([source / "completion.json"]))
    reference_rows = read(source / "STAGE5_1_AUDIT/results.json")
    expected_keys = {
        (i, g, r, m)
        for i in range(8)
        for g in (64, 128)
        for r in REGIMES
        for m in ("Uniform-Parallel", "Straight", *CLASSICAL, *NEURAL)
    }
    if (
        len(reference_rows) != 384
        or {row_key(r) for r in reference_rows} != expected_keys
    ):
        raise ValueError("Reference audit not a complete384-row panel")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    for p in paths:
        if p.is_relative_to(root):
            target = output / "SOURCE_SNAPSHOT" / p.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    shutil.copyfile(model_path, output / "model_frozen.pt")
    write_json(
        output / "manifest.json",
        dict(
            stage="5.2",
            reference=str(source),
            model_sha256=OLD_SHA,
            physical_profile=manifest["physical_profile"],
            flow_penalty=FLOW_CONFIG,
            retraining=False,
            training_updates=0,
            closed_test_accessed=False,
            development_task_sha256=[t.fingerprint() for t in tasks],
            head_count=4,
            refinement_steps=25,
            refined_heads_per_task=1,
            source_sha256=seal,
            cache_compatibility_sha256=unchanged,
            old_inputs_sha256=old_seal,
            selection=(
                "Same Stage51 choose(): both grids admissible, lowest nominal128 rise"
            ),
            fine_grid_used_for_selection=True,
            physical_validation=False,
            reused_classical_runs=16,
            rerun_classical_runs=0,
            q_penalty_added=False,
            device=torch.cuda.get_device_name(),
        ),
    )
    model = load_model(model_path, "cuda").eval()
    neural = []
    pending = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
        for i, task in enumerate(tasks):
            tick = time.perf_counter()
            condition, counts = condition_worker(task)
            with torch.inference_mode():
                params = (
                    model(torch.from_numpy(condition[None]).cuda())[0].cpu().numpy()
                )
            folder = output / f"dev_{i:02d}" / "Manifold-UNet-R0"
            pending[pool.submit(neural_task, task, params, folder)] = (
                i,
                counts,
                time.perf_counter() - tick,
            )
        for future in as_completed(pending):
            i, counts, seconds = pending[future]
            r0, r25 = future.result()
            if r25["evaluations"] != 25 or len(r0["heads"]) != 4:
                raise RuntimeError("Neural budget/head count incomplete")
            neural.append(
                dict(
                    task=i,
                    r0=r0,
                    r25=r25,
                    conditioning=counts,
                    conditioning_inference_seconds=seconds,
                )
            )
            print(
                f"NEURAL {i}: R0={r0['selected']['fine']['silicon_peak_c']:.3f}C "
                f"R25={r25['selected']['fine']['silicon_peak_c']:.3f}C "
                f"valid={r0['selected_admissible_both']}/{r25['selected_admissible_both']}",
                flush=True,
            )
        neural.sort(key=lambda r: r["task"])
        write_json(output / "neural_results.json", neural)
        chosen = []
        for n in neural:
            folder = output / f"dev_{n['task']:02d}" / "Manifold-UNet-R0"
            for key, method, sub in [
                ("r0", NEURAL[0], ""),
                ("r25", NEURAL[1], "R25-C"),
            ]:
                chosen.append(
                    dict(
                        task=n["task"],
                        method=method,
                        path=str(folder / sub / "selected.npz"),
                        admissible_both=n[key]["selected_admissible_both"],
                    )
                )
        write_json(output / "selection_audit.json", chosen)
        audit = output / "STAGE5_2_AUDIT"
        audit.mkdir()
        jobs = [
            pool.submit(audit_neural, task, i, chosen, grid, audit, reference_rows)
            for i, task in enumerate(tasks)
            for grid in (64, 128)
        ]
        new_rows = [r for j in jobs for r in j.result()]
    rows = merge_rows(reference_rows, new_rows)
    write_report(audit, rows)
    write_json(
        output / "paired_results.json",
        {
            f"{grid}_{regime}": paired_table(rows, grid, regime)
            for grid in (64, 128)
            for regime in REGIMES
        },
    )
    totals = dict(forward_calls=0, forward_rhs=0, adjoint_calls=0, adjoint_rhs=0)
    for n in neural:
        for counts in (n["conditioning"]["solver_counts"], n["r25"]["solver_counts"]):
            for k, v in counts.items():
                totals[k] += v
        for key in ("r0", "r25"):
            totals["forward_calls"] += n[key]["selection_forward_calls"]
            totals["forward_rhs"] += n[key]["selection_forward_rhs"]
    totals["forward_calls"] += 2 * len(new_rows)
    totals["forward_rhs"] += 4 * len(new_rows)
    write_json(
        output / "compute_accounting.json",
        dict(
            new_logical_solves=totals,
            convention="LU refinement corrections and tests excluded",
            reused_classical_runs=16,
            training_updates=0,
            speedup_claim=False,
            total_wall_seconds=time.perf_counter() - start,
        ),
    )
    verify_seal(seal)
    verify_seal(old_seal)
    write_json(
        output / "completion.json",
        dict(
            completed=True,
            old_weights_unchanged=True,
            old_results_unchanged=True,
            physics_changed=False,
            closed_test_accessed=False,
            retraining=False,
            elapsed_seconds=time.perf_counter() - start,
            files={
                str(p.relative_to(output)): digest(p)
                for p in output.rglob("*")
                if p.is_file()
            },
        ),
    )
    print("COMPLETED " + str(output), flush=True)


if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        default="artifacts/mcl_stage51_20261028",
    )
    parser.add_argument(
        "--selection",
        default="artifacts/inputs/development_tasks.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    run(parser.parse_args())
