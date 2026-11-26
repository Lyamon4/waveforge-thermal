"""Immutable Stage5.1 local campaign: matched controls, fine tuning and mesh audit."""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import torch

from waveforge.experiments.audit_mcl_stage5_admission import evaluate_task as raw_audit
from waveforge.experiments.audit_mcl_stage5_admission import summarize
from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.stage3_training import (
    protected_geometries,
    task_from_dict,
    worker_init,
    write_json,
)
from waveforge.lab.mcl.stage4_inventory import seal_files, verify_seal
from waveforge.lab.mcl.stage5_physics import stage5_spec
from waveforge.lab.mcl.stage5_qualification import qualify
from waveforge.lab.mcl.stage5_training import condition_worker
from waveforge.lab.mcl.stage51 import (
    FLOW_CONFIG,
    apply_selection_gate,
    require_complete_budget,
)
from waveforge.lab.mcl.stage51_evaluation import neural_task, optimize
from waveforge.lab.mcl.stage51_training import CONFIG, train

OLD = Path("artifacts/mcl_stage5_20261018")
OLD_MODEL_SHA = "95701d5448a5ad6845ef2851ed0bf3bd11399cbb6908e9de5ff492cbd068b344"


def evaluate_task(task, index, chosen, grid, output):
    rows = raw_audit(task, index, chosen, grid, output)
    status = {r["method"]: r["admissible_both"] for r in chosen if r["task"] == index}
    apply_selection_gate(rows, status)
    write_json(Path(output) / f"dev_{index:02d}_{grid}.json", rows, exclusive=False)
    return rows


def write_audit_report(audit, rows, *, smoke):
    audit = Path(audit)
    if smoke:
        write_json(audit / "results.json", rows)
        return
    summarize(audit, rows)
    report_path = audit / "ИТОГ_RU.md"
    original = report_path.read_text(encoding="utf-8")
    body = original[original.index("## Сетка") :]
    report_path.write_text(
        "# Stage 5.1 — фактический результат\n\n"
        "Физика, геометрия и пороги Stage5 сохранены. Старые веса не изменены. "
        "U-Net дообучена с таким же штрафом за поток, как Adam/MMA. "
        "Все методы отбирают конструкции по прямым проверкам на ОБЕИХ сетках.\n\n"
        "Сетка128 участвует в отборе: "
        "она НЕ является независимым финальным тестом. "
        "Все8 задач открытые; закрытые тесты не открывались. Это редуцированная "
        "модель, не разрешённое CFD и не испытание реального водоблока.\n\n"
        "Проценты означают уменьшение перегрева над входной водой25°C. "
        "Результаты с общим запасным Uniform и без него приведены отдельно.\n\n" + body,
        encoding="utf-8",
    )


def run(args):
    worker_init()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Requested RTX4060 unavailable; do not silently change device"
        )
    root = Path(__file__).resolve().parents[3]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    old_model = OLD / "model_frozen.pt"
    if digest(old_model) != OLD_MODEL_SHA:
        raise RuntimeError("Stage5 source weight hash mismatch")
    old_seal = seal_files(
        [
            old_model,
            *OLD.glob("TRAINING_AND_RAW_RUN/*.pt"),
            OLD / "FINAL_AUDIT/comparison.csv",
            OLD / "FINAL_AUDIT/metrics.json",
        ]
    )
    shutil.copyfile(old_model, output / "old_model_frozen.pt")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    all_tasks = tuple(task_from_dict(t) for t in selection["tasks"])
    tasks = all_tasks[: 1 if args.smoke else 8]
    old_manifest = json.loads((OLD / "TRAINING_AND_RAW_RUN/manifest.json").read_text())
    if [t.fingerprint() for t in all_tasks[:8]] != old_manifest[
        "development_task_sha256"
    ]:
        raise RuntimeError("Development tasks differ from sealed Stage5")
    protected, hygiene = protected_geometries(
        root / "artifacts/epyc_v2_20260923/splits", all_tasks
    )
    excluded = {
        json.loads(line)["sha256"]
        for line in (OLD / "TRAINING_AND_RAW_RUN/training_tasks.jsonl")
        .read_text()
        .splitlines()
    }
    protocol = root / "src/waveforge/experiments/run_mcl_stage51.py"
    paths = list((root / "src/waveforge").rglob("*.py"))
    paths += [*(root / "tests").glob("test_mcl*.py"), protocol]
    paths += [
        Path(args.selection),
        root / "artifacts/epyc_v2_20260923/splits/manifest.json",
        root / "artifacts/epyc_v2_20260923/splits/geometry_exclusions.json",
    ]
    source_seal = seal_files(paths)
    for path in paths:
        if path.is_relative_to(root):
            dst = output / "SOURCE_SNAPSHOT" / path.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
    steps, batch = (3, 1) if args.smoke else (240, 4)
    write_json(
        output / "manifest.json",
        dict(
            stage="5.1",
            smoke_only=args.smoke,
            physical_profile=asdict(stage5_spec()),
            tim_k=40.0,
            protocol_sha256=digest(protocol),
            source_sha256=source_seal,
            old_artifacts_sha256=old_seal,
            old_model_sha256=OLD_MODEL_SHA,
            old_training_task_hashes_excluded=len(excluded),
            training=dict(CONFIG, steps=steps, batch=batch),
            flow_penalty=FLOW_CONFIG,
            device=torch.cuda.get_device_name(),
            physics_device="cpu_sparse_LU",
            development_task_sha256=[t.fingerprint() for t in tasks],
            data_hygiene=hygiene,
            selection="admissible on64 AND128, lowest nominal128 peak, earlier step",
            fine_mesh_used_in_selection=True,
            physical_validation=False,
            closed_test_accessed=False,
        ),
    )
    q = qualify(tasks[0])
    write_json(output / "qualification.json", q)
    if not q["passed"]:
        raise RuntimeError("Physics qualification failed")
    print(
        "QUALIFICATION PASS; network=RTX4060/CUDA; physics=persistent CPU workers",
        flush=True,
    )
    timings = {}
    baselines, neural = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
        base_start = time.perf_counter()
        pending = {
            pool.submit(
                optimize,
                t,
                method,
                output / f"dev_{i:02d}" / method,
                steps=3 if args.smoke else 600,
            ): (i, method)
            for i, t in enumerate(tasks)
            for method in ("Manifold-Adam", "Manifold-MMA")
        }
        for f in as_completed(pending):
            i, method = pending[f]
            r = f.result()
            require_complete_budget(r, 3 if args.smoke else 600)
            baselines.append(dict(task=i, **r))
            print(
                f"BASE {i} {method}:128 "
                f"T={r['selected']['fine']['silicon_peak_c']:.3f}C "
                f"both_valid={r['selected_admissible_both']}",
                flush=True,
            )
        timings["baselines_wall_seconds"] = time.perf_counter() - base_start
        write_json(output / "baseline_results.json", baselines)
        verify_seal(source_seal)
        verify_seal(old_seal)
        model, training = train(
            pool,
            output,
            protected,
            initial_checkpoint=old_model,
            excluded_hashes=excluded,
            steps=steps,
            batch=batch,
            device="cuda",
        )
        timings["training_seconds"] = training["elapsed_seconds"]
        chosen = []
        for r in baselines:
            chosen.append(
                dict(
                    task=r["task"],
                    method=r["method"],
                    admissible_both=r["selected_admissible_both"],
                    path=str(
                        output / f"dev_{r['task']:02d}" / r["method"] / "selected.npz"
                    ),
                )
            )
        for i, task in enumerate(tasks):
            inf_start = time.perf_counter()
            condition, counts = condition_worker(task)
            with torch.no_grad():
                params = (
                    model(torch.from_numpy(condition[None]).cuda())[0].cpu().numpy()
                )
            seconds = time.perf_counter() - inf_start
            folder = output / f"dev_{i:02d}" / "Manifold-UNet-R0"
            r0, r25 = neural_task(task, params, folder)
            neural.append(
                dict(
                    task=i,
                    r0=r0,
                    r25=r25,
                    conditioning=counts,
                    conditioning_inference_seconds=seconds,
                )
            )
            chosen += [
                dict(
                    task=i,
                    method="Manifold-UNet-R0",
                    path=str(folder / "selected.npz"),
                    admissible_both=r0["selected_admissible_both"],
                ),
                dict(
                    task=i,
                    method="Manifold-UNet-R25-C",
                    path=str(folder / "R25-C/selected.npz"),
                    admissible_both=r25["selected_admissible_both"],
                ),
            ]
            print(
                f"NEURAL {i}: R0={r0['selected']['fine']['silicon_peak_c']:.3f}C "
                f"R25={r25['selected']['fine']['silicon_peak_c']:.3f}C "
                f"both_valid={r25['selected_admissible_both']}",
                flush=True,
            )
        write_json(output / "neural_results.json", neural)
        write_json(output / "selection_audit.json", chosen)
        audit = output / "STAGE5_1_AUDIT"
        audit.mkdir()
        jobs = [
            pool.submit(evaluate_task, task, i, chosen, grid, audit)
            for i, task in enumerate(tasks)
            for grid in (64, 128)
        ]
        rows = [row for job in jobs for row in job.result()]
    write_audit_report(audit, rows, smoke=args.smoke)
    # Logical linear solves: one pressure RHS and three thermal RHS per forward.
    totals = {
        key: training["conditioning_solver_counts"][key]
        + training["objective_solver_counts"][key]
        for key in training["objective_solver_counts"]
    }
    for r in [*baselines, *(n["r25"] for n in neural)]:
        for key, value in r["solver_counts"].items():
            totals[key] += value
    selection_calls = sum(r["selection_forward_calls"] for r in baselines)
    selection_rhs = sum(r["selection_forward_rhs"] for r in baselines)
    for n in neural:
        for key, value in n["conditioning"]["solver_counts"].items():
            totals[key] += value
        for variant in ("r0", "r25"):
            selection_calls += n[variant]["selection_forward_calls"]
            selection_rhs += n[variant]["selection_forward_rhs"]
    totals["forward_calls"] += selection_calls + 2 * len(rows)
    totals["forward_rhs"] += selection_rhs + 4 * len(rows)
    # qualify(): two reference+torch pairs, one torch and nine reference probes.
    # Thermal transpose:3 RHS; pressure thermal/flow adjoints:1 RHS each.
    qualification_counts = dict(
        forward_calls=28, forward_rhs=56, adjoint_calls=3, adjoint_rhs=5
    )
    for key, value in qualification_counts.items():
        totals[key] += value
    write_json(
        output / "compute_accounting.json",
        dict(
            logical_systems=totals,
            selection_forward_calls=selection_calls,
            selection_forward_rhs=selection_rhs,
            audit_forward_calls=2 * len(rows),
            audit_forward_rhs=4 * len(rows),
            qualification=qualification_counts,
            convention=(
                "logical solves; iterative refinement corrections internal "
                "to LU excluded; tests excluded"
            ),
            hardware="RTX4060 network; CPU sparse physics; concurrent tasks",
            speedup_claim=False,
        ),
    )
    timings["total_wall_seconds"] = time.perf_counter() - start
    write_json(output / "timings.json", timings)
    verify_seal(source_seal)
    verify_seal(old_seal)
    write_json(
        output / "completion.json",
        dict(
            completed=True,
            smoke_only=args.smoke,
            old_weights_unchanged=True,
            physics_source_unchanged=True,
            closed_test_accessed=False,
            files={
                str(p.relative_to(output)): digest(p)
                for p in output.rglob("*")
                if p.is_file()
            },
            elapsed_seconds=time.perf_counter() - start,
        ),
    )
    print("COMPLETED " + str(output), flush=True)


if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    parser.add_argument(
        "--selection",
        default="artifacts/inputs/development_tasks.json",
    )
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
