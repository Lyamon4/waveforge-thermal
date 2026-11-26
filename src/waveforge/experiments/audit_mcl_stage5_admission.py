"""Repair omitted admission gates without changing sealed weights or physics.

Re-select from existing checkpoint receipts, then independently evaluate frozen
maps on64/128 grids. Old raw campaign remains immutable and visibly superseded.
"""

import argparse
import csv
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from waveforge.lab.mcl.baselines import digest, uniform_parallel
from waveforge.lab.mcl.contracts import problem_from_task, straight_channel
from waveforge.lab.mcl.high_fidelity_eval import refine_problem
from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init, write_json
from waveforge.lab.mcl.stage4_inventory import seal_files, verify_seal
from waveforge.lab.mcl.stage5_admission import admission, select, system
from waveforge.lab.mcl.stage5_evaluation import REGIMES, pressure_for
from waveforge.lab.mcl.stage5_physics import stage5_spec
from waveforge.lab.mcl.stage5_preflight import solve_profile

METHODS = ("Manifold-Adam", "Manifold-MMA", "Manifold-UNet-R0", "Manifold-UNet-R25-C")


def selection_receipts(source, output):
    chosen = []
    for i in range(8):
        for method in METHODS:
            sub = "Manifold-UNet-R0/R25-C" if method.endswith("R25-C") else method
            folder = source / f"dev_{i:02d}" / sub
            report = json.loads((folder / "result.json").read_text())
            candidates = report.get("checkpoints", report.get("heads"))
            best = select(candidates)
            if best is None:
                raise RuntimeError("No admissible nominal raw checkpoint")
            old = report["selected"]
            if method == "Manifold-UNet-R0" and best["file"] != old["file"]:
                raise RuntimeError(
                    "Admissible neural head changed: R25 must be rerun from new head"
                )
            target = output / "selected" / f"dev_{i:02d}" / method / "selected.npz"
            target.parent.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(folder / best["file"], target)
            chosen.append(
                dict(
                    task=i,
                    method=method,
                    path=str(target),
                    source=str(folder / best["file"]),
                    source_sha256=digest(folder / best["file"]),
                    selected_step=best["step"],
                    original_selected_step=old["step"],
                    selection_changed=best["file"] != old["file"],
                    rejected_checkpoints=[
                        dict(step=r["step"], file=r["file"], **admission(r))
                        for r in candidates
                        if not admission(r)["admissible"]
                    ],
                )
            )
    write_json(output / "selection_audit.json", chosen)
    return chosen


def record(design, problem, pressure=None):
    result = solve_profile(design, problem, k_tim=40.0, pressure_drop_pa=pressure)
    d = result.diagnostics
    s = problem.spec
    labels, n = label(design)
    common = np.intersect1d(
        labels[0, s.port_start : s.port_stop], labels[-1, s.port_start : s.port_stop]
    )
    geometry = n == 1 and np.any(common > 0) and design.sum() == s.nx * s.ny // 4
    row = dict(
        accepted=bool(geometry),
        peak_rise_k=float(result.scenario_peaks.max()),
        silicon_peak_c=float(result.scenario_peaks.max() + 25),
        flow_ml_s=result.flow_ml_s,
        pressure_pa=pressure or s.pressure_drop_pa,
        diagnostics=d,
    )
    row["admission"] = admission(row)
    return row, result


def evaluate_task(task, index, chosen, grid, output):
    worker_init()
    start = time.perf_counter()
    problem = problem_from_task(task, stage5_spec())
    designs = {
        "Uniform-Parallel": uniform_parallel(),
        "Straight": straight_channel(problem.spec),
    }
    for r in chosen:
        if r["task"] == index:
            with np.load(r["path"], allow_pickle=False) as data:
                designs[r["method"]] = data["binary"].copy()
    rows = []
    for method, original in designs.items():
        p, d = refine_problem(problem, original, grid // 64)
        nominal, fields = record(d, p)
        np.savez_compressed(
            Path(output) / f"dev_{index:02d}_{grid}_{method}.npz",
            binary=d,
            temperature=fields.temperature,
            pressure=fields.pressure,
            qx=fields.qx,
            qy=fields.qy,
        )
        for regime in REGIMES:
            dp = pressure_for(nominal["flow_ml_s"], regime)
            row = nominal if regime == "nominal" else record(d, p, dp)[0]
            rows.append(dict(row, task=index, method=method, regime=regime, grid=grid))
    for regime in REGIMES:
        uniform = next(
            r
            for r in rows
            if r["method"] == "Uniform-Parallel" and r["regime"] == regime
        )
        for row in rows:
            if row["regime"] != regime:
                continue
            row["comparison_available"] = uniform["admission"]["admissible"]
            row["system"] = (
                system(row, uniform) if row["comparison_available"] else None
            )
    write_json(Path(output) / f"dev_{index:02d}_{grid}.json", rows)
    print(f"AUDIT dev{index} grid{grid} {time.perf_counter() - start:.1f}s", flush=True)
    return rows


def summarize(output, rows):
    metrics = {}
    flat = []
    lines = [
        "# Stage 5 — итог с полными ограничениями",
        "",
        "Этот отчёт заменяет температурный рейтинг исходного run_001. "
        "Исходные данные и веса сохранены без изменений.",
        "",
        "Исправление: отбор обязан учитывать не только невязки и связность, "
        "но и Re < 1200, воду < 60 °C, кремний ≤ 95 °C. "
        "Балансы < 1e-10, невязки < 1e-12. "
        "Это допуск редуцированной модели, не физическая CFD-валидация.",
        "",
        "Номинальные кандидаты заново выбраны только среди ранее сохранённых "
        "точек того же бюджета. На сетке 128×128 каждая ячейка делится на 2×2 без "
        "повторной оптимизации, полная мощность сохраняется. "
        "Давление для заданного расхода/мощности насоса "
        "пересчитывается на каждой сетке.",
        "",
        "Проценты — уменьшение перегрева над входной водой 25 °C, "
        "не процент от температуры по шкале Цельсия. "
        "Положительный процент означает улучшение. "
        "Все 8 задач — открытые проверочные задачи (development), не закрытый тест.",
        "",
    ]
    for grid in (64, 128):
        for regime in REGIMES:
            lookup = {
                (r["task"], r["method"]): r
                for r in rows
                if r["grid"] == grid and r["regime"] == regime
            }
            metric = {}
            lines += [
                f"## Сетка {grid}×{grid}, режим {regime}",
                "",
                "| Метод | Допущено новых конструкций | Побед над Uniform "
                "без fallback | Медиана системы против Uniform |",
                "|---|---:|---:|---:|",
            ]
            for method in METHODS:
                rr = [lookup[i, method] for i in range(8)]
                valid = sum(r["admission"]["admissible"] for r in rr)
                available = all(r["comparison_available"] for r in rr)
                gains = (
                    [r["system"]["system_gain_percent"] for r in rr]
                    if available
                    else []
                )
                wins = (
                    sum(
                        r["admission"]["admissible"]
                        and r["peak_rise_k"]
                        < lookup[i, "Uniform-Parallel"]["peak_rise_k"] - 1e-6
                        for i, r in enumerate(rr)
                    )
                    if available
                    else None
                )
                metric[method] = dict(
                    admissible_count=valid,
                    task_count=8,
                    comparison_available=available,
                    raw_wins_vs_uniform=wins,
                    fallback_count=sum(r["system"]["fallback_used"] for r in rr)
                    if available
                    else None,
                    median_system_gain_vs_uniform=float(np.median(gains))
                    if available
                    else None,
                )
                gain_text = (
                    f"{np.median(gains):.3f}%"
                    if available
                    else "Нельзя сравнить: Uniform вне допуска"
                )
                wins_text = f"{wins}/8" if available else "Нет сравнения"
                lines.append(f"| {method} | {valid}/8 | {wins_text} | {gain_text} |")
            gains = []
            if all(r["comparison_available"] for r in lookup.values()):
                for i in range(8):
                    base = min(
                        lookup[i, m]["system"]["system_peak_rise_k"]
                        for m in METHODS[:2]
                    )
                    ours = lookup[i, METHODS[-1]]["system"]["system_peak_rise_k"]
                    gains.append(100 * (base - ours) / base)
            metric["unet_vs_best"] = dict(
                median_gain_percent=float(np.median(gains)) if gains else None,
                wins=sum(g > 1e-6 for g in gains),
                losses=sum(g < -1e-6 for g in gains),
                pairs=len(gains),
                per_task_gain_percent=gains,
            )
            text = (
                f"{np.median(gains):.3f}%, побед {sum(g > 1e-6 for g in gains)}/8"
                if gains
                else "не допущено к сравнению"
            )
            lines += [
                "",
                f"U-Net+R25-C против лучшей допустимой системы Adam/MMA: {text}.",
                "",
            ]
            metrics[f"{grid}_{regime}"] = metric
    for row in rows:
        flat.append(
            dict(
                task=row["task"],
                grid=row["grid"],
                regime=row["regime"],
                method=row["method"],
                silicon_c=row["silicon_peak_c"],
                water_c=row["diagnostics"]["peak_open_water_temperature_c"],
                re=row["diagnostics"]["max_local_reynolds_estimate"],
                pressure_pa=row["pressure_pa"],
                flow_ml_s=row["flow_ml_s"],
                admissible=row["admission"]["admissible"],
                reasons=";".join(row["admission"]["rejection_reasons"]),
                system_peak_k=row["system"]["system_peak_rise_k"]
                if row["system"]
                else None,
                fallback_used=row["system"]["fallback_used"] if row["system"] else None,
            )
        )
    write_json(output / "metrics.json", metrics)
    write_json(output / "results.json", rows)
    with (output / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    (output / "ИТОГ_RU.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    root = Path(__file__).resolve().parents[3]
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads((source / "manifest.json").read_text())
    receipt = json.loads((source / "completion.json").read_text())
    verify_seal(old["source_input_sha256"])
    verify_seal(old["old_artifacts_sha256"])
    verify_seal({str(source / k): v for k, v in receipt["files"].items()})
    start = time.perf_counter()
    selection = Path(
        "artifacts/inputs/development_tasks.json"
    )
    tasks = [task_from_dict(t) for t in json.loads(selection.read_text())["tasks"][:8]]
    assert [t.fingerprint() for t in tasks] == old["development_task_sha256"]
    files = [
        *(root / "src/waveforge").rglob("*.py"),
        root / "tests/test_mcl_stage5_admission.py",
        selection,
        source / "completion.json",
    ]
    seal = seal_files(files)
    write_json(
        output / "manifest.json",
        dict(
            stage="5-final-admission-audit",
            source_sha256=seal,
            inherited_gates=dict(
                re_lt=1200,
                water_c_lt=60,
                silicon_c_le=95,
                residual_lt=1e-12,
                balance_lt=1e-10,
            ),
            original_run=str(source),
            old_weights_sha256=digest(source / "model_frozen.pt"),
            closed_test_accessed=False,
            retraining=False,
            physics_changed=False,
        ),
    )
    chosen = selection_receipts(source, output)
    rows = []
    with ProcessPoolExecutor(max_workers=3, initializer=worker_init) as pool:
        pending = [
            pool.submit(evaluate_task, t, i, chosen, g, output)
            for g in (64, 128)
            for i, t in enumerate(tasks)
        ]
        for f in as_completed(pending):
            rows.extend(f.result())
    summarize(output, rows)
    verify_seal(seal)
    verify_seal(old["source_input_sha256"])
    verify_seal(old["old_artifacts_sha256"])
    verify_seal({str(source / k): v for k, v in receipt["files"].items()})
    write_json(
        output / "completion.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            files={
                str(p.relative_to(output)): digest(p)
                for p in output.rglob("*")
                if p.is_file()
            },
            old_results_unchanged=True,
            closed_test_accessed=False,
        ),
    )
    print("ADMISSION_AUDIT_COMPLETED", flush=True)


if __name__ == "__main__":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())
