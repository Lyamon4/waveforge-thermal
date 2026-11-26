"""Stage5 qualification, fresh training and matched open-development controls."""

import argparse
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

from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.stage3_training import (
    NORMALIZATION,
    protected_geometries,
    task_from_dict,
    worker_init,
    write_json,
)
from waveforge.lab.mcl.stage4_inventory import load_inventory, seal_files, verify_seal
from waveforge.lab.mcl.stage5_evaluation import (
    REGIMES,
    audit_regimes,
    neural_task,
    optimize,
)
from waveforge.lab.mcl.stage5_physics import stage5_spec
from waveforge.lab.mcl.stage5_qualification import benchmark_neural, qualify
from waveforge.lab.mcl.stage5_training import CONFIG, condition_worker, train


def summarize(output, rows, timings, *, smoke=False):
    output = Path(output)
    write_json(output / "all_results.json", rows)
    flat = []
    metrics = {}
    methods = (
        "Manifold-Adam",
        "Manifold-MMA",
        "Manifold-UNet-R0",
        "Manifold-UNet-R25-C",
    )
    task_count = len({row["task"] for row in rows})
    budget = 3 if smoke else 600
    for row in rows:
        flat.append(
            dict(
                task=row["task"],
                regime=row["regime"],
                method=row["method"],
                peak_c=row["silicon_peak_c"],
                flow_ml_s=row["flow_ml_s"],
                pressure_pa=row["pressure_pa"],
                silicon_le95=row["silicon_le95"],
                local_re=row["diagnostics"]["max_local_reynolds_estimate"],
                **row["fallback"],
            )
        )
    with (output / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# Stage5 — результаты на открытых задачах",
        "",
        "Исследовательская редуцированная модель, "
        "не CFD и не испытание промышленного водоблока.",
        "Все температуры — кремний; инженерный ориентир 95 °C. "
        "Проценты рассчитаны по перегреву над водой 25 °C.",
        "Закрытые тесты не открывались. Итоговые веса выбраны заранее: "
        f"последний шаг обучения, без выбора по этим {task_count} задачам.",
        "ТЕХНИЧЕСКИЙ SMOKE, НЕ НАУЧНЫЙ РЕЗУЛЬТАТ."
        if smoke
        else "Полный плановый прогон.",
        "",
    ]
    for regime in REGIMES:
        lookup = {(r["task"], r["method"]): r for r in rows if r["regime"] == regime}
        tasks = sorted({t for t, m in lookup})
        metric = {}
        lines += [
            f"## Режим {regime}",
            "",
            "| Метод | Медианный выигрыш новых каналов над Uniform | "
            "Строго лучше Uniform | Медианный выигрыш системы с запасным Uniform |",
            "|---|---:|---:|---:|",
        ]
        for method in methods:
            selected = [lookup[t, method] for t in tasks]
            raw = [r["fallback"]["raw_gain_percent"] for r in selected]
            system = [r["fallback"]["system_gain_percent"] for r in selected]
            wins = sum(
                r["accepted"]
                and r["peak_rise_k"]
                < lookup[t, "Uniform-Parallel"]["peak_rise_k"] - 1e-6
                for t, r in zip(tasks, selected, strict=True)
            )
            metric[method] = dict(
                median_raw_gain_percent=float(np.median(raw)),
                median_system_gain_percent=float(np.median(system)),
                strict_wins_vs_uniform=wins,
                task_count=len(tasks),
                raw_invalid=sum(not r["accepted"] for r in selected),
                raw_over95=sum(not r["silicon_le95"] for r in selected),
            )
            lines.append(
                f"| {method} | {np.median(raw):.3f}% | {wins}/{len(tasks)} | "
                f"{np.median(system):.3f}% |"
            )
        gains = []
        for t in tasks:
            ref = min(
                lookup[t, m]["fallback"]["system_peak_rise_k"]
                for m in ("Manifold-Adam", "Manifold-MMA")
            )
            ours = lookup[t, "Manifold-UNet-R25-C"]["fallback"]["system_peak_rise_k"]
            gains.append(100 * (ref - ours) / ref)
        metric["unet_system_vs_best_classical_system"] = dict(
            median_gain_percent=float(np.median(gains)), per_task_gain_percent=gains
        )
        metrics[regime] = metric
        lines += [
            "",
            "Система U-Net+R25-C против лучшего из двух оптимизаторов "
            f"с тем же запасным Uniform: медиана {np.median(gains):.3f}%.",
            "",
        ]
    lines += [
        "## Что именно измерено",
        "",
        "- Новая конструкция и система с общим запасным Uniform показаны раздельно.",
        "- В других режимах пересчитаны замороженные конструкции; "
        "повторного обучения или оптимизации под них нет.",
        f"- Сравнение классических методов — по {budget} обращениям к целевой функции; "
        "MMA может закончить раньше, фактическое число сохранено.",
        "- Тайминги сохранены, но параллельный запуск не даёт права объявлять "
        "точное ускорение на одинаковом выделенном оборудовании.",
        "- Порог Reynolds здесь диагностический; это не разрешённое CFD-течение "
        "и не доказательство отсутствия локальной турбулентности.",
        "- Микроширина280мкм и теплоотдача25830 — параметры редуцированной модели, "
        "а не разрешённые сеткой микроканалы.",
        "- Никакой обещанный процент выигрыша не использовался для подгонки физики.",
    ]
    write_json(output / "metrics.json", metrics)
    write_json(output / "timings.json", timings)
    (output / "SUMMARY_RU.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    root = Path(__file__).resolve().parents[3]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    worker_init()
    tasks, _, old_seal = load_inventory(root, args.selection)
    selection_tasks = tuple(
        task_from_dict(t) for t in json.loads(Path(args.selection).read_text())["tasks"]
    )
    protected, hygiene = protected_geometries(
        root / "artifacts/epyc_v2_20260923/splits", selection_tasks
    )
    # Seal all package sources, including transitive imports, not only new modules.
    paths = list((root / "src/waveforge").rglob("*.py"))
    paths += list((root / "tests").glob("test_mcl*.py"))
    paths += [
        Path(args.selection),
        root / "artifacts/epyc_v2_20260923/splits/manifest.json",
        root / "artifacts/epyc_v2_20260923/splits/geometry_exclusions.json",
        root / "src/waveforge/experiments/run_mcl_stage5.py",
    ]
    seal = seal_files(paths)
    steps, batch = (3, 1) if args.smoke else (800, 4)
    manifest = dict(
        stage="5",
        smoke_only=args.smoke,
        physical_profile=asdict(stage5_spec()),
        tim_k=40.0,
        contact_each=2e-6,
        physical_validation=False,
        source_input_sha256=seal,
        old_artifacts_sha256=old_seal,
        data_hygiene=hygiene,
        development_task_sha256=[t.fingerprint() for t in tasks],
        training=dict(CONFIG, steps=steps, batch=batch),
        normalization=NORMALIZATION,
        common_fallback="sealed Uniform-Parallel for every method",
        closed_test_accessed=False,
    )
    write_json(output / "manifest.json", manifest)
    for path in paths:
        if path.is_relative_to(root):
            dst = output / "SOURCE_SNAPSHOT" / path.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
    qualification = qualify(tasks[0])
    write_json(output / "qualification.json", qualification)
    if not qualification["passed"]:
        raise RuntimeError("Stage5 qualification failed; no training")
    bench = benchmark_neural()
    write_json(output / "runtime_benchmark.json", bench)
    print(
        "QUALIFICATION PASS; network device=" + bench["selected_neural_device"],
        flush=True,
    )
    if args.qualify_only:
        return
    verify_seal(seal)
    verify_seal(old_seal)
    if args.smoke:
        tasks = tasks[:1]
    timings = {}
    baseline_results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
        # Baselines first, then training: no oversubscription of sparse physics CPUs.
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
        for future in as_completed(pending):
            i, method = pending[future]
            report = future.result()
            baseline_results.append(dict(task=i, **report))
            print(
                f"BASELINE {i} {method}: "
                f"raw={report['selected']['silicon_peak_c']:.3f}C "
                f"fallback={report['fallback']['fallback_used']}",
                flush=True,
            )
        timings["baselines_wall_seconds"] = time.perf_counter() - base_start
        verify_seal(seal)
        verify_seal(old_seal)
        model, training = train(
            pool,
            output,
            protected,
            steps=steps,
            batch=batch,
            device=bench["selected_neural_device"],
        )
        timings["training_seconds"] = training["elapsed_seconds"]
        verify_seal(seal)
        verify_seal(old_seal)
        neural = []
        for i, t in enumerate(tasks):
            inference_start = time.perf_counter()
            condition, counter = condition_worker(t)
            with torch.no_grad():
                device = next(model.parameters()).device
                params = (
                    model(torch.from_numpy(condition[None]).to(device))[0].cpu().numpy()
                )
            conditioning_inference_seconds = time.perf_counter() - inference_start
            r0, r25 = neural_task(
                t, params, output / f"dev_{i:02d}" / "Manifold-UNet-R0"
            )
            neural.append(
                dict(
                    task=i,
                    r0=r0,
                    r25=r25,
                    conditioning=counter,
                    conditioning_inference_seconds=conditioning_inference_seconds,
                )
            )
            print(
                f"NEURAL {i}: R0={r0['selected']['silicon_peak_c']:.3f}C "
                f"R25={r25['selected']['silicon_peak_c']:.3f}C",
                flush=True,
            )
        write_json(output / "neural_results.json", neural)
        rows = []
        for i, t in enumerate(tasks):
            folder = output / f"dev_{i:02d}"
            paths = {
                method: folder / method / "selected.npz"
                for method in ("Manifold-Adam", "Manifold-MMA", "Manifold-UNet-R0")
            }
            paths["Manifold-UNet-R25-C"] = (
                folder / "Manifold-UNet-R0/R25-C/selected.npz"
            )
            rows.extend(
                dict(r, task=i)
                for r in audit_regimes(t, paths, folder / "operating_regimes.json")
            )
    write_json(output / "baseline_results.json", baseline_results)
    timings["total_wall_seconds"] = time.perf_counter() - start
    summarize(output, rows, timings, smoke=args.smoke)
    verify_seal(seal)
    verify_seal(old_seal)
    files = {
        str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()
    }
    write_json(
        output / "completion.json",
        dict(
            completed=True,
            smoke_only=args.smoke,
            files=files,
            source_inputs_unchanged=True,
            old_results_unchanged=True,
            closed_test_accessed=False,
            elapsed_seconds=time.perf_counter() - start,
        ),
    )
    print("COMPLETED " + str(output), flush=True)


if __name__ == "__main__":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--selection",
        default="artifacts/inputs/development_tasks.json",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be1..4")
    run(args)
