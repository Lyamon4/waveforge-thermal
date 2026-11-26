"""Package measured solver-free weights, full results and readable PNG figures."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from waveforge.lab.mcl.baselines import digest
from waveforge.lab.mcl.locked_evaluation import write

ROOT = Path(__file__).resolve().parents[1]


def plots(output, table, summary, benchmark, tasks, binary):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, layout="constrained")
    x = np.arange(1, len(table) + 1)
    for ax, mode, title in zip(
        axes,
        ("nominal", "power"),
        ("Одинаковое давление", "Одинаковая мощность насоса"),
        strict=True,
    ):
        values = table[mode + "_gap_percent"].to_numpy()
        ax.bar(x, values, color=np.where(values <= 0, "#2d8276", "#d48435"), width=0.75)
        ax.axhline(0, color="#26313c", lw=1)
        ax.axhline(np.median(values), color="#26313c", ls="--", lw=1.5)
        ax.set_title(
            f"{title}: медианная потеря {np.median(values):.2f}%",
            loc="left",
            fontsize=18,
        )
        ax.set_ylabel("Изменение перегрева, %")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(x[::2])
    axes[-1].set_xlabel("Новая проверочная задача")
    fig.suptitle(
        "Быстрый режим: качество близкое, но потери ещё есть",
        fontsize=23,
        fontweight="bold",
    )
    fig.supxlabel(
        "32 новые задачи · положительный столбик — хуже прежней системы · "
        "проверка SciPy 128×128",
        fontsize=12,
    )
    fig.savefig(output / "01_КАЧЕСТВО_32_ЗАДАЧИ.png", dpi=180)
    plt.close(fig)

    pairs = benchmark["pairs"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    colours = {
        "U-Net (наша модель)": "#2474b5",
        "Adam600": "#8b8d91",
        "MMA600": "#d48435",
    }
    for offset, (name, colour) in zip((-0.22, 0, 0.22), colours.items(), strict=True):
        values = []
        for p in pairs:
            seconds = p["neural_ms"] / 1000
            values.append(
                seconds if name.startswith("U-Net") else seconds * p["speedups"][name]
            )
        positions = np.arange(3) + offset
        axes[0].scatter(values, positions, s=75, label=name, color=colour)
        for value, y in zip(values, positions, strict=True):
            label = f"{value * 1000:.0f} мс" if value < 1 else f"{value:.1f} с"
            axes[0].annotate(
                label,
                (value, y),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=11,
            )
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.01, 210)
    axes[0].set_yticks(range(3), [f"Задача {i + 1}" for i in range(3)])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Время, с · логарифмическая шкала")
    axes[0].set_title("Скорость полного быстрого запуска", fontsize=17)
    axes[0].legend(loc="lower left", fontsize=11)
    axes[0].grid(axis="x", alpha=0.2)
    gains = np.array([p["nominal_gain_percent"] for p in pairs])
    axes[1].bar(range(1, 4), gains, color=np.where(gains >= 0, "#2d8276", "#d48435"))
    axes[1].axhline(0, color="#26313c", lw=1)
    axes[1].set_xticks(range(1, 4), ["Задача 1", "Задача 2", "Задача 3"])
    axes[1].set_ylabel("Снижение перегрева, %")
    axes[1].set_title("Качество против лучшего Adam/MMA", fontsize=17)
    for i, gain in enumerate(gains, 1):
        axes[1].text(
            i,
            gain + (0.2 if gain >= 0 else -0.2),
            f"{gain:+.2f}%",
            ha="center",
            va="bottom" if gain >= 0 else "top",
            fontsize=15,
        )
    axes[1].set_ylim(min(gains) - 1.5, max(gains) + 1.5)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_axisbelow(True)
    fig.suptitle(
        "Ускорение есть. Равное качество во всех задачах пока не достигнуто",
        fontsize=20,
        fontweight="bold",
    )
    fig.supxlabel(
        "3 задачи · прогретые программы · загрузка, сохранение и независимая "
        "проверка исключены у всех методов",
        fontsize=11,
    )
    fig.savefig(output / "02_СКОРОСТЬ_И_КАЧЕСТВО.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    ax.imshow(
        binary.T,
        origin="lower",
        extent=(0, 75.4, 0, 72),
        interpolation="nearest",
        cmap=ListedColormap(["#f1f3f5", "#2474b5"]),
    )
    number = 0
    for chip in tasks[0]["chiplets"]:
        xx, yy = chip["x"] * 75.4 / 64, chip["y"] * 72 / 64
        width, height = chip["width"] * 75.4 / 64, chip["height"] * 72 / 64
        ax.add_patch(
            Rectangle((xx, yy), width, height, fill=False, edgecolor="#d34a3a", lw=1.8)
        )
        if chip["width"] == 8:
            number += 1
            label = f"CCD {number}"
        else:
            label = "I/O"
        ax.text(
            xx + width / 2,
            yy + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
        )
    ax.annotate(
        "Вход воды",
        xy=(0, 36),
        xytext=(-2, 36),
        ha="right",
        va="center",
        fontsize=13,
        color="#2474b5",
        arrowprops=dict(arrowstyle="->", color="#2474b5"),
    )
    ax.annotate(
        "Выход",
        xy=(75.4, 36),
        xytext=(79, 36),
        ha="left",
        va="center",
        fontsize=13,
        color="#2474b5",
        arrowprops=dict(arrowstyle="<-", color="#2474b5"),
    )
    ax.set_xlim(-15, 90)
    ax.set_ylim(0, 72)
    ax.set_xlabel("мм")
    ax.set_ylabel("мм")
    ax.set_title(
        "Одна конструкция быстрого режима\n"
        "Синие области — вода; красные рамки — чиплеты",
        fontsize=18,
        pad=15,
    )
    fig.supxlabel(
        "Реальная карта задачи 1 · каналы занимают 25% площади · расчётная геометрия",
        fontsize=12,
    )
    fig.savefig(output / "03_ГОТОВАЯ_КОНСТРУКЦИЯ.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(args):
    index = json.loads((ROOT / "models/index.json").read_text())
    old = [(entry["file"], entry["sha256"]) for entry in index["models"]]
    for filename, sha in old:
        if digest(ROOT / "models" / filename) != sha:
            raise RuntimeError(f"old weights changed: {filename}")
    args.output.mkdir(parents=True, exist_ok=False)
    weights = args.output / "ВЕСА"
    results = args.output / "РЕЗУЛЬТАТЫ"
    images = args.output / "КАРТИНКИ"
    for directory in (weights, results, images):
        directory.mkdir()
    for filename in ("model_frozen.pt", "selector.joblib"):
        shutil.copy2(args.checkpoint.parent / filename, weights / filename)
    for filename in ("comparison.csv", "summary.json"):
        shutil.copy2(args.audit / filename, results / filename)
    shutil.copy2(args.fresh / "tasks.json", results / "tasks.json")
    shutil.copy2(
        args.fresh / "frozen_before_sampling.json",
        results / "frozen_before_sampling.json",
    )
    shutil.copytree(args.benchmark, results / "benchmark")
    for directory in sorted(args.audit.glob("case_*")):
        target = results / directory.name
        target.mkdir()
        for filename in ("selected.npz", "result.json"):
            shutil.copy2(directory / filename, target / filename)
    summary = json.loads((args.audit / "summary.json").read_text())
    benchmark = json.loads((args.benchmark / "summary.json").read_text())
    table = pd.read_csv(args.audit / "comparison.csv")
    tasks = json.loads((args.fresh / "tasks.json").read_text())["tasks"]
    with np.load(args.audit / "case_000/selected.npz") as saved:
        binary = saved["binary"]
    plots(images, table, summary, benchmark, tasks, binary)
    report = dict(
        model_sha256=digest(weights / "model_frozen.pt"),
        selector_sha256=digest(weights / "selector.joblib"),
        online_physical_solves=0,
        output="one binary 64x64 channel map, exactly 1024 fluid cells",
        architecture=(
            "raw-input U-Net bridge, frozen eight-proposal U-Net portfolio, "
            "learned tree selector"
        ),
        proposal_count=8,
        input="three 360W workloads, eight CCD regions plus I/O; seven raw input maps",
        training=dict(
            raw_adapter_layouts=1280, additional_updates=4000, teacher_free=False
        ),
        physics_unchanged=True,
        old_weights_preserved=True,
        paid_gpu_used=False,
        timings=summary["timing"],
        quality={mode: summary[mode] for mode in ("nominal", "power")},
        benchmark_three_tasks=benchmark["pairs"],
        strict_no_quality_loss=False,
        near_quality_gate_passed=False,
        status="experimental: speed target reached, quality target not reached",
        startup_excluded_from_warm_timings=True,
        online_physical_certification=False,
        tests=dict(passed=381, skipped=7),
        official_test_workloads_opened=False,
    )
    write(args.output / "ИТОГИ.json", report)
    model_target = ROOT / "models/solver_free_20261030"
    result_target = ROOT / "results/solver_free_20261030"
    shutil.copytree(weights, model_target)
    shutil.copytree(results, result_target)
    shutil.copytree(images, result_target / "images")
    shutil.copy2(args.output / "ИТОГИ.json", result_target / "result.json")
    index["models"].append(
        dict(
            file="solver_free_20261030/model_frozen.pt",
            architecture="GeneratorOnlyPolicy",
            sha256=digest(model_target / "model_frozen.pt"),
            selector_file="selector.joblib",
            selector_sha256=digest(model_target / "selector.joblib"),
            role="experimental solver-free inference; quality gate not passed",
            import_path="waveforge.lab.mcl.solver_free.load_solver_free",
            online_physical_solves=0,
            results="../results/solver_free_20261030",
        )
    )
    index["experimental_solver_free_example"] = (
        "python examples/infer_solver_free.py "
        "--checkpoint models/solver_free_20261030/model_frozen.pt "
        "--cuda-graphs --output artifacts/solver_free_example"
    )
    write(ROOT / "models/index.json", index)
    for filename, sha in old:
        if digest(ROOT / "models" / filename) != sha:
            raise RuntimeError("old weights changed during packaging")
    write(
        args.output / "ФАЙЛЫ.json",
        {
            str(p.relative_to(args.output)): dict(
                bytes=p.stat().st_size, sha256=digest(p)
            )
            for p in args.output.rglob("*")
            if p.is_file()
        },
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "model_sha256",
                    "online_physical_solves",
                    "strict_no_quality_loss",
                    "status",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--fresh", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
