"""Paper-grade RKNP figure pack assembled from frozen WaveForge evidence."""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from waveforge.design.branching_baseline import (
    BranchingTreeParameters,
    build_branching_tree,
)
from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady
from waveforge.reproducibility import artifact_sha256
from waveforge.verification.high_fidelity import replicate_design

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

PRODUCTION_SEEDS = (20260911, 20260912, 20260913)
PREVIOUS_SEEDS = (20260828, 20260829, 20260830)
SNAPSHOT_STEPS = (0, 1, 2, 4, 8, 16, 32, 48, 64)
SOURCE_BOUNDS = {
    "A": (0.40, 0.60, 0.62, 0.82),
    "B": (0.18, 0.38, 0.62, 0.82),
    "C": (0.62, 0.82, 0.62, 0.82),
}
INK = "#132238"
BLUE = "#2474b5"
CYAN = "#27a6b8"
ORANGE = "#e8792e"
RED = "#c23b3b"
GREEN = "#27865b"
GOLD = "#dda62b"
PALE = "#eef3f7"


@dataclass(frozen=True)
class FigureSpec:
    """One immutable paper figure entry."""

    figure_id: str
    stem: str
    title: str
    caption_ru: str
    claim_limit: str


@dataclass(frozen=True)
class PaperEvidence:
    """Minimal locked evidence used to prevent claim drift in reporting."""

    nca2_status: str
    tree_peak_256: float
    nca2_peaks_256: dict[int, float]
    primary_passing_seeds: tuple[int, ...]
    catastrophic_seed: int


def _spec(
    number: int,
    slug: str,
    title: str,
    caption_ru: str,
    claim_limit: str = "Только зафиксированные результаты WaveForge.",
) -> FigureSpec:
    figure_id = f"fig{number:02d}_{slug}"
    return FigureSpec(figure_id, figure_id, title, caption_ru, claim_limit)


FIGURE_SPECS = (
    _spec(
        1,
        "problem_setup",
        "Thermal inverse-design problem",
        "Постановка задачи: три равномощных hotspot, охлаждаемая нижняя граница и бюджет высокопроводящего материала 25%.",
    ),
    _spec(
        2,
        "waveforge_workflow",
        "WaveForge scientific workflow",
        "Полный проверяемый контур: постановка, differentiable optimization, строгая бинаризация и независимая SciPy-проверка.",
    ),
    _spec(
        3,
        "nca_architecture",
        "Pure neural cellular automaton",
        "Архитектура pure NCA: локальное общее правило, 64 синхронных шага и persistent physical conditioning.",
    ),
    _spec(
        4,
        "solver_validation",
        "Validated finite-volume physics",
        "Manufactured solution демонстрирует второй порядок сходимости reference solver.",
    ),
    _spec(
        5,
        "gate2_design_evolution",
        "Classical inverse-design evolution",
        "Переход от начального поля к continuous и strict-binary конструкции Gate 2A.",
    ),
    _spec(
        6,
        "strong_tree_baseline",
        "Strong parametric branching baseline",
        "Лучшее параметрическое дерево из заранее зафиксированного геометрического family.",
    ),
    _spec(
        7,
        "topology_comparison",
        "Three cooling-design strategies",
        "Сопоставление strong tree, pixel optimization и лучшей NCA-топологии при одинаковом бюджете.",
    ),
    _spec(
        8,
        "nca2_seed_gallery",
        "NCA-2 outcomes across all seeds",
        "Все три production seed без cherry-picking: два сильных результата и один collapse.",
    ),
    _spec(
        9,
        "success_failure_anatomy",
        "Anatomy of success and failure",
        "Геометрическое различие между collapsed seed и лучшим seed при одной архитектуре и protocol.",
    ),
    _spec(
        10,
        "nca_growth_rollout",
        "Cooling topology grown by local updates",
        "Рост material logit лучшего NCA seed от нулевого состояния до шага 64.",
    ),
    _spec(
        11,
        "temperature_scenarios",
        "Independent temperature verification",
        "Температурные поля A/B/C всех production seed, пересчитанные CPU SciPy на сетке 256×256.",
    ),
    _spec(
        12,
        "training_stability",
        "Training dynamics and continuation stages",
        "Динамика objective трёх seed и границы prospective continuation stages.",
    ),
    _spec(
        13,
        "protocol_qualification",
        "Prospective protocol qualification",
        "Сравнение constant-LR Protocol A и decayed-LR Protocol B на development seeds.",
    ),
    _spec(
        14,
        "performance_against_tree",
        "Solver-verified performance",
        "Сравнение Tmax strong tree, предыдущего WaveForge optimizer и всех NCA-2 seeds.",
    ),
    _spec(
        15,
        "grid_transfer",
        "Grid-transfer diagnostic",
        "Согласованность независимой проверки 128×128 и primary 256×256.",
    ),
    _spec(
        16,
        "budget_connectivity",
        "Budget and engineering connectivity",
        "Material budget и connectivity diagnostics публикуются отдельно от thermal verdict.",
    ),
    _spec(
        17,
        "research_timeline",
        "Evidence accumulated by successive gates",
        "Честная последовательность: validation, inverse design, strong challenge, capacity evidence и instability.",
    ),
    _spec(
        18,
        "graphical_abstract",
        "WaveForge Thermal — graphical abstract",
        "Графическое резюме эксперимента: NCA выращивает топологию, physics направляет обучение, SciPy подтверждает результат.",
        "Обязательно показать NCA2_NO_GO_EFFECT: 2/3 effect passes, stability gate failed; generalization не проверена.",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_paper_evidence(project_root: Path) -> PaperEvidence:
    """Load raw machine-readable evidence and fail if the campaign is relabelled."""
    nca2 = _read_json(project_root / "artifacts/nca2_stabilization/nca2_verdict.json")
    challenge = _read_json(
        project_root / "artifacts/gate2a_challenge/challenge_verdict.json"
    )
    seeds = {int(item["seed"]): item["verdict"] for item in nca2["seeds"]}
    if tuple(sorted(seeds)) != PRODUCTION_SEEDS:
        raise ValueError("NCA-2 production seed registry changed")
    passing = tuple(seed for seed in PRODUCTION_SEEDS if seeds[seed]["primary_pass"])
    collapsed = tuple(
        seed for seed in PRODUCTION_SEEDS if not seeds[seed]["noncollapse_pass"]
    )
    if len(collapsed) != 1:
        raise ValueError("Expected exactly one registered catastrophic NCA-2 seed")
    return PaperEvidence(
        nca2_status=str(nca2["campaign"]["status"]),
        tree_peak_256=float(challenge["winner"]["worst_peak_256"]),
        nca2_peaks_256={seed: float(seeds[seed]["peak_256"]) for seed in seeds},
        primary_passing_seeds=passing,
        catastrophic_seed=collapsed[0],
    )


def _paper_style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.hashsalt": "waveforge-rknp-paper-pack",
    }


def _title(figure: Figure, text: str, subtitle: str | None = None) -> None:
    figure.suptitle(text, fontsize=16, fontweight="bold", color=INK, y=0.995)
    if subtitle:
        figure.text(0.5, 0.90, subtitle, ha="center", fontsize=9, color="#536477")


def _clean_image_axis(axis: Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_sources(axis: Axes, *, labels: bool = True) -> None:
    for name, (x0, x1, y0, y1) in SOURCE_BOUNDS.items():
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="#ff6b4a",
                edgecolor="white",
                alpha=0.82,
                linewidth=1.5,
            )
        )
        if labels:
            axis.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                name,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
    axis.plot([0, 1], [0, 0], color="#20a7d8", linewidth=6, solid_capstyle="butt")


def _load_array(root: Path, relative: str) -> np.ndarray:
    return np.load(root / relative, allow_pickle=False).copy()


def _load_image(root: Path, relative: str) -> np.ndarray:
    return plt.imread(root / relative)


def _tree_design() -> np.ndarray:
    parameters = BranchingTreeParameters(
        x_sink=0.5,
        x_junction=0.5,
        y_junction=0.475,
        trunk_to_branch_width_ratio=1.25,
    )
    return build_branching_tree(parameters).design.copy()


def _verified_temperature(root: Path, seed: int, scenario: str) -> np.ndarray:
    design = _load_array(
        root,
        f"artifacts/nca2_stabilization/production_seed_{seed}/design_binary_64.npy",
    )
    grid = Grid2D(nx=256, ny=256)
    source = area_overlap_rectangular_source(grid, SOURCE_BOUNDS[scenario], 1.0)
    conductivity = 1.0 + 19.0 * replicate_design(design, factor=4) ** 3
    temperature = solve_steady(
        grid,
        conductivity,
        source,
        BoundaryConditions.production(),
    ).temperature
    metrics = pd.read_csv(
        root / "artifacts/nca2_stabilization/verified_256_metrics.csv"
    )
    expected = float(metrics.loc[metrics["seed"] == seed, f"peak_{scenario}"].iloc[0])
    if not np.isclose(float(temperature.max()), expected, rtol=1.0e-12, atol=1.0e-13):
        raise RuntimeError("paper figure solve differs from frozen verified metric")
    return temperature


def _imshow_design(axis: Axes, design: np.ndarray, title: str) -> None:
    axis.imshow(
        design,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="magma",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    _draw_sources(axis, labels=False)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")


def _box(
    axis: Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    color: str,
) -> None:
    x, y = xy
    width, height = size
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor=color,
            edgecolor="none",
            alpha=0.97,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=10,
    )


def _arrow(axis: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.6, color=INK
        )
    )


def _render_fig01(root: Path, _: PaperEvidence) -> Figure:
    figure, axis = plt.subplots(figsize=(9.4, 6.2))
    plate = np.tile(np.linspace(0.12, 0.45, 256)[:, None], (1, 256))
    axis.imshow(
        plate, origin="lower", extent=(0, 1, 0, 1), cmap="inferno", vmin=0, vmax=1
    )
    _draw_sources(axis)
    axis.text(
        0.5,
        -0.06,
        "Cooled boundary: T = 0",
        ha="center",
        va="top",
        color=BLUE,
        fontweight="bold",
    )
    axis.text(
        1.04,
        0.78,
        "3 equal-power\nsource scenarios",
        transform=axis.transAxes,
        va="center",
        fontsize=11,
    )
    axis.text(
        1.04,
        0.47,
        "Design D(x,y) ∈ [0,1]\nk(D)=1+19D³",
        transform=axis.transAxes,
        va="center",
        fontsize=11,
    )
    axis.text(
        1.04,
        0.19,
        "Locked budget\nmean(D) = 0.25",
        transform=axis.transAxes,
        va="center",
        fontsize=11,
        color=GREEN,
        fontweight="bold",
    )
    axis.set_xlim(0, 1.38)
    axis.set_ylim(-0.12, 1)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")
    _title(
        figure,
        "Thermal inverse-design problem",
        "Generic 2D conduction — not a calibrated industrial device",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def _render_fig02(root: Path, _: PaperEvidence) -> Figure:
    figure, axis = plt.subplots(figsize=(12.2, 4.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    labels = (
        "Heat-source\nconditions",
        "NCA / design\nvariables",
        "Differentiable\nCUDA physics",
        "Strict binary\nD ≥ 0.5",
        "Independent\nSciPy 256²",
        "Scientific\nverdict",
    )
    colors = (ORANGE, CYAN, BLUE, GOLD, GREEN, INK)
    xs = np.linspace(0.03, 0.84, len(labels))
    for index, (x, label, color) in enumerate(zip(xs, labels, colors, strict=True)):
        _box(axis, (float(x), 0.43), (0.13, 0.22), label, color)
        if index < len(labels) - 1:
            _arrow(axis, (float(x) + 0.13, 0.54), (float(xs[index + 1]) - 0.01, 0.54))
    axis.annotate(
        "thermal loss + backpropagation",
        xy=(0.36, 0.42),
        xytext=(0.36, 0.18),
        ha="center",
        color=BLUE,
        arrowprops={
            "arrowstyle": "->",
            "color": BLUE,
            "connectionstyle": "arc3,rad=-0.25",
        },
    )
    axis.text(
        0.5,
        0.83,
        "Optimization proposes; verified physics decides",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    _title(
        figure,
        "WaveForge scientific workflow",
        "Low-fidelity optimization is never used as final proof",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def _render_fig03(root: Path, _: PaperEvidence) -> Figure:
    figure, axis = plt.subplots(figsize=(11.0, 6.0))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _box(axis, (0.04, 0.58), (0.16, 0.22), "Zero state\n16 × 64 × 64", INK)
    _box(axis, (0.04, 0.20), (0.16, 0.22), "Persistent input\nsource + sink", ORANGE)
    _box(axis, (0.30, 0.39), (0.18, 0.25), "Concatenate\n18 channels", CYAN)
    _box(
        axis, (0.57, 0.39), (0.18, 0.25), "Conv 3×3: 18→64\nSiLU\nConv 1×1: 64→16", BLUE
    )
    _box(axis, (0.82, 0.39), (0.14, 0.25), "tanh × 0.1\nresidual update", GREEN)
    _arrow(axis, (0.20, 0.69), (0.29, 0.55))
    _arrow(axis, (0.20, 0.31), (0.29, 0.48))
    _arrow(axis, (0.48, 0.515), (0.56, 0.515))
    _arrow(axis, (0.75, 0.515), (0.81, 0.515))
    axis.add_patch(
        FancyArrowPatch(
            (0.89, 0.38),
            (0.13, 0.57),
            connectionstyle="arc3,rad=-0.33",
            arrowstyle="-|>",
            mutation_scale=13,
            color=GREEN,
            linewidth=2,
        )
    )
    axis.text(
        0.58,
        0.15,
        "Shared weights across every cell and all 64 synchronous updates",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        0.58,
        0.08,
        "11,472 trainable parameters • no labels • no tree prior • no coordinates",
        ha="center",
        fontsize=10,
        color="#536477",
    )
    _title(
        figure,
        "Pure neural cellular automaton",
        "Only physical conditioning can break the zero-state symmetry",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def _render_fig04(root: Path, _: PaperEvidence) -> Figure:
    frame = pd.read_csv(root / "artifacts/gate1_physics/validation_metrics.csv")
    rows = frame[frame["name"].str.startswith("manufactured_relative_l2_n")]
    resolutions = np.array([int(name.rsplit("n", 1)[1]) for name in rows["name"]])
    errors = rows["value"].to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].loglog(
        resolutions,
        errors,
        "o-",
        color=BLUE,
        linewidth=2.5,
        markersize=7,
        label="measured",
    )
    reference = errors[0] * (resolutions[0] / resolutions) ** 2
    axes[0].loglog(
        resolutions, reference, "--", color=ORANGE, label="second-order reference"
    )
    axes[0].xaxis.set_major_locator(FixedLocator(resolutions))
    axes[0].xaxis.set_major_formatter(
        FixedFormatter([str(value) for value in resolutions])
    )
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel("Grid resolution N")
    axes[0].set_ylabel("Relative L2 error")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend()
    axes[0].set_title("Manufactured-solution convergence")
    exact = _load_image(root, "artifacts/gate1_physics/manufactured_solution_error.png")
    axes[1].imshow(exact)
    _clean_image_axis(axes[1])
    axes[1].set_title("Spatial error at 128×128")
    figure.text(
        0.50,
        0.04,
        "Empirical order = 2.0003  •  L2 error: 8.04×10⁻⁴ → 5.02×10⁻⁵",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=GREEN,
    )
    _title(
        figure,
        "Validated finite-volume physics",
        "Flux-form discretization with harmonic face conductivity",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.91))
    return figure


def _render_fig05(root: Path, _: PaperEvidence) -> Figure:
    paths = (
        "artifacts/gate2_design/design_initial.png",
        "artifacts/gate2_design/design_optimized_continuous.png",
        "artifacts/gate2_design/design_optimized_binary.png",
    )
    titles = ("Initial design", "Optimized continuous D", "Strict binary D ≥ 0.5")
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))
    for axis, path, title in zip(axes, paths, titles, strict=True):
        axis.imshow(_load_image(root, path))
        _clean_image_axis(axis)
        axis.set_title(title)
    _title(
        figure,
        "Classical differentiable inverse design (Gate 2A)",
        "Three robust seeds passed independent 256×256 verification",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


def _render_fig06(root: Path, evidence: PaperEvidence) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    _imshow_design(axes[0], _tree_design(), "Best parametric tree")
    axes[1].imshow(
        _load_image(root, "artifacts/gate2a_challenge/best_tree_temperature_maps.png")
    )
    _clean_image_axis(axes[1])
    axes[1].set_title("Independent temperature fields")
    figure.text(
        0.5,
        0.03,
        f"Winner: x_sink=0.500, J=(0.500, 0.475), width ratio=1.25  •  Tmax₂₅₆={evidence.tree_peak_256:.6f}",
        ha="center",
        fontweight="bold",
    )
    _title(
        figure,
        "Strong human-engineered branching baseline",
        "Post-result challenge family; 41,055 deterministic candidates evaluated at 64×64",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.90))
    return figure


def _render_fig07(root: Path, evidence: PaperEvidence) -> Figure:
    pixel = _load_array(
        root,
        "artifacts/gate2_design/production/robust/20260828/design_binary_64.npy",
    )
    nca = _load_array(
        root,
        "artifacts/nca2_stabilization/production_seed_20260912/design_binary_64.npy",
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.7))
    _imshow_design(
        axes[0],
        _tree_design(),
        f"Parametric tree\nTmax={evidence.tree_peak_256:.6f}",
    )
    _imshow_design(axes[1], pixel, "Pixel WaveForge\nTmax=0.156507")
    _imshow_design(axes[2], nca, "Pure NCA (best seed)\nTmax=0.154835")
    figure.text(
        0.5,
        0.025,
        "Identical strict-binary material budget ≈25%  •  independent SciPy 256×256",
        ha="center",
        fontweight="bold",
        color=GREEN,
    )
    _title(
        figure,
        "Three strategies, one thermal objective",
        "The neural rule can represent a topology competitive with direct pixel optimization",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.89))
    return figure


def _render_fig08(root: Path, evidence: PaperEvidence) -> Figure:
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.8))
    for column, seed in enumerate(PRODUCTION_SEEDS):
        continuous = _load_array(
            root,
            f"artifacts/nca2_stabilization/production_seed_{seed}/design_continuous_64.npy",
        )
        binary = _load_array(
            root,
            f"artifacts/nca2_stabilization/production_seed_{seed}/design_binary_64.npy",
        )
        _imshow_design(axes[0, column], continuous, f"seed {seed} — continuous")
        passed = seed in evidence.primary_passing_seeds
        state = "effect PASS" if passed else "catastrophic collapse"
        _imshow_design(
            axes[1, column],
            binary,
            f"strict binary • Tmax={evidence.nca2_peaks_256[seed]:.6f}\n{state}",
        )
        axes[1, column].title.set_color(GREEN if passed else RED)
    _title(
        figure,
        "NCA-2 production outcomes — all registered seeds",
        "Two seeds beat the strong tree by ≥2%; one valid run collapses thermally",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.83))
    return figure


def _render_fig09(root: Path, evidence: PaperEvidence) -> Figure:
    bad = _load_array(
        root,
        "artifacts/nca2_stabilization/production_seed_20260911/design_binary_64.npy",
    )
    good = _load_array(
        root,
        "artifacts/nca2_stabilization/production_seed_20260912/design_binary_64.npy",
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.3))
    _imshow_design(
        axes[0],
        bad,
        f"Failed basin • seed 20260911\nTmax={evidence.nca2_peaks_256[20260911]:.6f}",
    )
    _imshow_design(
        axes[1],
        good,
        f"Strong basin • seed 20260912\nTmax={evidence.nca2_peaks_256[20260912]:.6f}",
    )
    axes[0].title.set_color(RED)
    axes[1].title.set_color(GREEN)
    axes[0].annotate(
        "All high-k cells are sink-connected,\nbut heat routes are inefficient",
        xy=(0.5, 0.45),
        xytext=(0.02, 0.05),
        textcoords="axes fraction",
        color=RED,
        arrowprops={"arrowstyle": "->", "color": RED},
    )
    axes[1].annotate(
        "Shared upper distribution bus\ncoordinates three conductive paths",
        xy=(0.5, 0.64),
        xytext=(0.52, 0.05),
        textcoords="axes fraction",
        color=GREEN,
        arrowprops={"arrowstyle": "->", "color": GREEN},
    )
    _title(
        figure,
        "Training stability, not model capacity, limits NCA-2",
        "Same 11,472-parameter rule and protocol; different optimization basins",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def _render_fig10(root: Path, _: PaperEvidence) -> Figure:
    archive_path = (
        root
        / "artifacts/nca2_stabilization/production_seed_20260912/rollout_snapshots.npz"
    )
    archive = np.load(archive_path, allow_pickle=False)
    fields = [
        np.asarray(archive[f"step_{step}"][0], dtype=float) for step in SNAPSHOT_STEPS
    ]
    limit = max(float(np.max(np.abs(field))) for field in fields)
    figure, axes = plt.subplots(
        1, len(SNAPSHOT_STEPS), figsize=(17.5, 3.0), constrained_layout=True
    )
    for axis, field, step in zip(axes, fields, SNAPSHOT_STEPS, strict=True):
        image = axis.imshow(
            field,
            origin="lower",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(f"step {step}")
        _clean_image_axis(axis)
    figure.colorbar(image, ax=axes.tolist(), shrink=0.65, label="material_logit")
    _title(
        figure,
        "A cooling topology grown by 64 local neural updates",
        "Best production seed 20260912 • zero initial state • shared local rule",
    )
    return figure


def _render_fig11(root: Path, _: PaperEvidence) -> Figure:
    source = _load_image(
        root,
        "artifacts/nca2_stabilization/final_package/temperature_scenario_details.png",
    )
    figure, axis = plt.subplots(figsize=(10.8, 9.4))
    axis.imshow(source)
    _clean_image_axis(axis)
    _title(
        figure,
        "Independent CPU SciPy verification at 256×256",
        "Rows: all production seeds • columns: source scenarios A/B/C • per-seed shared scale",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _render_fig12(root: Path, evidence: PaperEvidence) -> Figure:
    figure, axes = plt.subplots(3, 1, figsize=(10.8, 8.7), sharex=True)
    for axis, seed in zip(axes, PRODUCTION_SEEDS, strict=True):
        frame = pd.read_csv(
            root
            / f"artifacts/nca2_stabilization/production_seed_{seed}/optimization_metrics.csv"
        )
        axis.plot(
            frame["iteration"],
            frame["total_objective"],
            color=BLUE,
            linewidth=1.2,
            label="total J",
        )
        axis.plot(
            frame["iteration"],
            frame["thermal_smooth"],
            color=ORANGE,
            linewidth=1.0,
            label="thermal smooth",
        )
        axis.axvspan(0, 250, color=CYAN, alpha=0.10)
        axis.axvspan(250, 500, color=GOLD, alpha=0.10)
        axis.axvspan(500, 1500, color=GREEN, alpha=0.07)
        axis.axvline(250, color="0.35", linestyle="--", linewidth=0.8)
        axis.axvline(500, color="0.35", linestyle="--", linewidth=0.8)
        state = "PASS" if seed in evidence.primary_passing_seeds else "collapse"
        axis.text(
            0.985,
            0.13,
            f"verified Tmax={evidence.nca2_peaks_256[seed]:.6f} • {state}",
            transform=axis.transAxes,
            ha="right",
            color=GREEN if state == "PASS" else RED,
            fontweight="bold",
        )
        axis.set_ylabel(str(seed))
        axis.grid(alpha=0.18)
        axis.legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Training iteration (zero-based)")
    figure.text(0.22, 0.875, "soft formation", color=CYAN, fontweight="bold")
    figure.text(0.405, 0.875, "sharpening", color=GOLD, fontweight="bold")
    figure.text(0.67, 0.875, "exact final objective", color=GREEN, fontweight="bold")
    _title(
        figure,
        "NCA-2 training stability across untouched seeds",
        "Prospective continuation: β/α/binary weight = 2/100/0 → 4/250/.01 → 8/500/.02",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def _render_fig13(root: Path, _: PaperEvidence) -> Figure:
    verdict = _read_json(
        root / "artifacts/nca2_stabilization/qualification_verdict.json"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    protocols = {item["protocol_id"]: item for item in verdict["protocols"]}
    x = np.arange(3)
    for protocol_id, color, offset in (("A", BLUE, -0.08), ("B", ORANGE, 0.08)):
        item = protocols[protocol_id]
        finals = np.array([seed["checkpoint_peaks"][-1] for seed in item["seeds"]])
        axes[0].scatter(
            x + offset,
            finals,
            s=70,
            color=color,
            label=f"Protocol {protocol_id}",
            zorder=3,
        )
        axes[0].plot(x + offset, finals, color=color, alpha=0.35)
    axes[0].set_xticks(x, [str(seed) for seed in (20260901, 20260902, 20260903)])
    axes[0].set_ylabel("Final binary Tmax (64×64)")
    axes[0].set_title("Development-seed outcomes")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()
    axes[1].axis("off")
    axes[1].text(
        0.05,
        0.74,
        "Protocol A selected",
        fontsize=15,
        fontweight="bold",
        color=BLUE,
        transform=axes[1].transAxes,
    )
    axes[1].text(
        0.05,
        0.55,
        "3/3 stable development seeds",
        fontsize=11,
        transform=axes[1].transAxes,
    )
    axes[1].text(
        0.05,
        0.39,
        "median final Tmax\nA: 0.189081  <  B: 0.191489",
        fontsize=11,
        transform=axes[1].transAxes,
    )
    axes[1].text(
        0.05,
        0.18,
        "Selection reason:\nLOWER_MEDIAN_FINAL_TMAX",
        fontsize=11,
        fontweight="bold",
        transform=axes[1].transAxes,
    )
    _title(
        figure,
        "Prospective qualification before production",
        "Constant LR=10⁻³ (A) versus piecewise decay (B); production seeds were untouched",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def _render_fig14(root: Path, evidence: PaperEvidence) -> Figure:
    gate2 = _read_json(root / "artifacts/gate2_design/gate2_verdict.json")
    previous = {
        seed: float(gate2["nominal_seeds"][str(seed)]["metrics"]["candidate_peak"])
        for seed in PREVIOUS_SEEDS
    }
    labels = (
        ["Strong tree"]
        + [f"Pixel {seed}" for seed in PREVIOUS_SEEDS]
        + [f"NCA-2 {seed}" for seed in PRODUCTION_SEEDS]
    )
    values = (
        [evidence.tree_peak_256]
        + list(previous.values())
        + [evidence.nca2_peaks_256[seed] for seed in PRODUCTION_SEEDS]
    )
    colors = (
        [GOLD]
        + [BLUE] * 3
        + [
            RED if seed == evidence.catastrophic_seed else GREEN
            for seed in PRODUCTION_SEEDS
        ]
    )
    figure, axis = plt.subplots(figsize=(10.8, 5.8))
    y = np.arange(len(labels))
    axis.barh(y, values, color=colors, alpha=0.90)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.axvline(
        evidence.tree_peak_256, color=GOLD, linestyle="--", linewidth=1.5, label="tree"
    )
    axis.axvline(
        0.98 * evidence.tree_peak_256,
        color=GREEN,
        linestyle=":",
        linewidth=1.8,
        label="2% effect threshold",
    )
    axis.axvline(
        1.02 * evidence.tree_peak_256,
        color=RED,
        linestyle=":",
        linewidth=1.5,
        label="non-collapse threshold",
    )
    for row, value in enumerate(values):
        axis.text(
            value + 0.0007,
            row,
            f"{value:.6f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    axis.set_xlim(0.145, 0.196)
    axis.set_xlabel("Worst-case peak temperature — lower is better")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right")
    _title(
        figure,
        "Solver-verified performance against a strong comparator",
        "NCA-2: 2/3 effect passes, but the preregistered stability gate fails",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def _render_fig15(root: Path, _: PaperEvidence) -> Figure:
    frame128 = pd.read_csv(
        root / "artifacts/nca2_stabilization/verified_128_metrics.csv"
    )
    frame256 = pd.read_csv(
        root / "artifacts/nca2_stabilization/verified_256_metrics.csv"
    )
    merged = frame128[["seed", "worst_peak"]].merge(
        frame256[["seed", "worst_peak"]], on="seed", suffixes=("_128", "_256")
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    axes[0].plot([0.15, 0.195], [0.15, 0.195], "--", color="0.4", label="identity")
    for _, row in merged.iterrows():
        axes[0].scatter(row["worst_peak_128"], row["worst_peak_256"], s=85, color=BLUE)
        axes[0].annotate(
            str(int(row["seed"])),
            (row["worst_peak_128"], row["worst_peak_256"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axes[0].set_xlabel("Tmax at 128×128")
    axes[0].set_ylabel("Tmax at 256×256")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    relative = (
        100
        * np.abs(merged["worst_peak_128"] - merged["worst_peak_256"])
        / merged["worst_peak_256"]
    )
    axes[1].bar([str(int(seed)) for seed in merged["seed"]], relative, color=CYAN)
    axes[1].set_ylabel("Absolute grid-transfer change (%)")
    axes[1].set_xlabel("Production seed")
    axes[1].set_ylim(0, max(0.35, float(relative.max()) * 1.25))
    for index, value in enumerate(relative):
        axes[1].text(
            index, value + 0.01, f"{value:.3f}%", ha="center", fontweight="bold"
        )
    axes[1].grid(axis="y", alpha=0.2)
    _title(
        figure,
        "Resolution transfer is small and systematic",
        "Primary authority remains independent CPU SciPy at 256×256",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def _render_fig16(root: Path, evidence: PaperEvidence) -> Figure:
    verified = pd.read_csv(
        root / "artifacts/nca2_stabilization/verified_256_metrics.csv"
    )
    connectivity = pd.read_csv(
        root / "artifacts/nca2_stabilization/connectivity_metrics.csv"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    fractions = verified["binary_material_fraction"].to_numpy()
    axes[0].axhspan(0.24, 0.26, color=GREEN, alpha=0.12, label="locked valid range")
    axes[0].axhline(0.25, color=INK, linestyle="--", linewidth=1)
    axes[0].bar(
        [str(seed) for seed in verified["seed"]],
        fractions,
        color=[
            RED if int(seed) == evidence.catastrophic_seed else GREEN
            for seed in verified["seed"]
        ],
    )
    axes[0].set_ylim(0.238, 0.262)
    axes[0].set_ylabel("Strict-binary material fraction")
    axes[0].set_title("Material budget")
    axes[0].legend()
    x = np.arange(len(connectivity))
    axes[1].bar(
        x - 0.18,
        connectivity["component_count"],
        width=0.36,
        color=ORANGE,
        label="component count",
    )
    axes[1].bar(
        x + 0.18,
        connectivity["sink_connected_fraction"],
        width=0.36,
        color=BLUE,
        label="sink-connected fraction",
    )
    axes[1].set_xticks(x, [str(seed) for seed in connectivity["seed"]])
    axes[1].set_ylabel("Diagnostic value")
    axes[1].set_title("High-k connectivity")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    figure.text(
        0.5,
        0.025,
        "All A/B/C sources intersect a sink-connected high-k component in every seed",
        ha="center",
        fontweight="bold",
        color=GREEN,
    )
    _title(
        figure,
        "Constraint validity and engineering diagnostics",
        "Connectivity is reported, but thermal performance defines the scientific verdict",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.90))
    return figure


def _render_fig17(root: Path, _: PaperEvidence) -> Figure:
    figure, axis = plt.subplots(figsize=(12.0, 5.3))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    stages = (
        (0.04, "G1", "Physics PASS\n2nd-order convergence", BLUE),
        (0.23, "G2A", "Inverse design PASS\n3/3 robust seeds", GREEN),
        (0.42, "Tree", "STRONG PASS\n2/3 beat tree by 5%", GOLD),
        (0.61, "NCA-1", "Capacity shown\n1/3 reproducible", ORANGE),
        (0.80, "NCA-2", "2/3 effect passes\nstability NO-GO", RED),
    )
    for index, (x, name, result, color) in enumerate(stages):
        axis.scatter(
            [x + 0.07],
            [0.58],
            s=950,
            color=color,
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )
        axis.text(
            x + 0.07,
            0.58,
            name,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        axis.text(
            x + 0.07,
            0.30,
            result,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        if index < len(stages) - 1:
            _arrow(axis, (x + 0.145, 0.58), (stages[index + 1][0] + 0.015, 0.58))
    axis.text(
        0.5,
        0.11,
        "Negative results remain part of the evidence chain; no experiment is overwritten",
        ha="center",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )
    _title(
        figure,
        "WaveForge evidence accumulated gate by gate",
        "A validated scientific core, a strong inverse-design result, and an unresolved neural stability problem",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def _render_fig18(root: Path, evidence: PaperEvidence) -> Figure:
    best = _load_array(
        root,
        "artifacts/nca2_stabilization/production_seed_20260912/design_binary_64.npy",
    )
    rollout = np.load(
        root
        / "artifacts/nca2_stabilization/production_seed_20260912/rollout_snapshots.npz",
        allow_pickle=False,
    )
    middle = np.asarray(rollout["step_16"][0], dtype=float)
    temperature = _verified_temperature(root, 20260912, "C")
    figure = plt.figure(figsize=(14.0, 6.4))
    grid = figure.add_gridspec(
        2, 5, height_ratios=(4, 1), width_ratios=(1.1, 1, 1, 1.2, 1.25)
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(5)]
    setup = np.tile(np.linspace(0.1, 0.5, 64)[:, None], (1, 64))
    axes[0].imshow(
        setup, origin="lower", extent=(0, 1, 0, 1), cmap="inferno", vmin=0, vmax=1
    )
    _draw_sources(axes[0])
    axes[0].set_title("1  Physical task")
    axes[1].imshow(middle, origin="lower", cmap="coolwarm")
    axes[1].set_title("2  Local NCA growth")
    axes[2].imshow(best, origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("3  Strict binary design")
    axes[3].imshow(
        temperature,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="inferno",
        interpolation="bilinear",
    )
    x0, x1, y0, y1 = SOURCE_BOUNDS["C"]
    axes[3].add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
        )
    )
    axes[3].plot([0, 1], [0, 0], color="#20a7d8", linewidth=4)
    axes[3].set_title("4  SciPy verification")
    for axis in axes[:4]:
        _clean_image_axis(axis)
    axes[4].axis("off")
    axes[4].text(
        0.04,
        0.82,
        "Best NCA seed",
        fontsize=14,
        fontweight="bold",
        color=GREEN,
        transform=axes[4].transAxes,
    )
    axes[4].text(
        0.04,
        0.65,
        f"Tmax₂₅₆ = {evidence.nca2_peaks_256[20260912]:.6f}",
        fontsize=12,
        transform=axes[4].transAxes,
    )
    improvement = (
        100
        * (evidence.tree_peak_256 - evidence.nca2_peaks_256[20260912])
        / evidence.tree_peak_256
    )
    axes[4].text(
        0.04,
        0.51,
        f"{improvement:.2f}% better\nthan strong tree",
        fontsize=12,
        fontweight="bold",
        color=GREEN,
        transform=axes[4].transAxes,
    )
    axes[4].text(
        0.04,
        0.27,
        "Campaign verdict",
        fontsize=11,
        fontweight="bold",
        transform=axes[4].transAxes,
    )
    axes[4].text(
        0.04,
        0.10,
        "NCA2_NO_GO_EFFECT\n2/3 effect passes;\nstability gate failed",
        fontsize=11,
        fontweight="bold",
        color=RED,
        transform=axes[4].transAxes,
    )
    footer = figure.add_subplot(grid[1, :])
    footer.axis("off")
    footer.text(
        0.5,
        0.62,
        "Pure NCA can discover a high-performance cooling topology from physics alone",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    footer.text(
        0.5,
        0.20,
        "Capacity demonstrated • cross-seed reliability not yet demonstrated • no unseen-layout generalization claim",
        ha="center",
        fontsize=10,
        color="#536477",
    )
    _title(
        figure,
        "WaveForge Thermal",
        "Physics-guided local neural growth with independent solver authority",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


_RENDERERS: dict[str, Callable[[Path, PaperEvidence], Figure]] = {
    spec.figure_id: globals()[f"_render_fig{index:02d}"]
    for index, spec in enumerate(FIGURE_SPECS, start=1)
}


def _guide_text(specs: tuple[FigureSpec, ...], evidence: PaperEvidence) -> str:
    lines = [
        "# WaveForge Thermal — комплект фигур для РКНП",
        "",
        "Все численные значения взяты из frozen CSV/JSON/NPY artifacts. Figures используют English labels для вставки в paper; подписи ниже — на русском.",
        "",
        f"Machine-readable scientific verdict: `{evidence.nca2_status}`.",
        "",
        "Важно: два из трёх NCA-2 seeds прошли effect threshold, но один seed показал catastrophic collapse. Поэтому набор демонстрирует capacity NCA, а не стабильный общий AI success.",
        "",
    ]
    for number, spec in enumerate(specs, start=1):
        lines.extend(
            (
                f"## Figure {number}. {spec.title}",
                "",
                spec.caption_ru,
                "",
                f"Ограничение claim: {spec.claim_limit}",
                "",
                f"Файлы: `{spec.stem}.png`, `{spec.stem}.svg`, `{spec.stem}.pdf` (в зависимости от выбранных formats).",
                "",
            )
        )
    return "\n".join(lines)


def _source_hashes(root: Path) -> dict[str, str]:
    relatives = (
        "artifacts/gate1_physics/validation_metrics.csv",
        "artifacts/gate2_design/gate2_verdict.json",
        "artifacts/gate2a_challenge/challenge_verdict.json",
        "artifacts/nca2_stabilization/nca2_verdict.json",
        "artifacts/nca2_stabilization/verified_128_metrics.csv",
        "artifacts/nca2_stabilization/verified_256_metrics.csv",
        "artifacts/nca2_stabilization/connectivity_metrics.csv",
        "artifacts/nca2_stabilization/qualification_verdict.json",
    )
    return {relative: artifact_sha256(root / relative) for relative in relatives}


def build_paper_figure_pack(
    project_root: Path,
    output_dir: Path,
    *,
    formats: tuple[str, ...] = ("png", "svg", "pdf"),
    dpi: int = 300,
    selected_figure_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Generate the bounded paper pack and its provenance manifest."""
    root = project_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_formats = {"png", "svg", "pdf"}
    if not formats or not set(formats) <= allowed_formats:
        raise ValueError("formats must be a non-empty subset of png/svg/pdf")
    registry = {spec.figure_id: spec for spec in FIGURE_SPECS}
    ids = selected_figure_ids or tuple(registry)
    unknown = set(ids) - set(registry)
    if unknown:
        raise ValueError(f"unknown figure ids: {sorted(unknown)}")
    selected_specs = tuple(registry[figure_id] for figure_id in ids)
    evidence = load_paper_evidence(root)
    if evidence.nca2_status != "NCA2_NO_GO_EFFECT":
        raise ValueError("paper pack is locked to the recorded NCA-2 negative verdict")

    outputs: list[Path] = []
    with plt.rc_context(_paper_style()):
        for spec in selected_specs:
            figure = _RENDERERS[spec.figure_id](root, evidence)
            for file_format in formats:
                path = output_dir / f"{spec.stem}.{file_format}"
                metadata = {"Creator": "WaveForge Thermal", "Title": spec.title}
                figure.savefig(
                    path,
                    dpi=dpi,
                    bbox_inches="tight",
                    metadata=metadata,
                )
                outputs.append(path)
            plt.close(figure)

    guide_path = output_dir / "FIGURE_GUIDE.md"
    guide_path.write_text(
        _guide_text(selected_specs, evidence), encoding="utf-8", newline="\n"
    )
    outputs.append(guide_path)
    artifact_hashes = {
        path.relative_to(output_dir).as_posix(): artifact_sha256(path)
        for path in outputs
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "scientific_verdict": evidence.nca2_status,
        "claim_scope": "fixed_A_B_C_task_capacity_not_generalization",
        "figure_count": len(selected_specs),
        "formats": list(formats),
        "dpi": dpi,
        "figures": [
            {
                "figure_id": spec.figure_id,
                "stem": spec.stem,
                "title": spec.title,
                "caption_ru": spec.caption_ru,
                "claim_limit": spec.claim_limit,
            }
            for spec in selected_specs
        ],
        "source_hashes": _source_hashes(root),
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload
