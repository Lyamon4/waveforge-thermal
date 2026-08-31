"""Paper-grade reporting for the frozen NCA-MT2B conditioning ablation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.ml.mt2b_conditioning import build_mt2b_conditioning
from waveforge.ml.mt2b_nca import MT2BNCA
from waveforge.ml.multitask_tasks import (
    VALIDATION_SEED,
    SourceLayoutTask,
    sample_primary_task,
)
from waveforge.ml.nca_training import model_state_sha256
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.fixed_operator import UniformPlateFactorization
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady
from waveforge.reproducibility import artifact_sha256
from waveforge.verification.high_fidelity import replicate_design

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

GeometryStratum = Literal["compact", "wide_horizontal", "vertically_spread", "mixed"]

INK = "#132238"
MUTED = "#607086"
RAW_COLOR = "#7C8DA6"
PHYSICS_COLOR = "#0077B6"
REFERENCE_COLOR = "#E58B3A"
GOOD_COLOR = "#14866D"
BAD_COLOR = "#C84A4A"
SINK_COLOR = "#00A6D6"
SOURCE_COLOR = "#F05A47"
SNAPSHOT_STEPS = (0, 1, 2, 4, 8, 16, 32, 48, 64)
STRATUM_ORDER: tuple[GeometryStratum, ...] = (
    "compact",
    "wide_horizontal",
    "vertically_spread",
    "mixed",
)


@dataclass(frozen=True)
class MT2BReportPaths:
    training_root: Path
    reference_root: Path
    evaluation_root: Path
    output_root: Path


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> dict[str, object]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.hashsalt": "waveforge-mt2b-paper-package",
    }


def _save_figure(
    figure: Figure,
    output_root: Path,
    stem: str,
    *,
    title: str,
) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("png", "svg", "pdf"):
        path = output_root / f"{stem}.{suffix}"
        figure.savefig(
            path,
            dpi=360,
            bbox_inches="tight",
            metadata={"Creator": "WaveForge Thermal", "Title": title},
        )
        paths.append(path)
    plt.close(figure)
    return paths


def _tasks() -> tuple[SourceLayoutTask, ...]:
    return tuple(sample_primary_task(VALIDATION_SEED, index) for index in range(32))


def _draw_task_overlay(axis: Axes, task: SourceLayoutTask) -> None:
    for scenario, bounds in zip(("A", "B", "C"), task.bounds, strict=True):
        x0, x1, y0, y1 = bounds
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="none",
                edgecolor=SOURCE_COLOR,
                linewidth=1.0,
            )
        )
        axis.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            scenario,
            ha="center",
            va="center",
            fontsize=6,
            color=SOURCE_COLOR,
            fontweight="bold",
        )
    axis.plot([0, 1], [0, 0], color=SINK_COLOR, linewidth=3)


def _image_axis(axis: Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _selected_updates(verdict: dict[str, object], variant: str) -> int:
    selected = dict(verdict["selected"])
    return int(dict(selected[variant])["completed_updates"])


def _selected_metrics(paths: MT2BReportPaths, variant: str) -> pd.DataFrame:
    verdict = _read_json(paths.evaluation_root / "mt2b_verdict.json")
    updates = _selected_updates(verdict, variant)
    return pd.read_csv(
        paths.evaluation_root
        / variant.lower()
        / f"checkpoint_{updates:06d}"
        / "solver_consistent_metrics.csv"
    )


def _selected_designs(
    paths: MT2BReportPaths, variant: str
) -> tuple[np.ndarray, np.ndarray]:
    verdict = _read_json(paths.evaluation_root / "mt2b_verdict.json")
    updates = _selected_updates(verdict, variant)
    directory = paths.evaluation_root / variant.lower() / f"checkpoint_{updates:06d}"
    return (
        np.load(directory / "continuous_designs_64.npy", allow_pickle=False),
        np.load(directory / "binary_designs_64.npy", allow_pickle=False),
    )


def _reference_designs(paths: MT2BReportPaths) -> np.ndarray:
    return np.stack(
        [
            np.load(
                paths.reference_root
                / "references"
                / f"task_{index:02d}"
                / "binary_design_64.npy",
                allow_pickle=False,
            )
            for index in range(32)
        ]
    )


def geometry_stratum(task: SourceLayoutTask) -> GeometryStratum:
    """Apply the locked geometry-only stratum definition."""
    x_coordinates = [center[0] for center in task.centers]
    y_coordinates = [center[1] for center in task.centers]
    horizontal_span = max(x_coordinates) - min(x_coordinates)
    vertical_span = max(y_coordinates) - min(y_coordinates)
    wide = horizontal_span >= 0.46
    vertical = vertical_span >= 0.21
    if wide and vertical:
        return "mixed"
    if wide:
        return "wide_horizontal"
    if vertical:
        return "vertically_spread"
    return "compact"


def illustrative_task_indices(gaps: NDArray[np.float64]) -> dict[str, int]:
    """Select disclosed best/median/worst ranks with stable index tie-breaking."""
    values = np.asarray(gaps, dtype=np.float64)
    if values.shape != (32,) or not np.isfinite(values).all():
        raise ValueError("illustrative selection requires 32 finite validation gaps")
    order = np.lexsort((np.arange(32), values))
    median_value = float(np.median(values))
    median_index = min(
        range(32),
        key=lambda index: (abs(values[index] - median_value), index),
    )
    return {
        "best": int(order[0]),
        "median": median_index,
        "worst": int(order[-1]),
    }


def _training_figure(paths: MT2BReportPaths) -> Figure:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    for variant, color in (("RAW", RAW_COLOR), ("PHYSICS", PHYSICS_COLOR)):
        frame = pd.read_csv(
            paths.training_root / variant.lower() / "training_metrics.csv"
        )
        x = frame["update"]
        objective = (
            frame["total_objective"].rolling(101, center=True, min_periods=1).median()
        )
        tmax = frame["exact_tmax"].rolling(101, center=True, min_periods=1).median()
        axes[0].plot(x, objective, color=color, linewidth=1.8, label=variant)
        axes[1].plot(x, tmax, color=color, linewidth=1.8, label=variant)
    for axis in axes:
        for boundary in (400, 800):
            axis.axvline(boundary, color="#AAB3BF", linestyle="--", linewidth=0.9)
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, ncol=2)
    axes[0].set_ylabel("Total objective\n(101-update rolling median)")
    axes[1].set_ylabel("Exact Tmax\n(101-update rolling median)")
    axes[1].set_xlabel("Optimizer update (zero-based)")
    figure.suptitle(
        "Matched RAW vs physics-transformed training", fontsize=16, weight="bold"
    )
    figure.text(
        0.5,
        0.93,
        "Same initialization, task stream, optimizer and NCA; "
        "only conditioning representation differs",
        ha="center",
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


def _checkpoint_figure(paths: MT2BReportPaths) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    for variant, color in (("RAW", RAW_COLOR), ("PHYSICS", PHYSICS_COLOR)):
        frame = pd.read_csv(
            paths.evaluation_root / variant.lower() / "checkpoint_summaries.csv"
        )
        x = frame["completed_updates"]
        axes[0].plot(
            x,
            100 * frame["median_relative_gap"],
            marker="o",
            color=color,
            label=variant,
        )
        axes[1].plot(
            x,
            100 * frame["p90_relative_gap"],
            marker="o",
            color=color,
            label=variant,
        )
    axes[0].set_title("Median gap to 600-step gradient")
    axes[1].set_title("90th-percentile gap")
    for axis in axes:
        axis.axhline(0, color=INK, linewidth=0.8)
        axis.set_xlabel("Checkpoint update")
        axis.set_ylabel("Relative gap (%)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(
        "Frozen-checkpoint quality on all 32 validation layouts",
        fontsize=15,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


def _paired_gap_figure(paths: MT2BReportPaths) -> Figure:
    raw = _selected_metrics(paths, "RAW")
    physics = _selected_metrics(paths, "PHYSICS")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    order = np.argsort(physics["relative_gap"].to_numpy())
    positions = np.arange(32)
    axes[0].scatter(
        positions,
        100 * raw.loc[order, "relative_gap"],
        s=32,
        color=RAW_COLOR,
        label="RAW",
        zorder=3,
    )
    axes[0].scatter(
        positions,
        100 * physics.loc[order, "relative_gap"],
        s=32,
        color=PHYSICS_COLOR,
        label="PHYSICS",
        zorder=3,
    )
    for x_position, raw_value, physics_value in zip(
        positions,
        100 * raw.loc[order, "relative_gap"],
        100 * physics.loc[order, "relative_gap"],
        strict=True,
    ):
        axes[0].plot(
            [x_position, x_position],
            [raw_value, physics_value],
            color="#CDD4DD",
            linewidth=0.7,
            zorder=1,
        )
    axes[0].axhline(0, color=INK, linewidth=0.9)
    axes[0].set_xlabel("Validation layouts, ordered by PHYSICS gap")
    axes[0].set_ylabel("Gap to gradient reference (%)")
    axes[0].set_title("Every layout is shown")
    axes[0].legend(frameon=False)
    deltas = 100 * (raw["relative_gap"] - physics["relative_gap"])
    axes[1].hist(deltas, bins=12, color=PHYSICS_COLOR, alpha=0.85, edgecolor="white")
    axes[1].axvline(0, color=INK, linewidth=1.0)
    axes[1].axvline(float(np.median(deltas)), color=GOOD_COLOR, linestyle="--")
    axes[1].set_xlabel("RAW gap − PHYSICS gap (percentage points)")
    axes[1].set_ylabel("Layout count")
    axes[1].set_title(f"Paired improvement; median={np.median(deltas):.2f} pp")
    for axis in axes:
        axis.grid(alpha=0.15)
    figure.suptitle(
        "Paired conditioning effect on unseen validation layouts",
        fontsize=15,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


def _strata_figure(paths: MT2BReportPaths) -> Figure:
    raw = _selected_metrics(paths, "RAW")
    physics = _selected_metrics(paths, "PHYSICS")
    tasks = _tasks()
    strata = np.asarray([geometry_stratum(task) for task in tasks])
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    width = 0.34
    positions = np.arange(len(STRATUM_ORDER))
    for offset, (variant, frame, color) in enumerate(
        (("RAW", raw, RAW_COLOR), ("PHYSICS", physics, PHYSICS_COLOR))
    ):
        medians = [
            100 * float(np.median(frame.loc[strata == stratum, "relative_gap"]))
            for stratum in STRATUM_ORDER
        ]
        axis.bar(
            positions + (offset - 0.5) * width,
            medians,
            width=width,
            color=color,
            label=variant,
        )
    axis.axhline(0, color=INK, linewidth=0.9)
    axis.set_xticks(positions, [item.replace("_", "\n") for item in STRATUM_ORDER])
    axis.set_ylabel("Median gap to gradient reference (%)")
    axis.set_title(
        "Performance by prospectively defined geometry stratum", weight="bold"
    )
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.18)
    figure.tight_layout()
    return figure


def _topology_panel(
    axis: Axes,
    design: np.ndarray,
    task: SourceLayoutTask,
    *,
    title: str,
) -> None:
    axis.imshow(
        design,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    _draw_task_overlay(axis, task)
    _image_axis(axis)
    axis.set_title(title, fontsize=8)


def _atlas_figure(paths: MT2BReportPaths, page: int) -> Figure:
    tasks = _tasks()
    reference = _reference_designs(paths)
    _, raw = _selected_designs(paths, "RAW")
    _, physics = _selected_designs(paths, "PHYSICS")
    raw_metrics = _selected_metrics(paths, "RAW")
    physics_metrics = _selected_metrics(paths, "PHYSICS")
    start = page * 8
    figure, axes = plt.subplots(8, 4, figsize=(9.4, 17.2), constrained_layout=True)
    for row, index in enumerate(range(start, start + 8)):
        task = tasks[index]
        source_map = task.sources.sum(axis=0)
        axes[row, 0].imshow(
            source_map,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="Reds",
            interpolation="nearest",
        )
        _draw_task_overlay(axes[row, 0], task)
        _image_axis(axes[row, 0])
        axes[row, 0].set_title(
            f"Layout {index:02d} • {geometry_stratum(task).replace('_', ' ')}",
            fontsize=8,
        )
        _topology_panel(
            axes[row, 1],
            reference[index],
            task,
            title=(
                "Gradient • Tmax="
                f"{raw_metrics.loc[index, 'reference_tmax_scipy64']:.4f}"
            ),
        )
        _topology_panel(
            axes[row, 2],
            raw[index],
            task,
            title=f"RAW • gap={100 * raw_metrics.loc[index, 'relative_gap']:.1f}%",
        )
        _topology_panel(
            axes[row, 3],
            physics[index],
            task,
            title=(
                f"PHYSICS • gap={100 * physics_metrics.loc[index, 'relative_gap']:.1f}%"
            ),
        )
    figure.suptitle(
        f"Complete validation topology atlas • layouts {start:02d}–{start + 7:02d}",
        fontsize=15,
        weight="bold",
    )
    return figure


def _representative_figure(paths: MT2BReportPaths) -> Figure:
    tasks = _tasks()
    reference = _reference_designs(paths)
    raw_continuous, raw_binary = _selected_designs(paths, "RAW")
    physics_continuous, physics_binary = _selected_designs(paths, "PHYSICS")
    physics_metrics = _selected_metrics(paths, "PHYSICS")
    indices = illustrative_task_indices(
        physics_metrics["relative_gap"].to_numpy(dtype=np.float64)
    )
    figure, axes = plt.subplots(3, 5, figsize=(13.0, 8.0), constrained_layout=True)
    for row, (label, index) in enumerate(indices.items()):
        task = tasks[index]
        panels = (
            (reference[index], "600-step gradient", "Greys", 0.0, 1.0),
            (raw_continuous[index], "RAW continuous", "viridis", 0.0, 1.0),
            (raw_binary[index], "RAW binary", "Greys", 0.0, 1.0),
            (
                physics_continuous[index],
                "PHYSICS continuous",
                "viridis",
                0.0,
                1.0,
            ),
            (physics_binary[index], "PHYSICS binary", "Greys", 0.0, 1.0),
        )
        for column, (field, title, cmap, lower, upper) in enumerate(panels):
            axes[row, column].imshow(
                field,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                interpolation="nearest",
            )
            _draw_task_overlay(axes[row, column], task)
            _image_axis(axes[row, column])
            axes[row, column].set_title(title, fontsize=9)
        axes[row, 0].set_ylabel(
            f"{label.upper()} rank\nlayout {index:02d}\nPHYSICS gap "
            f"{100 * physics_metrics.loc[index, 'relative_gap']:.2f}%",
            fontsize=8,
        )
    figure.suptitle(
        "Disclosed illustrative ranks: best, median-nearest and worst PHYSICS gap",
        fontsize=15,
        weight="bold",
    )
    return figure


def _load_selected_physics_model(paths: MT2BReportPaths) -> MT2BNCA:
    checkpoint = paths.evaluation_root / "frozen_models" / "physics_selected.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MT2BNCA().to(dtype=torch.float32)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if model_state_sha256(model) != payload["model_state_sha256"]:
        raise RuntimeError("frozen PHYSICS model hash mismatch during reporting")
    return model


def _conditioning(task: SourceLayoutTask) -> torch.Tensor:
    sources = torch.as_tensor(task.sources[None], dtype=torch.float64)
    factorization = UniformPlateFactorization(grid_size=64, conductivity=1.0)

    def solve(array: NDArray[np.float64]) -> NDArray[np.float64]:
        batch, scenarios, ny, nx = array.shape
        result = factorization.solve_many(array.reshape(batch * scenarios, ny, nx))
        if result.maximum_normalized_residual > 1.0e-10:
            raise RuntimeError("report conditioning field residual is invalid")
        return result.temperature.reshape(array.shape)

    return build_mt2b_conditioning(
        sources,
        variant="PHYSICS",
        temperature_solver=solve,
    )


def _conditioning_figure() -> Figure:
    task = _tasks()[0]
    condition = _conditioning(task)[0].numpy()
    labels = (
        "Source sum / 25",
        "Low-k temperature mean",
        "Low-k temperature max",
        "Sink mask",
    )
    cmaps = ("Reds", "inferno", "inferno", "Blues")
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), constrained_layout=True)
    for axis, field, label, cmap in zip(axes, condition, labels, cmaps, strict=True):
        image = axis.imshow(
            field,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=cmap,
            interpolation="nearest",
        )
        _draw_task_overlay(axis, task)
        _image_axis(axis)
        axis.set_title(label)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(
        "Physics-transformed conditioning contains no optimized topology "
        "or teacher design",
        fontsize=14,
        weight="bold",
    )
    return figure


def _rollout_figure(paths: MT2BReportPaths) -> Figure:
    model = _load_selected_physics_model(paths)
    task = _tasks()[0]
    condition = _conditioning(task)
    with torch.no_grad():
        rollout = model.rollout(
            condition,
            steps=64,
            snapshot_steps=SNAPSHOT_STEPS,
        )
    fields = [rollout.snapshots[step][0, 0].numpy() for step in SNAPSHOT_STEPS]
    limit = max(float(np.max(np.abs(field))) for field in fields)
    limit = max(limit, 1.0e-12)
    figure, axes = plt.subplots(1, 9, figsize=(17.0, 2.7), constrained_layout=True)
    for axis, step, field in zip(axes, SNAPSHOT_STEPS, fields, strict=True):
        axis.imshow(
            field,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        _draw_task_overlay(axis, task)
        _image_axis(axis)
        axis.set_title(f"step {step}")
    figure.suptitle(
        "Frozen PHYSICS-NCA local growth from exact zero state • validation layout 00",
        fontsize=14,
        weight="bold",
    )
    return figure


def _architecture_figure() -> Figure:
    figure, axis = plt.subplots(figsize=(13.0, 4.2))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = (
        (0.3, 1.2, 2.1, 1.5, "Thermal task", "3 source maps\n+ sink boundary"),
        (3.0, 1.2, 2.3, 1.5, "Physics transform", "uniform low-k solve\nTmean + Tmax"),
        (
            5.9,
            1.2,
            2.2,
            1.5,
            "Shared local NCA",
            "12,624 parameters\n64 × local 3×3 steps",
        ),
        (8.7, 1.2, 1.8, 1.5, "Exact projection", "25% material\n1024 / 4096 cells"),
        (11.1, 1.2, 1.6, 1.5, "Frozen design", "no Adam\nno backward"),
    )
    for x, y, width, height, heading, body in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="#F5F8FB",
                edgecolor=INK,
                linewidth=1.2,
            )
        )
        axis.text(x + width / 2, y + 1.08, heading, ha="center", weight="bold")
        axis.text(x + width / 2, y + 0.53, body, ha="center", va="center", color=MUTED)
    for left, right in pairwise(boxes):
        axis.add_patch(
            FancyArrowPatch(
                (left[0] + left[2] + 0.08, 1.95),
                (right[0] - 0.08, 1.95),
                arrowstyle="-|>",
                mutation_scale=14,
                color=PHYSICS_COLOR,
                linewidth=1.4,
            )
        )
    axis.text(
        6.5,
        3.55,
        "TRAINING: physics loss → backward through differentiable solver "
        "→ same shared NCA weights",
        ha="center",
        fontsize=11,
        weight="bold",
    )
    axis.text(
        6.5,
        0.35,
        "INFERENCE: one unseen layout → fixed physical representation "
        "→ 64 local updates → topology",
        ha="center",
        fontsize=11,
        weight="bold",
        color=GOOD_COLOR,
    )
    return figure


def _temperature_fields(
    design: np.ndarray,
    task: SourceLayoutTask,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = Grid2D(nx=256, ny=256)
    transferred = replicate_design(design, factor=4)
    conductivity = 1.0 + 19.0 * transferred
    fields: list[np.ndarray] = []
    for bounds in task.bounds:
        source = area_overlap_rectangular_source(grid, bounds, 1.0)
        result = solve_steady(
            grid,
            conductivity,
            source,
            BoundaryConditions.production(),
        )
        if result.normalized_residual > 1.0e-10:
            raise RuntimeError("report temperature solve residual exceeds tolerance")
        fields.append(result.temperature)
    return fields[0], fields[1], fields[2]


def _temperature_figure(paths: MT2BReportPaths) -> Figure:
    tasks = _tasks()
    physics_metrics = _selected_metrics(paths, "PHYSICS")
    index = illustrative_task_indices(
        physics_metrics["relative_gap"].to_numpy(dtype=np.float64)
    )["median"]
    task = tasks[index]
    reference = _reference_designs(paths)[index]
    _, raw = _selected_designs(paths, "RAW")
    _, physics = _selected_designs(paths, "PHYSICS")
    families = (
        ("600-step gradient", reference),
        ("Frozen RAW-NCA", raw[index]),
        ("Frozen PHYSICS-NCA", physics[index]),
    )
    field_families = [
        (label, _temperature_fields(design, task)) for label, design in families
    ]
    lower = min(float(field.min()) for _, fields in field_families for field in fields)
    upper = max(float(field.max()) for _, fields in field_families for field in fields)
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 9.6), constrained_layout=True)
    for row, (label, fields) in enumerate(field_families):
        for column, (scenario, field) in enumerate(zip("ABC", fields, strict=True)):
            image = axes[row, column].imshow(
                field,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="inferno",
                vmin=lower,
                vmax=upper,
            )
            _draw_task_overlay(axes[row, column], task)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_title(
                f"{label} • {scenario}\nTmax={field.max():.6f}", fontsize=9
            )
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Temperature")
    figure.suptitle(
        "Independent SciPy 256×256 temperature fields • "
        f"median-nearest layout {index:02d}",
        fontsize=15,
        weight="bold",
    )
    return figure


def _grid_transfer_figure(paths: MT2BReportPaths) -> Figure:
    frame = pd.read_csv(paths.evaluation_root / "selected_verified_256.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    color_map = {
        "REFERENCE": REFERENCE_COLOR,
        "RAW": RAW_COLOR,
        "PHYSICS": PHYSICS_COLOR,
    }
    for family, color in color_map.items():
        group = frame.loc[frame["family"] == family].sort_values("task_index")
        metrics = (
            _selected_metrics(paths, family)
            if family in {"RAW", "PHYSICS"}
            else _selected_metrics(paths, "RAW")
        )
        values64 = (
            metrics["candidate_tmax_scipy64"].to_numpy()
            if family in {"RAW", "PHYSICS"}
            else metrics["reference_tmax_scipy64"].to_numpy()
        )
        values256 = group["tmax_256"].to_numpy()
        axes[0].scatter(values64, values256, s=24, color=color, alpha=0.8, label=family)
        relative = 100 * (values256 - values64) / values64
        axes[1].boxplot(
            relative,
            positions=[list(color_map).index(family)],
            widths=0.55,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.7},
            medianprops={"color": INK},
        )
    combined = np.concatenate(
        [
            _selected_metrics(paths, "RAW")["candidate_tmax_scipy64"].to_numpy(),
            _selected_metrics(paths, "PHYSICS")["candidate_tmax_scipy64"].to_numpy(),
            frame["tmax_256"].to_numpy(),
        ]
    )
    bounds = (float(combined.min()), float(combined.max()))
    axes[0].plot(bounds, bounds, color=INK, linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("Independent SciPy Tmax at 64×64")
    axes[0].set_ylabel("Independent SciPy Tmax at 256×256")
    axes[0].legend(frameon=False)
    axes[0].set_title("Grid-transfer agreement")
    axes[1].set_xticks(range(3), list(color_map))
    axes[1].set_ylabel("Relative 64→256 change (%)")
    axes[1].set_title("Resolution sensitivity across all layouts")
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.tight_layout()
    return figure


def _summary_figure(paths: MT2BReportPaths) -> Figure:
    verdict = _read_json(paths.evaluation_root / "mt2b_verdict.json")
    raw = _selected_metrics(paths, "RAW")
    physics = _selected_metrics(paths, "PHYSICS")
    bootstrap = dict(verdict["bootstrap"])
    rows = [
        [
            "RAW control",
            f"{100 * np.median(raw['relative_gap']):.2f}%",
            f"{100 * np.quantile(raw['relative_gap'], 0.9):.2f}%",
            f"{int(np.sum(raw['relative_gap'] < 0))}/32",
        ],
        [
            "PHYSICS",
            f"{100 * np.median(physics['relative_gap']):.2f}%",
            f"{100 * np.quantile(physics['relative_gap'], 0.9):.2f}%",
            f"{int(np.sum(physics['relative_gap'] < 0))}/32",
        ],
    ]
    figure, axis = plt.subplots(figsize=(11.0, 5.0))
    axis.axis("off")
    axis.text(
        0.02,
        0.93,
        f"NCA-MT2B verdict: {verdict['status']}",
        fontsize=20,
        weight="bold",
        color=(
            GOOD_COLOR
            if "GO" in str(verdict["status"]) and "NO_GO" not in str(verdict["status"])
            else BAD_COLOR
        ),
    )
    axis.text(0.02, 0.84, str(verdict["exact_reason"]), fontsize=10, color=MUTED)
    table = axis.table(
        cellText=rows,
        colLabels=("Frozen generator", "Median gap", "P90 gap", "Wins vs gradient"),
        cellLoc="center",
        colLoc="center",
        bbox=(0.02, 0.38, 0.96, 0.34),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#EAF0F6" if row == 0 else "#F7F9FB")
        if row == 0:
            cell.set_text_props(weight="bold")
    delta = raw["relative_gap"].to_numpy() - physics["relative_gap"].to_numpy()
    axis.text(
        0.02,
        0.25,
        f"PHYSICS improves the paired gap on {np.sum(delta > 0)}/32 layouts; "
        f"median reduction {100 * np.median(delta):.2f} percentage points.",
        fontsize=11,
        weight="bold",
    )
    axis.text(
        0.02,
        0.16,
        "Paired bootstrap 95% CI for the median reduction: "
        f"[{100 * float(bootstrap['lower_bound']):.2f}, "
        f"{100 * float(bootstrap['upper_bound']):.2f}] percentage points.",
        fontsize=10,
    )
    axis.text(
        0.02,
        0.06,
        "Scope: development validation ablation only. Test ID/OOD remain sealed; "
        "no claim of final generalization yet.",
        fontsize=9,
        color=BAD_COLOR,
    )
    return figure


def _report_text(paths: MT2BReportPaths) -> str:
    verdict = _read_json(paths.evaluation_root / "mt2b_verdict.json")
    raw = _selected_metrics(paths, "RAW")
    physics = _selected_metrics(paths, "PHYSICS")
    selected = dict(verdict["selected"])
    bootstrap = dict(verdict["bootstrap"])
    diagnostics = dict(verdict["diagnostics"])
    raw_diag = dict(diagnostics["RAW"])
    physics_diag = dict(diagnostics["PHYSICS"])
    raw_cause = dict(raw_diag["condition_causality"])
    physics_cause = dict(physics_diag["condition_causality"])
    lines = [
        "# WaveForge Thermal — NCA-MT2B report",
        "",
        f"## Verdict: `{verdict['status']}`",
        "",
        str(verdict["exact_reason"]),
        "",
        "This is a matched development ablation between an architecturally "
        "identical RAW NCA and a physics-transformed NCA. Both models use the "
        "same initialization, procedural task stream, optimizer, 64-step local "
        "rollout and exact 25% material budget.",
        "",
        "| Variant | Selected update | Median gap | P90 gap | Worst gap | "
        "Wins vs gradient |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, frame in (("RAW", raw), ("PHYSICS", physics)):
        summary = dict(selected[variant])
        lines.append(
            f"| {variant} | {summary['completed_updates']} | "
            f"{100 * float(summary['median_relative_gap']):.3f}% | "
            f"{100 * float(summary['p90_relative_gap']):.3f}% | "
            f"{100 * float(summary['worst_relative_gap']):.3f}% | "
            f"{int(np.sum(frame['relative_gap'] < 0))}/32 |"
        )
    delta = raw["relative_gap"].to_numpy() - physics["relative_gap"].to_numpy()
    lines.extend(
        [
            "",
            "## Paired conditioning effect",
            "",
            f"- PHYSICS lower gap than RAW: `{int(np.sum(delta > 0))}/32` layouts",
            "- median paired gap reduction: "
            f"`{100 * np.median(delta):.6f}` percentage points",
            f"- percentile bootstrap 95% CI: "
            f"`[{100 * float(bootstrap['lower_bound']):.6f}, "
            f"{100 * float(bootstrap['upper_bound']):.6f}]` percentage points",
            "- bootstrap seed/resamples: "
            f"`{bootstrap['seed']}` / `{bootstrap['resamples']}`",
            "",
            "## Diagnostics",
            "",
            f"- RAW matched-conditioning wins: `{raw_cause['matched_win_count']}/32`",
            "- PHYSICS matched-conditioning wins: "
            f"`{physics_cause['matched_win_count']}/32`",
            "- Every generated binary design contains exactly "
            "`1024/4096 = 25%` high-k cells.",
            "- Candidate and gradient-reference designs were scored through "
            "the same independent SciPy64 path.",
            "- All selected RAW, PHYSICS and reference designs were "
            "secondarily verified at SciPy256.",
            "",
            "## Scientific interpretation",
            "",
            "The physical channels are deterministic transforms of the original "
            "task and uniform low-conductivity baseline physics. They contain no "
            "optimized design, gradient reference, adjoint sensitivity, validation "
            "statistics or teacher topology.",
            "",
            "## Claim limits",
            "",
            "The 32 validation layouts are a development set used for checkpoint "
            "selection. The sealed ID and OOD test sets remain unopened, so this "
            "experiment by itself is not the final generalization result and must "
            "not be presented as one.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_mt2b_paper_package(paths: MT2BReportPaths) -> dict[str, object]:
    """Generate all frozen scientific and presentation figures plus provenance."""
    paths.output_root.mkdir(parents=True, exist_ok=True)
    if not (paths.evaluation_root / "selected_verified_256.csv").is_file():
        raise FileNotFoundError(
            "secondary SciPy256 verification must precede reporting"
        )
    renderers = [
        (
            "01_training_curves",
            "Matched training curves",
            partial(_training_figure, paths),
        ),
        (
            "02_checkpoint_quality",
            "Checkpoint quality",
            partial(_checkpoint_figure, paths),
        ),
        (
            "03_paired_gap_distribution",
            "Paired validation gaps",
            partial(_paired_gap_figure, paths),
        ),
        (
            "04_geometry_strata",
            "Geometry-stratified quality",
            partial(_strata_figure, paths),
        ),
        (
            "05_representative_topologies",
            "Representative topologies",
            partial(_representative_figure, paths),
        ),
        (
            "06_conditioning_channels",
            "Physics-transformed conditioning",
            _conditioning_figure,
        ),
        ("07_nca_rollout", "Frozen NCA rollout", partial(_rollout_figure, paths)),
        (
            "08_temperature_maps",
            "Independent temperature maps",
            partial(_temperature_figure, paths),
        ),
        (
            "09_grid_transfer",
            "Grid-transfer diagnostic",
            partial(_grid_transfer_figure, paths),
        ),
        ("10_result_summary", "Result summary", partial(_summary_figure, paths)),
        ("11_architecture", "MT2B architecture", _architecture_figure),
    ]
    for page in range(4):
        renderers.append(
            (
                f"12_topology_atlas_page_{page + 1}",
                f"Complete topology atlas page {page + 1}",
                partial(_atlas_figure, paths, page),
            )
        )
    outputs: list[Path] = []
    with plt.rc_context(_style()):
        for stem, title, renderer in renderers:
            figure = renderer()
            outputs.extend(_save_figure(figure, paths.output_root, stem, title=title))
    report_path = paths.output_root / "MT2B_REPORT.md"
    report_path.write_text(
        _report_text(paths),
        encoding="utf-8",
        newline="\n",
    )
    outputs.append(report_path)
    guide_path = paths.output_root / "FIGURE_GUIDE.md"
    guide_path.write_text(
        "# Figure guide\n\n"
        "All figures are generated from frozen artifacts in PNG, SVG and PDF. "
        "Pages 1–4 of the topology atlas contain all 32 layouts and prevent "
        "best-case-only presentation. The best/median/worst figure is explicitly "
        "rank-selected by PHYSICS gap and is illustrative, not the primary evidence.\n",
        encoding="utf-8",
        newline="\n",
    )
    outputs.append(guide_path)
    manifest = {
        "schema_version": 1,
        "scientific_verdict": _read_json(paths.evaluation_root / "mt2b_verdict.json")[
            "status"
        ],
        "figure_count": len(renderers),
        "formats": ["png", "svg", "pdf"],
        "dpi": 360,
        "test_id_accessed": False,
        "test_ood_accessed": False,
        "artifacts": {
            path.relative_to(paths.output_root).as_posix(): artifact_sha256(path)
            for path in outputs
        },
    }
    manifest_path = paths.output_root / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
