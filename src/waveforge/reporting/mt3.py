"""Paper-grade development reporting for the frozen WaveForge MT3 campaign."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from numpy.typing import NDArray

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.experiments.run_mt2b_evaluation import validation_tasks
from waveforge.ml.mt3_evaluation import (
    MT3CheckpointSummary,
    MT3DevelopmentVerdict,
    classify_mt3_development,
    select_mt3_checkpoint,
)
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady
from waveforge.reproducibility import artifact_sha256
from waveforge.verification.high_fidelity import replicate_design

FIELD_COLOR = "#4C78A8"
SENS_COLOR = "#E45756"
REFERENCE_COLOR = "#6B7280"
GOOD_COLOR = "#2A9D8F"
INK = "#152238"
MUTED = "#5E6B7A"


@dataclass(frozen=True)
class MT3ReportPaths:
    """Frozen MT3 artifact roots used by the development package."""

    training_root: Path
    evaluation_root: Path
    reference_root: Path
    output_root: Path


TemperatureFieldProvider = Callable[
    [np.ndarray, SourceLayoutTask],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]


def illustrative_layout_indices(gaps: NDArray[np.float64]) -> dict[str, int]:
    """Select disclosed rank-based layouts without visual cherry-picking."""
    values = np.asarray(gaps, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("illustrative gaps must be a finite nonempty vector")
    order = np.argsort(values, kind="stable")
    return {
        "best": int(order[0]),
        "median": int(order[len(order) // 2]),
        "worst": int(order[-1]),
    }


def save_figure_triplet(
    figure: Figure,
    output_dir: Path,
    name: str,
) -> tuple[Path, Path, Path]:
    """Save one figure as 300-dpi PNG plus vector SVG and PDF."""
    if not name or Path(name).name != name:
        raise ValueError("figure name must be one safe path component")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_dir / f"{name}.{suffix}" for suffix in ("png", "svg", "pdf"))
    figure.savefig(paths[0], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(paths[1], bbox_inches="tight", facecolor="white")
    figure.savefig(paths[2], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return paths


def build_mt3_report_markdown(
    *,
    sens: MT3CheckpointSummary,
    field: MT3CheckpointSummary,
    verdict: MT3DevelopmentVerdict,
    figure_names: tuple[str, ...],
    verified_256: pd.DataFrame | None = None,
) -> str:
    """Build the disclosure-first MT3 development report."""
    figures = "\n".join(f"- `{name}`" for name in figure_names)
    sens_row = (
        f"| SENS_UNET_BEST4_R25 | {sens.completed_updates} | "
        f"{100 * sens.median_r25_relative_gap:.3f}% | "
        f"{100 * sens.p90_r25_relative_gap:.3f}% | "
        f"{100 * sens.worst_r25_relative_gap:.3f}% | "
        f"{sens.r25_win_count} |"
    )
    field_row = (
        f"| FIELD_UNET_BEST4_R25 | {field.completed_updates} | "
        f"{100 * field.median_r25_relative_gap:.3f}% | "
        f"{100 * field.p90_r25_relative_gap:.3f}% | "
        f"{100 * field.worst_r25_relative_gap:.3f}% | "
        f"{field.r25_win_count} |"
    )
    secondary = ""
    if verified_256 is not None:
        pivot = verified_256.pivot(
            index="task_index", columns="family", values="worst_peak"
        )
        if tuple(pivot.index) != tuple(range(32)):
            raise ValueError("256 verification must contain all 32 layouts")
        field_256 = (pivot["FIELD_UNET_BEST4_R25"] - pivot["REFERENCE"]) / pivot[
            "REFERENCE"
        ]
        sens_256 = (pivot["SENS_UNET_BEST4_R25"] - pivot["REFERENCE"]) / pivot[
            "REFERENCE"
        ]
        secondary = f"""
## Secondary independent SciPy 256x256 verification

- SENS median gap: {100 * float(np.median(sens_256)):.3f}%.
- FIELD median gap: {100 * float(np.median(field_256)):.3f}%.
- This grid-transfer diagnostic does not replace the preregistered SciPy64
  development checkpoint gate.
"""
    return f"""# WaveForge MT3 development report

## Scope

This package reports **development validation only**. ID/OOD test layouts remain sealed.
It does not claim final generalization performance.

## Primary method and accounting

The preregistered primary method is `SENS_UNET_BEST4_R25`: frozen SENS_UNET
generates four candidates, performs four forward-only physics scores, selects the
best candidate, and applies exactly 25 gradient-refinement updates to one candidate.
FIELD_UNET is shown as the matched conditioning control.

## Frozen checkpoint results

| Method | Updates | Median gap | P90 gap | Worst gap | Wins / 32 |
|---|---:|---:|---:|---:|---:|
{sens_row}
{field_row}
{secondary}

## Development verdict

`{verdict.status}` - {verdict.exact_reason}.

## Figures

{figures}
"""


def _load_summaries(path: Path) -> list[MT3CheckpointSummary]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries = [MT3CheckpointSummary(**row) for row in payload]
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("MT3 checkpoint summaries are unreadable") from error
    if not summaries:
        raise RuntimeError("MT3 checkpoint summaries are empty")
    return summaries


def _selected_directory(
    paths: MT3ReportPaths,
    variant: str,
    completed_updates: int,
) -> Path:
    directory = (
        paths.evaluation_root / variant.lower() / f"checkpoint_{completed_updates:06d}"
    )
    if not (directory / "validation_metrics.csv").is_file():
        raise RuntimeError(f"missing selected {variant} metrics")
    return directory


def _selected_metrics(
    paths: MT3ReportPaths,
    variant: str,
    completed_updates: int,
) -> pd.DataFrame:
    frame = pd.read_csv(
        _selected_directory(paths, variant, completed_updates)
        / "validation_metrics.csv"
    )
    if len(frame) != 32 or frame["task_index"].tolist() != list(range(32)):
        raise RuntimeError(f"selected {variant} metrics violate task ordering")
    return frame


def _task_arrays(
    paths: MT3ReportPaths,
    variant: str,
    completed_updates: int,
    task_index: int,
) -> dict[str, NDArray[np.float64]]:
    source = (
        _selected_directory(paths, variant, completed_updates)
        / "tasks"
        / f"task_{task_index:02d}.npz"
    )
    with np.load(source, allow_pickle=False) as payload:
        return {
            name: np.asarray(payload[name], dtype=np.float64)
            for name in (
                "candidate_binary_designs",
                "refined_continuous_design",
                "refined_binary_design",
            )
        }


def _reference_design(paths: MT3ReportPaths, task_index: int) -> np.ndarray:
    array = np.load(
        paths.reference_root
        / "references"
        / f"task_{task_index:02d}"
        / "binary_design_64.npy",
        allow_pickle=False,
    ).astype(np.float64, copy=False)
    if array.shape != (64, 64) or int(np.count_nonzero(array)) != 1024:
        raise RuntimeError("reference design violates exact binary budget")
    return array


def _draw_overlay(axis: plt.Axes, task: SourceLayoutTask) -> None:
    for left, right, bottom, top in task.bounds:
        axis.add_patch(
            Rectangle(
                (left, bottom),
                right - left,
                top - bottom,
                fill=False,
                edgecolor="#FFB000",
                linewidth=1.1,
            )
        )
    axis.axhline(1.0 / 64.0, color="#00A6D6", linewidth=1.4)


def _design_panel(
    axis: plt.Axes,
    design: np.ndarray,
    task: SourceLayoutTask,
    title: str,
    *,
    continuous: bool = False,
) -> None:
    axis.imshow(
        design,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="viridis" if continuous else "Greys",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    _draw_overlay(axis, task)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=8)


def _training_figure(paths: MT3ReportPaths) -> Figure:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    for variant, color in (("field_unet", FIELD_COLOR), ("sens_unet", SENS_COLOR)):
        frame = pd.read_csv(paths.training_root / variant / "training_metrics.csv")
        window = min(75, max(1, len(frame) // 20))
        label = variant.upper()
        axes[0].plot(
            frame["update"],
            frame["mean_loss"].rolling(window, min_periods=1).median(),
            color=color,
            label=label,
        )
        axes[1].plot(
            frame["update"],
            frame["best_candidate_thermal_smooth"]
            .rolling(window, min_periods=1)
            .median(),
            color=color,
            label=label,
        )
    axes[0].set_ylabel("Training objective")
    axes[1].set_ylabel("Best candidate thermal objective")
    axes[1].set_xlabel("Optimizer update")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Matched 4,000-update FIELD and SENS training trajectories")
    figure.tight_layout()
    return figure


def _checkpoint_figure(summaries: list[MT3CheckpointSummary]) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for variant, color in (("FIELD_UNET", FIELD_COLOR), ("SENS_UNET", SENS_COLOR)):
        rows = sorted(
            (summary for summary in summaries if summary.variant == variant),
            key=lambda summary: summary.completed_updates,
        )
        updates = [row.completed_updates for row in rows]
        axes[0].plot(
            updates,
            [100 * row.median_r25_relative_gap for row in rows],
            marker="o",
            color=color,
            label=variant,
        )
        axes[1].plot(
            updates,
            [100 * row.p90_r25_relative_gap for row in rows],
            marker="o",
            color=color,
            label=variant,
        )
    for axis, title in zip(axes, ("Median gap", "P90 gap"), strict=True):
        axis.axhline(0.0, color=INK, linewidth=0.8)
        axis.set_xlabel("Frozen checkpoint update")
        axis.set_ylabel("Gap to 600-step gradient (%)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Development checkpoint quality - independent SciPy64")
    figure.tight_layout()
    return figure


def _paired_gap_figure(field: pd.DataFrame, sens: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(32)
    axis.plot(
        x,
        100 * field["r25_relative_gap"],
        marker="o",
        color=FIELD_COLOR,
        linewidth=1.2,
        label="FIELD_UNET_BEST4_R25",
    )
    axis.plot(
        x,
        100 * sens["r25_relative_gap"],
        marker="o",
        color=SENS_COLOR,
        linewidth=1.2,
        label="SENS_UNET_BEST4_R25",
    )
    axis.axhline(0.0, color=INK, linewidth=1.0, label="600-step gradient")
    axis.set_xlabel("Development layout index")
    axis.set_ylabel("Relative Tmax gap (%)")
    axis.set_title("Per-layout solver-matched performance")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    return figure


def _gap_distribution_figure(field: pd.DataFrame, sens: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    values = [
        100 * field["r25_relative_gap"].to_numpy(dtype=np.float64),
        100 * sens["r25_relative_gap"].to_numpy(dtype=np.float64),
    ]
    violin = axis.violinplot(values, showmedians=True, showextrema=True)
    for body, color in zip(violin["bodies"], (FIELD_COLOR, SENS_COLOR), strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.7)
    axis.axhline(0.0, color=INK, linewidth=1.0)
    axis.set_xticks((1, 2), ("FIELD", "SENS"))
    axis.set_ylabel("Relative Tmax gap (%)")
    axis.set_title("Distribution across all 32 development layouts")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _representative_figure(
    paths: MT3ReportPaths,
    sens_update: int,
    sens_metrics: pd.DataFrame,
) -> Figure:
    tasks = validation_tasks()
    indices = illustrative_layout_indices(
        sens_metrics["r25_relative_gap"].to_numpy(dtype=np.float64)
    )
    figure, axes = plt.subplots(3, 4, figsize=(11.2, 8.5))
    for row, (rank, index) in enumerate(indices.items()):
        task = tasks[index]
        arrays = _task_arrays(paths, "SENS_UNET", sens_update, index)
        source_sum = task.sources.sum(axis=0)
        axes[row, 0].imshow(
            source_sum,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="Reds",
        )
        _draw_overlay(axes[row, 0], task)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_title(f"{rank.upper()} layout {index:02d}", fontsize=8)
        _design_panel(
            axes[row, 1],
            _reference_design(paths, index),
            task,
            "600-step gradient",
        )
        _design_panel(
            axes[row, 2],
            arrays["refined_continuous_design"],
            task,
            "SENS continuous R25",
            continuous=True,
        )
        _design_panel(
            axes[row, 3],
            arrays["refined_binary_design"],
            task,
            f"SENS binary gap={100 * sens_metrics.loc[index, 'r25_relative_gap']:.2f}%",
        )
    figure.suptitle("Predeclared best, median-rank and worst development examples")
    figure.tight_layout()
    return figure


def _candidate_atlas_figure(
    paths: MT3ReportPaths,
    sens_update: int,
    sens_metrics: pd.DataFrame,
) -> Figure:
    tasks = validation_tasks()
    indices = illustrative_layout_indices(
        sens_metrics["r25_relative_gap"].to_numpy(dtype=np.float64)
    )
    figure, axes = plt.subplots(3, 6, figsize=(15.0, 7.8))
    for row, (rank, index) in enumerate(indices.items()):
        task = tasks[index]
        arrays = _task_arrays(paths, "SENS_UNET", sens_update, index)
        for head in range(4):
            _design_panel(
                axes[row, head],
                arrays["candidate_binary_designs"][head],
                task,
                f"Head {head}",
            )
        _design_panel(
            axes[row, 4],
            arrays["refined_binary_design"],
            task,
            f"Selected + R25\n{rank} rank",
        )
        _design_panel(
            axes[row, 5],
            _reference_design(paths, index),
            task,
            "600-step gradient",
        )
    figure.suptitle("Four generated candidates, one selected refinement chain")
    figure.tight_layout()
    return figure


def _refinement_figure(
    paths: MT3ReportPaths,
    sens_update: int,
    sens_metrics: pd.DataFrame,
) -> Figure:
    indices = illustrative_layout_indices(
        sens_metrics["r25_relative_gap"].to_numpy(dtype=np.float64)
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for rank, index in indices.items():
        path = (
            _selected_directory(paths, "SENS_UNET", sens_update)
            / "tasks"
            / f"task_{index:02d}_trace.json"
        )
        records = json.loads(path.read_text(encoding="utf-8"))["records"]
        axes[0].plot(
            [row["iteration"] for row in records],
            [row["total_objective"] for row in records],
            label=f"{rank} layout {index:02d}",
        )
        axes[1].plot(
            [row["iteration"] for row in records],
            [row["exact_peak"] for row in records],
            label=f"{rank} layout {index:02d}",
        )
    axes[0].set_title("Refinement objective")
    axes[1].set_title("Exact peak during refinement")
    for axis in axes:
        axis.set_xlabel("Refinement update")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Exactly one selected candidate receives 25 updates")
    figure.tight_layout()
    return figure


def _architecture_figure() -> Figure:
    figure, axis = plt.subplots(figsize=(13.5, 4.3))
    axis.set_xlim(0, 13.5)
    axis.set_ylim(0, 4.3)
    axis.axis("off")
    boxes = (
        (0.2, 1.25, 2.0, "Thermal task", "3 source layouts\n+ sink boundary"),
        (2.8, 1.25, 2.0, "Physics probe", "Tmean, Tmax\n+ initial sensitivity"),
        (5.4, 1.25, 2.0, "Shared U-Net", "2.92 M parameters\n4 candidate heads"),
        (8.0, 1.25, 2.0, "Physics score", "4 forward solves\nselect best head"),
        (
            10.6,
            1.25,
            2.5,
            "Minimal refinement",
            "one candidate only\n25 gradient updates",
        ),
    )
    for x, y, width, title, body in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                1.55,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="#F4F7FA",
                edgecolor=INK,
                linewidth=1.2,
            )
        )
        axis.text(x + width / 2, y + 1.10, title, ha="center", weight="bold")
        axis.text(x + width / 2, y + 0.53, body, ha="center", va="center")
    for left, right in pairwise(boxes):
        axis.add_patch(
            FancyArrowPatch(
                (left[0] + left[2] + 0.08, 2.03),
                (right[0] - 0.08, 2.03),
                arrowstyle="-|>",
                mutation_scale=14,
                color=SENS_COLOR,
            )
        )
    axis.text(
        6.75,
        3.72,
        "Teacher-free physics training across procedural layouts",
        ha="center",
        fontsize=14,
        weight="bold",
    )
    axis.text(
        6.75,
        0.45,
        "At inference: frozen weights; only the preregistered R25 hybrid step adapts",
        ha="center",
        color=MUTED,
    )
    return figure


def temperature_fields_256(
    design: np.ndarray,
    task: SourceLayoutTask,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return independent SciPy256 scenario fields for one frozen binary design."""
    binary = np.asarray(design, dtype=np.float64)
    if (
        binary.shape != (64, 64)
        or not np.isin(binary, (0.0, 1.0)).all()
        or int(np.count_nonzero(binary)) != 1024
    ):
        raise ValueError("temperature maps require an exact-budget binary design")
    grid = Grid2D(nx=256, ny=256)
    conductivity = 1.0 + 19.0 * replicate_design(binary, factor=4)
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
            raise RuntimeError("temperature-map residual exceeds tolerance")
        fields.append(result.temperature)
    return fields[0], fields[1], fields[2]


def _temperature_figure(
    paths: MT3ReportPaths,
    sens_update: int,
    sens_metrics: pd.DataFrame,
    verified: pd.DataFrame,
    provider: TemperatureFieldProvider,
) -> Figure:
    tasks = validation_tasks()
    index = illustrative_layout_indices(
        sens_metrics["r25_relative_gap"].to_numpy(dtype=np.float64)
    )["median"]
    task = tasks[index]
    families = (
        ("REFERENCE", _reference_design(paths, index), "600-step gradient"),
        (
            "FIELD_UNET_BEST4_R25",
            _task_arrays(paths, "FIELD_UNET", sens_update, index)[
                "refined_binary_design"
            ],
            "FIELD U-Net + R25",
        ),
        (
            "SENS_UNET_BEST4_R25",
            _task_arrays(paths, "SENS_UNET", sens_update, index)[
                "refined_binary_design"
            ],
            "SENS U-Net + R25",
        ),
    )
    computed: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for family, design, label in families:
        fields = tuple(
            np.asarray(field, dtype=np.float64) for field in provider(design, task)
        )
        if (
            len(fields) != 3
            or any(field.shape != (256, 256) for field in fields)
            or any(not np.isfinite(field).all() for field in fields)
        ):
            raise RuntimeError("temperature provider returned invalid fields")
        expected_rows = verified[
            (verified["task_index"] == index) & (verified["family"] == family)
        ]
        if len(expected_rows) != 1:
            raise RuntimeError("selected 256 verification row is missing")
        expected = float(expected_rows.iloc[0]["worst_peak"])
        if abs(max(float(field.max()) for field in fields) - expected) > 1.0e-10:
            raise RuntimeError("temperature map disagrees with 256 verification")
        computed.append((label, fields))
    lower = min(float(field.min()) for _, fields in computed for field in fields)
    upper = max(float(field.max()) for _, fields in computed for field in fields)
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 9.6))
    image = None
    for row, (label, fields) in enumerate(computed):
        for column, (scenario, field) in enumerate(zip("ABC", fields, strict=True)):
            image = axes[row, column].imshow(
                field,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="inferno",
                vmin=lower,
                vmax=upper,
            )
            _draw_overlay(axes[row, column], task)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_title(
                f"{label} - scenario {scenario}\nTmax={field.max():.6f}",
                fontsize=8,
            )
    figure.suptitle(
        f"Independent SciPy 256x256 temperature fields - layout {index:02d}"
    )
    figure.subplots_adjust(top=0.91, right=0.86, wspace=0.16, hspace=0.28)
    if image is not None:
        colorbar_axis = figure.add_axes((0.89, 0.12, 0.025, 0.74))
        figure.colorbar(image, cax=colorbar_axis, label="Temperature")
    return figure


def _grid_transfer_figure(
    field_metrics: pd.DataFrame,
    sens_metrics: pd.DataFrame,
    verified: pd.DataFrame,
) -> tuple[Figure, pd.DataFrame]:
    pivot = verified.pivot(index="task_index", columns="family", values="worst_peak")
    summary = pd.DataFrame(
        {
            "task_index": np.arange(32),
            "field_gap_64": field_metrics["r25_relative_gap"],
            "sens_gap_64": sens_metrics["r25_relative_gap"],
            "field_gap_256": (pivot["FIELD_UNET_BEST4_R25"] - pivot["REFERENCE"])
            / pivot["REFERENCE"],
            "sens_gap_256": (pivot["SENS_UNET_BEST4_R25"] - pivot["REFERENCE"])
            / pivot["REFERENCE"],
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x = np.arange(32)
    axes[0].plot(x, 100 * summary["field_gap_64"], color=FIELD_COLOR, label="64x64")
    axes[0].plot(
        x,
        100 * summary["field_gap_256"],
        color=FIELD_COLOR,
        linestyle="--",
        label="256x256",
    )
    axes[1].plot(x, 100 * summary["sens_gap_64"], color=SENS_COLOR, label="64x64")
    axes[1].plot(
        x,
        100 * summary["sens_gap_256"],
        color=SENS_COLOR,
        linestyle="--",
        label="256x256",
    )
    for axis, title in zip(
        axes, ("FIELD grid transfer", "SENS grid transfer"), strict=True
    ):
        axis.axhline(0.0, color=INK, linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Development layout index")
        axis.set_ylabel("Gap to gradient (%)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    return figure, summary


def _speed_and_best_result_figure(
    paths: MT3ReportPaths,
    selected_update: int,
    verified: pd.DataFrame,
) -> Figure:
    """Compare locked optimization effort and the best verified FIELD result."""
    pivot = verified.pivot(index="task_index", columns="family", values="worst_peak")
    required = {"REFERENCE", "FIELD_UNET_BEST4_R25"}
    if not required.issubset(pivot.columns) or tuple(pivot.index) != tuple(range(32)):
        raise RuntimeError("speed/result figure requires complete 256 verification")
    gaps = (pivot["FIELD_UNET_BEST4_R25"] - pivot["REFERENCE"]) / pivot["REFERENCE"]
    task_index = int(gaps.idxmin())
    reference_peak = float(pivot.loc[task_index, "REFERENCE"])
    field_peak = float(pivot.loc[task_index, "FIELD_UNET_BEST4_R25"])
    improvement = 100.0 * (reference_peak - field_peak) / reference_peak
    task = validation_tasks()[task_index]
    reference_design = _reference_design(paths, task_index)
    field_design = _task_arrays(
        paths,
        "FIELD_UNET",
        selected_update,
        task_index,
    )["refined_binary_design"]

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.2))
    methods = ("600-step gradient", "U-Net + R25")
    updates = (600, 25)
    axes[0, 0].barh(methods, updates, color=(REFERENCE_COLOR, FIELD_COLOR))
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim(0, 650)
    axes[0, 0].set_xlabel(
        "Task-specific gradient updates (+4 forward-only U-Net candidate scores)"
    )
    axes[0, 0].set_title("Optimization effort: 24x fewer gradient updates")
    axes[0, 0].grid(axis="x", alpha=0.25)
    for row, value in enumerate(updates):
        axes[0, 0].text(value + 10, row, f"{value}", va="center", color=INK)
    peaks = (reference_peak, field_peak)
    axes[0, 1].barh(methods, peaks, color=(REFERENCE_COLOR, FIELD_COLOR))
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim(0.0, max(peaks) * 1.16)
    axes[0, 1].set_xlabel("Worst-case Tmax (independent SciPy 256x256)")
    axes[0, 1].set_title(f"Best development result: {improvement:.2f}% lower Tmax")
    axes[0, 1].grid(axis="x", alpha=0.25)
    for row, value in enumerate(peaks):
        axes[0, 1].text(
            value + max(peaks) * 0.015,
            row,
            f"{value:.6f}",
            va="center",
            color=INK,
        )

    _design_panel(
        axes[1, 0],
        reference_design,
        task,
        f"600-step gradient topology - layout {task_index:02d}",
    )
    _design_panel(
        axes[1, 1],
        field_design,
        task,
        f"FIELD U-Net + R25 topology - layout {task_index:02d}",
    )
    figure.suptitle(
        "WaveForge MT3: less task-specific optimization and lower verified temperature",
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Development layout; exact 25% material; ID/OOD test layouts remain sealed.",
        ha="center",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    return figure


def _readme_ru(
    sens: MT3CheckpointSummary,
    field: MT3CheckpointSummary,
    verdict: MT3DevelopmentVerdict,
) -> str:
    return f"""# WaveForge MT3 - текущий пакет результатов

Это **development validation**, а не финальный ID/OOD test. Закрытые тестовые
задачи не открывались.

- Главный метод: SENS_UNET -> 4 кандидата -> 4 быстрые проверки физикой ->
  один лучший кандидат -> ровно 25 шагов доработки.
- Выбранный checkpoint SENS: {sens.completed_updates} updates.
- Median gap SENS к 600-step gradient: {100 * sens.median_r25_relative_gap:.3f}%.
- Median gap FIELD control: {100 * field.median_r25_relative_gap:.3f}%.
- Машинный verdict: `{verdict.status}`.

Папка `figures` содержит одинаковые изображения в PNG 300 dpi, SVG и PDF.
"""


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_mt3_development_package(
    paths: MT3ReportPaths,
    *,
    include_temperature_maps: bool = True,
    temperature_field_provider: TemperatureFieldProvider = temperature_fields_256,
) -> Path:
    """Build one disclosure-complete development package from frozen artifacts."""
    summaries = _load_summaries(paths.evaluation_root / "checkpoint_summaries.json")
    sens_rows = [summary for summary in summaries if summary.variant == "SENS_UNET"]
    sens = select_mt3_checkpoint(sens_rows)
    matched_field = [
        summary
        for summary in summaries
        if summary.variant == "FIELD_UNET"
        and summary.completed_updates == sens.completed_updates
    ]
    if len(matched_field) != 1:
        raise RuntimeError("matched FIELD summary is missing at selected SENS update")
    field = matched_field[0]
    verdict = classify_mt3_development(
        median_gap=sens.median_r25_relative_gap,
        p90_gap=sens.p90_r25_relative_gap,
        worst_gap=sens.worst_r25_relative_gap,
        wins=sens.r25_win_count,
        valid_count=sens.task_count - sens.invalid_count,
        exact_budget_count=sens.exact_budget_count,
    )
    field_metrics = _selected_metrics(paths, "FIELD_UNET", sens.completed_updates)
    sens_metrics = _selected_metrics(paths, "SENS_UNET", sens.completed_updates)
    figures_dir = paths.output_root / "figures"
    models_dir = paths.output_root / "models"
    data_dir = paths.output_root / "data"
    for directory in (figures_dir, models_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    builders: tuple[tuple[str, Callable[[], Figure]], ...] = (
        ("01_training_curves", lambda: _training_figure(paths)),
        ("02_checkpoint_quality", lambda: _checkpoint_figure(summaries)),
        (
            "03_per_layout_gap",
            lambda: _paired_gap_figure(field_metrics, sens_metrics),
        ),
        (
            "04_gap_distribution",
            lambda: _gap_distribution_figure(field_metrics, sens_metrics),
        ),
        (
            "05_representative_topologies",
            lambda: _representative_figure(paths, sens.completed_updates, sens_metrics),
        ),
        (
            "06_four_candidate_atlas",
            lambda: _candidate_atlas_figure(
                paths, sens.completed_updates, sens_metrics
            ),
        ),
        (
            "07_refinement_trajectories",
            lambda: _refinement_figure(paths, sens.completed_updates, sens_metrics),
        ),
        ("08_method_diagram", _architecture_figure),
    )
    figure_names: list[str] = []
    for name, builder in builders:
        save_figure_triplet(builder(), figures_dir, name)
        figure_names.append(name)

    verified_frame: pd.DataFrame | None = None
    if include_temperature_maps:
        verified = paths.evaluation_root / "selected_verified_256.csv"
        if not verified.is_file():
            raise RuntimeError("selected 256x256 verification is not available")
        verified_frame = pd.read_csv(verified)
        expected_families = {
            "REFERENCE",
            "FIELD_UNET_BEST4_R25",
            "SENS_UNET_BEST4_R25",
        }
        if (
            len(verified_frame) != 96
            or set(verified_frame["family"]) != expected_families
        ):
            raise RuntimeError("selected 256x256 verification is incomplete")
        save_figure_triplet(
            _temperature_figure(
                paths,
                sens.completed_updates,
                sens_metrics,
                verified_frame,
                temperature_field_provider,
            ),
            figures_dir,
            "09_temperature_maps_256",
        )
        figure_names.append("09_temperature_maps_256")
        grid_figure, grid_summary = _grid_transfer_figure(
            field_metrics,
            sens_metrics,
            verified_frame,
        )
        save_figure_triplet(grid_figure, figures_dir, "10_grid_transfer_64_to_256")
        figure_names.append("10_grid_transfer_64_to_256")
        save_figure_triplet(
            _speed_and_best_result_figure(
                paths,
                sens.completed_updates,
                verified_frame,
            ),
            figures_dir,
            "11_speed_and_best_result",
        )
        figure_names.append("11_speed_and_best_result")
        grid_summary.to_csv(data_dir / "grid_transfer_metrics.csv", index=False)

    report = build_mt3_report_markdown(
        sens=sens,
        field=field,
        verdict=verdict,
        figure_names=tuple(figure_names),
        verified_256=verified_frame,
    )
    paths.output_root.mkdir(parents=True, exist_ok=True)
    (paths.output_root / "MT3_REPORT.md").write_text(
        report,
        encoding="utf-8",
        newline="\n",
    )
    (paths.output_root / "README_RU.md").write_text(
        _readme_ru(sens, field, verdict),
        encoding="utf-8",
        newline="\n",
    )
    pd.DataFrame(
        [
            {"method": "SENS_UNET_BEST4_R25", **asdict(sens)},
            {"method": "FIELD_UNET_BEST4_R25", **asdict(field)},
        ]
    ).to_csv(
        paths.output_root / "performance_table.csv", index=False, lineterminator="\n"
    )
    field_metrics.to_csv(data_dir / "field_validation_metrics.csv", index=False)
    sens_metrics.to_csv(data_dir / "sens_validation_metrics.csv", index=False)
    _write_json(paths.output_root / "development_verdict.json", asdict(verdict))
    _write_json(paths.output_root / "selected_checkpoint.json", asdict(sens))
    for variant in ("field_unet", "sens_unet"):
        source = (
            paths.training_root
            / variant
            / f"checkpoint_{sens.completed_updates:06d}.pt"
        )
        shutil.copy2(source, models_dir / f"{variant}_selected.pt")

    files = sorted(
        path
        for path in paths.output_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "scope": "development_validation_only",
        "test_id_accessed": False,
        "test_ood_accessed": False,
        "files": {
            path.relative_to(paths.output_root).as_posix(): artifact_sha256(path)
            for path in files
        },
    }
    _write_json(paths.output_root / "manifest.json", manifest)
    return paths.output_root
