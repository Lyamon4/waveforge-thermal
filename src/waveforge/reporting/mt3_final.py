"""Disclosure-first reporting primitives for the frozen MT3 final campaign."""

from __future__ import annotations

import json
import shutil
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from waveforge.design.epyc9754_benchmark import build_epyc9754_scale_benchmark
from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.ml.multitask_tasks import SourceLayoutTask, build_frozen_splits
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady
from waveforge.reproducibility import artifact_sha256
from waveforge.verification.high_fidelity import replicate_design

FIELD_COLOR = "#2F6BFF"
SENS_COLOR = "#E45756"
ADAM_COLOR = "#5E6B7A"
MMA_COLOR = "#F2A541"
GOOD_COLOR = "#188977"
INK = "#172033"
MUTED = "#64748B"


@dataclass(frozen=True)
class FinalFigureSpec:
    """One registered final figure and the claim boundary attached to it."""

    figure_id: str
    stem: str
    claim_limit: str


FINAL_FIGURE_SPECS: tuple[FinalFigureSpec, ...] = (
    FinalFigureSpec(
        "fig01_final_summary", "01_final_summary", "full frozen ID/OOD result"
    ),
    FinalFigureSpec(
        "fig02_id_gap_distribution", "02_id_gap_distribution", "frozen ID split only"
    ),
    FinalFigureSpec(
        "fig03_ood_gap_distribution", "03_ood_gap_distribution", "frozen OOD split only"
    ),
    FinalFigureSpec(
        "fig04_solver_verified_scatter",
        "04_solver_verified_scatter",
        "independent SciPy256 values",
    ),
    FinalFigureSpec(
        "fig05_quality_compute_pareto",
        "05_quality_compute_pareto",
        "registered evaluation budgets",
    ),
    FinalFigureSpec(
        "fig06_adam_budget_trajectory",
        "06_adam_budget_trajectory",
        "single-start Adam trajectory",
    ),
    FinalFigureSpec(
        "fig07_adam_vs_mma", "07_adam_vs_mma", "registered 600-evaluation baselines"
    ),
    FinalFigureSpec(
        "fig08_field_vs_sens", "08_field_vs_sens", "matched frozen neural variants"
    ),
    FinalFigureSpec(
        "fig09_multistart_comparison",
        "09_multistart_comparison",
        "registered 16-task subset",
    ),
    FinalFigureSpec(
        "fig10_id_topology_gallery",
        "10_id_topology_gallery",
        "rank-selected ID layouts",
    ),
    FinalFigureSpec(
        "fig11_ood_topology_gallery",
        "11_ood_topology_gallery",
        "rank-selected OOD layouts",
    ),
    FinalFigureSpec(
        "fig12_candidate_diversity", "12_candidate_diversity", "four frozen U-Net heads"
    ),
    FinalFigureSpec(
        "fig13_test_layout_atlas", "13_test_layout_atlas", "all frozen test layouts"
    ),
    FinalFigureSpec(
        "fig14_method_diagram",
        "14_method_diagram",
        "inference and evaluation accounting",
    ),
    FinalFigureSpec(
        "fig15_connectivity_diagnostics",
        "15_connectivity_diagnostics",
        "engineering diagnostic, not primary gate",
    ),
    FinalFigureSpec(
        "fig16_epyc_package_and_workloads",
        "16_epyc_package_and_workloads",
        "EPYC 9754-scale synthetic benchmark; not an exact proprietary model",
    ),
    FinalFigureSpec(
        "fig17_epyc_topology_comparison",
        "17_epyc_topology_comparison",
        "EPYC 9754-scale synthetic benchmark; secondary only",
    ),
    FinalFigureSpec(
        "fig18_epyc_temperature_maps",
        "18_epyc_temperature_maps",
        "EPYC 9754-scale synthetic benchmark; secondary only",
    ),
)

EXPECTED_FINAL_ARTIFACT_COUNTS: dict[str, int] = {
    "neural_results": 96,
    "adam_results": 96,
    "mma_results": 48,
    "verification_rows": 288,
    "primary_rows": 48,
    "multistart_rows": 16,
    "epyc_results": 1,
}


@dataclass(frozen=True)
class ConnectivityDiagnostics:
    """Strict high-k connectivity diagnostics for one binary topology."""

    component_count: int
    sink_connected_material_fraction: float
    source_contacts: tuple[bool, ...]
    engineering_connectivity_pass: bool


@dataclass(frozen=True)
class MT3FinalEvidence:
    """Complete table-level evidence needed by final statistical figures."""

    primary_rows: pd.DataFrame
    verification_rows: pd.DataFrame
    multistart_rows: pd.DataFrame
    budget_rows: pd.DataFrame
    verdicts: dict[str, object]
    epyc_result: dict[str, object]


@dataclass(frozen=True)
class MT3FinalReportPaths:
    """Frozen result, selected model, and final package locations."""

    result_root: Path
    training_root: Path
    output_root: Path


TemperatureProvider = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], ...],
]


def validate_final_artifact_counts(counts: dict[str, int]) -> None:
    """Reject incomplete or duplicated frozen final result collections."""
    for name, expected in EXPECTED_FINAL_ARTIFACT_COUNTS.items():
        observed = counts.get(name)
        if observed != expected:
            raise RuntimeError(f"{name}: expected {expected}, observed {observed}")


def ranked_layout_indices(gaps: NDArray[np.float64]) -> dict[str, int]:
    """Return deterministic best, median-rank, and worst layout indices."""
    values = np.asarray(gaps, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("gaps must be a finite nonempty vector")
    order = np.argsort(values, kind="stable")
    return {
        "best": int(order[0]),
        "median": int(order[len(order) // 2]),
        "worst": int(order[-1]),
    }


def _components(mask: NDArray[np.bool_]) -> list[set[tuple[int, int]]]:
    height, width = mask.shape
    unvisited = {tuple(index) for index in np.argwhere(mask)}
    components: list[set[tuple[int, int]]] = []
    while unvisited:
        start = min(unvisited)
        unvisited.remove(start)
        component = {start}
        queue: deque[tuple[int, int]] = deque((start,))
        while queue:
            row, column = queue.popleft()
            for candidate in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= candidate[0] < height
                    and 0 <= candidate[1] < width
                    and candidate in unvisited
                ):
                    unvisited.remove(candidate)
                    component.add(candidate)
                    queue.append(candidate)
        components.append(component)
    return components


def connectivity_diagnostics(
    design: NDArray[np.float64],
    sources: NDArray[np.float64],
) -> ConnectivityDiagnostics:
    """Measure strict high-k components and their contact with the bottom sink."""
    binary = np.asarray(design)
    source_maps = np.asarray(sources)
    if binary.ndim != 2 or not np.isin(binary, (0.0, 1.0)).all():
        raise ValueError("design must be a two-dimensional strict binary mask")
    if source_maps.ndim != 3 or source_maps.shape[1:] != binary.shape:
        raise ValueError("sources must have shape (scenario, height, width)")
    mask = binary.astype(bool, copy=False)
    material_count = int(np.count_nonzero(mask))
    components = _components(mask)
    sink_component_cells: set[tuple[int, int]] = set()
    for component in components:
        if any(row == 0 for row, _ in component):
            sink_component_cells.update(component)
    sink_mask = np.zeros_like(mask)
    for row, column in sink_component_cells:
        sink_mask[row, column] = True
    contacts = tuple(bool(np.any(sink_mask & (source > 0.0))) for source in source_maps)
    fraction = len(sink_component_cells) / material_count if material_count else 0.0
    return ConnectivityDiagnostics(
        component_count=len(components),
        sink_connected_material_fraction=float(fraction),
        source_contacts=contacts,
        engineering_connectivity_pass=bool(contacts) and all(contacts),
    )


def build_final_report_markdown(
    *,
    primary_rows: pd.DataFrame,
    verdicts: dict[str, object],
    epyc_result: dict[str, object],
    figure_stems: tuple[str, ...],
) -> str:
    """Build the disclosure-first scientific summary for frozen final results."""
    required = {
        "split",
        "primary_relative_gap",
        "candidate_tmax_scipy256",
        "strong_single_tmax_scipy256",
        "strong_single_family",
    }
    if not required.issubset(primary_rows.columns):
        raise ValueError("primary rows are incomplete")
    summaries: list[str] = []
    speedups: list[float] = []
    for split in ("test_id", "test_ood"):
        frame = primary_rows[primary_rows["split"] == split]
        if frame.empty:
            raise ValueError(f"missing {split} rows")
        gaps = frame["primary_relative_gap"].to_numpy(dtype=np.float64)
        split_payload = verdicts.get(split)
        if not isinstance(split_payload, dict):
            raise ValueError(f"missing {split} verdict")
        verdict = split_payload.get("verdict")
        if not isinstance(verdict, dict) or "status" not in verdict:
            raise ValueError(f"missing {split} verdict status")
        bootstrap = split_payload.get("bootstrap")
        if not isinstance(bootstrap, dict):
            bootstrap = {}
        lower = float(bootstrap.get("lower_bound", np.nan))
        upper = float(bootstrap.get("upper_bound", np.nan))
        speedup = float(verdict.get("equivalent_evaluation_speedup", 20.0))
        speedups.append(speedup)
        summaries.append(
            f"| {split} | {len(frame)} | {100 * np.median(gaps):.3f}% | "
            f"{100 * np.mean(gaps):.3f}% | {100 * np.quantile(gaps, 0.9):.3f}% | "
            f"{100 * np.max(gaps):.3f}% | "
            f"[{100 * np.min(gaps):.3f}%, {100 * np.max(gaps):.3f}%] | "
            f"{int(np.count_nonzero(gaps < 0.0))}/{len(frame)} | "
            f"[{100 * lower:.3f}%, {100 * upper:.3f}%] | "
            f"`{verdict['status']}` |"
        )
    if epyc_result.get("exact_proprietary_cpu_model") is not False:
        raise ValueError("EPYC disclosure flag is missing")
    if epyc_result.get("affects_primary_id_ood_verdict") is not False:
        raise ValueError("EPYC result must remain secondary")
    figures = "\n".join(f"- `{stem}`" for stem in figure_stems)
    table_header = (
        "| Split | Tasks | Median gap | Mean gap | P90 gap | Worst gap | "
        "Range | Wins | 95% bootstrap CI of median | Verdict |"
    )
    return f"""# WaveForge MT3 final frozen-test report

## Scientific scope

The preregistered primary method is `SENS_UNET_BEST4_R25`: one frozen shared
U-Net generates four candidates, four forward-only physics scores select one,
and exactly one candidate receives 25 task-specific refinement updates.
`FIELD_UNET` is reported as the matched frozen control even when it performs
better. All headline temperatures use the same independent SciPy 256x256
evaluation path as the conventional optimizers.

## Frozen ID/OOD results

{table_header}
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(summaries)}

The comparator for each task is the better of registered Adam-600 and MMA-600.
The primary method uses {min(speedups):.1f}x fewer equivalent task-specific
physics evaluations (30 versus 600). No claim of a global optimum is made.

## EPYC-scale secondary benchmark

`{epyc_result.get("label", "EPYC_9754_SCALE_SYNTHETIC")}` is a presentation-relevant
secondary stress test. It is **not an exact proprietary AMD thermal model** and
uses disclosed synthetic 360 W workload allocations on a public package scale.
It does not affect the primary ID/OOD verdict.

## Registered figures

{figures}
"""


def collect_baseline_budget_rows(result_root: Path) -> pd.DataFrame:
    """Collect registered single-start budget snapshots from Adam and MMA."""
    rows: list[dict[str, object]] = []
    for method in ("adam", "mma"):
        pattern = f"baselines/{method}/*/task_*/start_0/result.json"
        for path in sorted(result_root.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"unreadable baseline result: {path}") from error
            if payload.get("status") != "PASS" or payload.get("start_index") != 0:
                raise RuntimeError(f"invalid registered baseline result: {path}")
            snapshots = payload.get("snapshot_tmax_scipy64")
            if not isinstance(snapshots, dict) or not snapshots:
                raise RuntimeError(f"missing baseline snapshots: {path}")
            for evaluation_text, tmax in snapshots.items():
                rows.append(
                    {
                        "method": str(payload["method"]),
                        "split": str(payload["split"]),
                        "task_index": int(payload["task_index"]),
                        "start_index": 0,
                        "evaluation": int(evaluation_text),
                        "tmax_scipy64": float(tmax),
                    }
                )
    if not rows:
        raise RuntimeError("no registered baseline budget rows found")
    return pd.DataFrame(rows).sort_values(
        ["split", "task_index", "method", "evaluation"], ignore_index=True
    )


def _method_gap_frame(primary: pd.DataFrame) -> pd.DataFrame:
    frame = primary.copy()
    reference = frame["strong_single_tmax_scipy256"].to_numpy(dtype=np.float64)
    values = {
        "SENS + R25": frame["candidate_tmax_scipy256"],
        "FIELD + R25": frame["field_r25_tmax_scipy256"],
        "SENS best-of-4": frame["sens_best4_tmax_scipy256"],
        "SENS + R50": frame["sens_r50_tmax_scipy256"],
    }
    rows: list[dict[str, object]] = []
    for label, temperatures in values.items():
        gaps = temperatures.to_numpy(dtype=np.float64) / reference - 1.0
        rows.extend(
            {
                "split": split,
                "task_index": int(task_index),
                "method": label,
                "gap": float(gap),
            }
            for split, task_index, gap in zip(
                frame["split"], frame["task_index"], gaps, strict=True
            )
        )
    return pd.DataFrame(rows)


def _summary_figure(evidence: MT3FinalEvidence) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    method_gaps = _method_gap_frame(evidence.primary_rows)
    primary = method_gaps[method_gaps["method"] == "SENS + R25"]
    for split, color in (("test_id", FIELD_COLOR), ("test_ood", SENS_COLOR)):
        values = primary[primary["split"] == split]["gap"].to_numpy(dtype=float)
        label = "ID" if split == "test_id" else "OOD"
        axes[0, 0].bar(
            label,
            100 * float(np.median(values)),
            color=color,
        )
        axes[0, 1].bar(label, 100 * float(np.mean(values < 0.0)), color=color)
    axes[0, 0].axhline(0.0, color=INK, linewidth=1.0)
    axes[0, 0].set_ylabel("Median gap to stronger Adam/MMA (%)")
    axes[0, 0].set_title("Preregistered SENS + R25 quality")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_ylabel("Win rate (%)")
    axes[0, 1].set_title("Tasks below the conventional reference")
    for axis, split, title in (
        (axes[1, 0], "test_id", "Frozen ID verdict"),
        (axes[1, 1], "test_ood", "Frozen OOD verdict"),
    ):
        payload = evidence.verdicts[split]
        verdict = payload["verdict"] if isinstance(payload, dict) else {}
        status = (
            verdict.get("status", "MISSING") if isinstance(verdict, dict) else "MISSING"
        )
        axis.axis("off")
        axis.text(0.5, 0.62, str(status), ha="center", fontsize=16, weight="bold")
        axis.text(
            0.5,
            0.38,
            "Independent SciPy 256x256\nexact 25% material",
            ha="center",
            va="center",
            color=MUTED,
        )
        axis.set_title(title)
    figure.suptitle("WaveForge MT3 - frozen unseen-layout evaluation", weight="bold")
    figure.tight_layout()
    return figure


def _gap_distribution_figure(evidence: MT3FinalEvidence, split: str) -> Figure:
    frame = _method_gap_frame(evidence.primary_rows)
    frame = frame[frame["split"] == split]
    methods = ("SENS + R25", "FIELD + R25", "SENS best-of-4", "SENS + R50")
    values = [
        100 * frame[frame["method"] == method]["gap"].to_numpy() for method in methods
    ]
    figure, axis = plt.subplots(figsize=(10.0, 5.6))
    violin = axis.violinplot(values, showmedians=True, showextrema=True)
    colors = (SENS_COLOR, FIELD_COLOR, "#A78BFA", "#FB7185")
    for body, color in zip(violin["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.72)
    axis.axhline(0.0, color=INK, linewidth=1.0, label="stronger Adam/MMA")
    axis.set_xticks(range(1, len(methods) + 1), methods, rotation=10)
    axis.set_ylabel("Relative Tmax gap (%)")
    axis.set_title(f"{'ID' if split == 'test_id' else 'OOD'} quality distribution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def _scatter_figure(evidence: MT3FinalEvidence) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    for axis, (split, title) in zip(
        axes,
        (("test_id", "Unseen ID layouts"), ("test_ood", "Unseen OOD layouts")),
        strict=True,
    ):
        frame = evidence.primary_rows[evidence.primary_rows["split"] == split]
        reference = frame["strong_single_tmax_scipy256"].to_numpy(dtype=float)
        axis.scatter(
            reference,
            frame["candidate_tmax_scipy256"],
            color=SENS_COLOR,
            label="SENS + R25",
            alpha=0.82,
        )
        axis.scatter(
            reference,
            frame["field_r25_tmax_scipy256"],
            color=FIELD_COLOR,
            label="FIELD + R25",
            alpha=0.72,
        )
        limits = (
            min(reference.min(), frame["candidate_tmax_scipy256"].min()) * 0.98,
            max(reference.max(), frame["candidate_tmax_scipy256"].max()) * 1.02,
        )
        axis.plot(limits, limits, color=INK, linewidth=1.0)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel("Stronger Adam/MMA Tmax")
        axis.set_ylabel("Frozen neural method Tmax")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Solver-verified comparison - independent SciPy 256x256")
    figure.tight_layout()
    return figure


def _pareto_figure(evidence: MT3FinalEvidence) -> Figure:
    frame = evidence.verification_rows.pivot_table(
        index=["split", "task_index"], columns="family", values="worst_peak"
    )
    reference = frame[["ADAM_600", "MMA_600"]].min(axis=1)
    families = {
        "SENS best-of-4": ("SENS_UNET_BEST4", 5, "#A78BFA"),
        "SENS + R25": ("SENS_UNET_BEST4_R25", 30, SENS_COLOR),
        "FIELD + R25": ("FIELD_UNET_BEST4_R25", 30, FIELD_COLOR),
        "SENS + R50": ("SENS_UNET_BEST4_R50", 55, "#FB7185"),
        "Adam-600": ("ADAM_600", 600, ADAM_COLOR),
        "MMA-600": ("MMA_600", 600, MMA_COLOR),
    }
    figure, axis = plt.subplots(figsize=(9.0, 5.7))
    for label, (family, evaluations, color) in families.items():
        gap = 100 * float(np.median(frame[family] / reference - 1.0))
        axis.scatter(evaluations, gap, color=color, s=75)
        axis.annotate(
            label, (evaluations, gap), xytext=(5, 5), textcoords="offset points"
        )
    axis.axhline(0.0, color=INK, linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlabel("Equivalent task-specific physics evaluations (log scale)")
    axis.set_ylabel("Median gap to stronger Adam/MMA (%)")
    axis.set_title("Quality-compute trade-off across all frozen tasks")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def _budget_figure(evidence: MT3FinalEvidence) -> Figure:
    frame = evidence.budget_rows
    final = frame[frame["evaluation"] == 600].pivot_table(
        index=["split", "task_index"], columns="method", values="tmax_scipy64"
    )
    reference = final.min(axis=1).rename("reference")
    merged = frame.join(reference, on=["split", "task_index"])
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    for axis, (split, title) in zip(
        axes,
        (("test_id", "ID"), ("test_ood", "OOD")),
        strict=True,
    ):
        subset = merged[merged["split"] == split].copy()
        subset["gap"] = subset["tmax_scipy64"] / subset["reference"] - 1.0
        for method, color in (("ADAM", ADAM_COLOR), ("MMA", MMA_COLOR)):
            grouped = subset[subset["method"] == method].groupby("evaluation")["gap"]
            axis.plot(
                grouped.median().index,
                100 * grouped.median().values,
                marker="o",
                color=color,
                label=method,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Optimizer evaluations")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Median SciPy64 gap to final stronger baseline (%)")
    figure.suptitle("Conventional optimizer convergence under registered budgets")
    figure.tight_layout()
    return figure


def _adam_mma_figure(evidence: MT3FinalEvidence) -> Figure:
    frame = evidence.primary_rows.copy()
    delta = 100 * (frame["adam600_tmax_scipy256"] / frame["mma600_tmax_scipy256"] - 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for split, color in (("test_id", FIELD_COLOR), ("test_ood", SENS_COLOR)):
        mask = frame["split"] == split
        axes[0].hist(delta[mask], bins=12, alpha=0.65, color=color, label=split)
    axes[0].axvline(0.0, color=INK)
    axes[0].set_xlabel("Adam-600 gap relative to MMA-600 (%)")
    axes[0].set_ylabel("Task count")
    axes[0].set_title("Per-task baseline difference")
    axes[0].legend(frameon=False)
    winner_counts = frame["strong_single_family"].value_counts()
    axes[1].bar(
        ("ADAM", "MMA"),
        (
            int(winner_counts.get("ADAM_600", 0)),
            int(winner_counts.get("MMA_600", 0)),
        ),
        color=(ADAM_COLOR, MMA_COLOR),
    )
    axes[1].set_ylabel("Tasks won")
    axes[1].set_title("Which 600-step baseline was stronger?")
    figure.suptitle("Strong conventional comparator is selected per task")
    figure.tight_layout()
    return figure


def _field_sens_figure(evidence: MT3FinalEvidence) -> Figure:
    frame = evidence.primary_rows
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    for axis, (split, title) in zip(
        axes,
        (("test_id", "ID"), ("test_ood", "OOD")),
        strict=True,
    ):
        subset = frame[frame["split"] == split]
        sens = subset["candidate_tmax_scipy256"].to_numpy(dtype=float)
        field = subset["field_r25_tmax_scipy256"].to_numpy(dtype=float)
        limits = (
            min(sens.min(), field.min()) * 0.98,
            max(sens.max(), field.max()) * 1.02,
        )
        axis.scatter(sens, field, color=FIELD_COLOR, alpha=0.8)
        axis.plot(limits, limits, color=INK)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel("Preregistered SENS + R25 Tmax")
        axis.set_ylabel("Matched FIELD + R25 Tmax")
        field_wins = int(np.count_nonzero(field < sens))
        axis.set_title(f"{title}: FIELD wins {field_wins}/{len(subset)}")
        axis.grid(alpha=0.25)
    figure.suptitle("Matched conditioning ablation - independent SciPy 256x256")
    figure.tight_layout()
    return figure


def _multistart_figure(evidence: MT3FinalEvidence) -> Figure:
    frame = evidence.multistart_rows
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for axis, (split, title) in zip(
        axes,
        (("test_id", "ID multistart subset"), ("test_ood", "OOD multistart subset")),
        strict=True,
    ):
        subset = frame[frame["split"] == split]
        gaps = 100 * subset["relative_gap_to_adam_multistart"].to_numpy(dtype=float)
        colors = np.where(gaps < 0.0, GOOD_COLOR, SENS_COLOR)
        axis.bar(np.arange(len(subset)), gaps, color=colors)
        axis.axhline(0.0, color=INK)
        axis.set_xlabel("Registered layout index")
        axis.set_ylabel("SENS + R25 gap to Adam best-of-4 (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Fair multiple-candidate comparison against Adam multistart")
    figure.tight_layout()
    return figure


def build_statistical_figures(evidence: MT3FinalEvidence) -> dict[str, Figure]:
    """Build the first nine registered final figures from frozen tables."""
    return {
        FINAL_FIGURE_SPECS[0].stem: _summary_figure(evidence),
        FINAL_FIGURE_SPECS[1].stem: _gap_distribution_figure(evidence, "test_id"),
        FINAL_FIGURE_SPECS[2].stem: _gap_distribution_figure(evidence, "test_ood"),
        FINAL_FIGURE_SPECS[3].stem: _scatter_figure(evidence),
        FINAL_FIGURE_SPECS[4].stem: _pareto_figure(evidence),
        FINAL_FIGURE_SPECS[5].stem: _budget_figure(evidence),
        FINAL_FIGURE_SPECS[6].stem: _adam_mma_figure(evidence),
        FINAL_FIGURE_SPECS[7].stem: _field_sens_figure(evidence),
        FINAL_FIGURE_SPECS[8].stem: _multistart_figure(evidence),
    }


def _load_npz(path: Path, key: str) -> NDArray[np.float64]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            array = np.asarray(payload[key], dtype=np.float64)
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError(f"unreadable array {key}: {path}") from error
    return array


def _task_design(
    result_root: Path,
    *,
    method: str,
    split: str,
    task_index: int,
) -> NDArray[np.float64]:
    task_name = f"task_{task_index:02d}"
    if method == "SENS + R25":
        path = result_root / "neural" / "sens_unet" / split / task_name / "designs.npz"
        key = "r25_binary_design"
    elif method == "FIELD + R25":
        path = result_root / "neural" / "field_unet" / split / task_name / "designs.npz"
        key = "r25_binary_design"
    elif method == "Adam-600":
        path = (
            result_root
            / "baselines"
            / "adam"
            / split
            / task_name
            / "start_0"
            / "designs.npz"
        )
        key = "binary_600"
    elif method == "MMA-600":
        path = (
            result_root
            / "baselines"
            / "mma"
            / split
            / task_name
            / "start_0"
            / "designs.npz"
        )
        key = "binary_600"
    else:
        raise ValueError(f"unknown design method: {method}")
    design = _load_npz(path, key)
    if design.shape != (64, 64) or int(np.count_nonzero(design)) != 1024:
        raise RuntimeError(f"{method} design violates the exact binary budget")
    return design


def _draw_task_overlay(axis: plt.Axes, task: SourceLayoutTask) -> None:
    for left, right, bottom, top in task.bounds:
        axis.add_patch(
            Rectangle(
                (left, bottom),
                right - left,
                top - bottom,
                fill=False,
                edgecolor="#FFB000",
                linewidth=1.15,
            )
        )
    axis.axhline(1.0 / 64.0, color="#00A6D6", linewidth=1.4)


def _design_panel(
    axis: plt.Axes,
    design: NDArray[np.float64],
    task: SourceLayoutTask | None,
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
    if task is not None:
        _draw_task_overlay(axis, task)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=8)


def _source_maps_256(task: SourceLayoutTask) -> NDArray[np.float64]:
    grid = Grid2D(nx=256, ny=256)
    return np.stack(
        [area_overlap_rectangular_source(grid, bounds, 1.0) for bounds in task.bounds]
    ).astype(np.float64, copy=False)


def scipy256_temperature_provider(
    design: NDArray[np.float64],
    source_maps: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Compute independent SciPy256 temperature fields for a binary topology."""
    binary = np.asarray(design, dtype=np.float64)
    if binary.shape != (64, 64) or int(np.count_nonzero(binary)) != 1024:
        raise ValueError(
            "temperature fields require an exact-budget 64x64 binary design"
        )
    sources = np.asarray(source_maps, dtype=np.float64)
    if sources.ndim != 3 or sources.shape[1:] != (256, 256):
        raise ValueError("temperature fields require 256x256 source maps")
    conductivity = 1.0 + 19.0 * replicate_design(binary, factor=4)
    grid = Grid2D(nx=256, ny=256)
    fields: list[NDArray[np.float64]] = []
    for source in sources:
        result = solve_steady(
            grid, conductivity, source, BoundaryConditions.production()
        )
        if result.normalized_residual > 1.0e-10:
            raise RuntimeError("temperature-map residual exceeds tolerance")
        fields.append(np.asarray(result.temperature, dtype=np.float64))
    return tuple(fields)


def _ranked_topology_figure(
    result_root: Path,
    evidence: MT3FinalEvidence,
    *,
    split: str,
    tasks: tuple[SourceLayoutTask, ...],
    temperature_provider: TemperatureProvider,
) -> Figure:
    frame = evidence.primary_rows[evidence.primary_rows["split"] == split].sort_values(
        "task_index"
    )
    positions = ranked_layout_indices(
        frame["primary_relative_gap"].to_numpy(dtype=np.float64)
    )
    figure, axes = plt.subplots(3, 6, figsize=(15.2, 8.0))
    for row, (rank, position) in enumerate(positions.items()):
        record = frame.iloc[position]
        task_index = int(record["task_index"])
        task = tasks[task_index]
        axes[row, 0].imshow(
            task.sources.sum(axis=0),
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="Reds",
        )
        _draw_task_overlay(axes[row, 0], task)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_title(f"{rank.upper()} layout {task_index:02d}", fontsize=8)
        sens = _task_design(
            result_root, method="SENS + R25", split=split, task_index=task_index
        )
        for column, method in enumerate(
            ("SENS + R25", "FIELD + R25", "Adam-600", "MMA-600"), start=1
        ):
            design = (
                sens
                if method == "SENS + R25"
                else _task_design(
                    result_root,
                    method=method,
                    split=split,
                    task_index=task_index,
                )
            )
            if method == "SENS + R25":
                title = (
                    f"{method}\ngap={100 * float(record['primary_relative_gap']):.2f}%"
                )
            else:
                title = method
            _design_panel(axes[row, column], design, task, title)
        fields = temperature_provider(sens, _source_maps_256(task))
        worst = max(fields, key=lambda field: float(np.max(field)))
        axes[row, 5].imshow(worst, origin="lower", cmap="inferno")
        axes[row, 5].set_xticks([])
        axes[row, 5].set_yticks([])
        axes[row, 5].set_title(f"SENS worst T\nTmax={np.max(worst):.6f}", fontsize=8)
    split_label = "ID" if split == "test_id" else "OOD"
    figure.suptitle(
        f"{split_label} best, median-rank and worst frozen examples",
        weight="bold",
    )
    figure.tight_layout()
    return figure


def _candidate_figure(
    result_root: Path,
    evidence: MT3FinalEvidence,
    tasks: tuple[SourceLayoutTask, ...],
) -> Figure:
    frame = evidence.primary_rows[
        evidence.primary_rows["split"] == "test_id"
    ].sort_values("task_index")
    positions = ranked_layout_indices(
        frame["primary_relative_gap"].to_numpy(dtype=float)
    )
    figure, axes = plt.subplots(3, 6, figsize=(14.8, 7.7))
    for row, (rank, position) in enumerate(positions.items()):
        task_index = int(frame.iloc[position]["task_index"])
        task = tasks[task_index]
        directory = (
            result_root / "neural" / "sens_unet" / "test_id" / f"task_{task_index:02d}"
        )
        candidates = _load_npz(directory / "designs.npz", "candidate_binary_designs")
        payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        selected_head = int(payload["selected_head"])
        for head in range(4):
            title = f"Head {head}" + (" (selected)" if head == selected_head else "")
            _design_panel(axes[row, head], candidates[head], task, title)
        _design_panel(
            axes[row, 4],
            _load_npz(directory / "designs.npz", "r25_binary_design"),
            task,
            f"Selected + R25\n{rank} rank",
        )
        field = _task_design(
            result_root, method="FIELD + R25", split="test_id", task_index=task_index
        )
        _design_panel(axes[row, 5], field, task, "FIELD matched control")
    figure.suptitle("Four frozen candidates; physics selects one for refinement")
    figure.tight_layout()
    return figure


def _layout_atlas_figure(
    id_tasks: tuple[SourceLayoutTask, ...],
    ood_tasks: tuple[SourceLayoutTask, ...],
) -> Figure:
    figure, axes = plt.subplots(6, 8, figsize=(15.5, 11.5))
    for position, task in enumerate(id_tasks + ood_tasks):
        axis = axes.flat[position]
        axis.imshow(
            task.sources.sum(axis=0),
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="Reds",
            vmin=0,
        )
        _draw_task_overlay(axis, task)
        axis.set_xticks([])
        axis.set_yticks([])
        split = "ID" if position < 32 else "OOD"
        index = position if position < 32 else position - 32
        axis.set_title(f"{split} {index:02d}", fontsize=7)
    figure.suptitle("All 48 prospectively frozen unseen source layouts", weight="bold")
    figure.tight_layout()
    return figure


def _method_figure() -> Figure:
    figure, axis = plt.subplots(figsize=(14.0, 4.7))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 4.7)
    axis.axis("off")
    boxes = (
        (0.2, 1.4, 2.0, "Unseen task", "sources + sink"),
        (2.8, 1.4, 2.0, "Physics probe", "Tmean/Tmax\n+sensitivity"),
        (5.4, 1.4, 2.0, "Frozen U-Net", "4 topology heads"),
        (8.0, 1.4, 2.0, "Fast score", "4 forward solves"),
        (10.6, 1.4, 2.8, "One refinement", "best candidate only\n25 updates"),
    )
    for x, y, width, title, body in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                1.6,
                boxstyle="round,pad=0.04,rounding_size=0.09",
                facecolor="#F3F6FA",
                edgecolor=INK,
            )
        )
        axis.text(x + width / 2, y + 1.15, title, ha="center", weight="bold")
        axis.text(x + width / 2, y + 0.55, body, ha="center", va="center")
    for first, second in pairwise(boxes):
        axis.add_patch(
            FancyArrowPatch(
                (first[0] + first[2] + 0.06, 2.2),
                (second[0] - 0.06, 2.2),
                arrowstyle="-|>",
                mutation_scale=15,
                color=SENS_COLOR,
            )
        )
    axis.text(
        7.0,
        4.1,
        "Teacher-free training; frozen weights on every final test layout",
        ha="center",
        fontsize=15,
        weight="bold",
    )
    axis.text(
        7.0,
        0.55,
        "30 equivalent task-specific evaluations versus 600 "
        "for each conventional baseline",
        ha="center",
        color=MUTED,
    )
    return figure


def _connectivity_figure(frame: pd.DataFrame) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11.7, 4.9))
    methods = ("SENS + R25", "FIELD + R25", "Adam-600", "MMA-600")
    values = [
        100
        * frame[frame["method"] == method][
            "sink_connected_material_fraction"
        ].to_numpy()
        for method in methods
    ]
    axes[0].boxplot(values, tick_labels=methods, showfliers=False)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Sink-connected high-k material (%)")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(axis="y", alpha=0.25)
    rates = [
        100 * frame[frame["method"] == method]["engineering_connectivity_pass"].mean()
        for method in methods
    ]
    axes[1].bar(methods, rates, color=(SENS_COLOR, FIELD_COLOR, ADAM_COLOR, MMA_COLOR))
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("All three sources contact sink component (%)")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Strict high-k connectivity - engineering diagnostic, not thermal gate"
    )
    figure.tight_layout()
    return figure


def _epyc_package_figure() -> Figure:
    benchmark = build_epyc9754_scale_benchmark(resolution=64)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    for axis, workload in zip(axes, benchmark.workloads, strict=True):
        powers = workload.region_powers_watts
        maximum = max(powers)
        axis.add_patch(
            Rectangle(
                (0, 0),
                benchmark.package_size_mm[0],
                benchmark.package_size_mm[1],
                facecolor="#E8EDF3",
                edgecolor=INK,
                linewidth=1.5,
            )
        )
        for region, power in zip(benchmark.regions, powers, strict=True):
            x, y = region.center_mm
            width, height = region.size_mm
            color = plt.cm.inferno(0.2 + 0.75 * power / maximum)
            axis.add_patch(
                Rectangle(
                    (x - width / 2, y - height / 2),
                    width,
                    height,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
            axis.text(
                x,
                y,
                f"{region.region_id}\n{power:.0f} W",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white",
            )
        axis.set_xlim(0, benchmark.package_size_mm[0])
        axis.set_ylim(0, benchmark.package_size_mm[1])
        axis.set_aspect("equal")
        axis.set_xlabel("Package width (mm)")
        axis.set_ylabel("Package height (mm)")
        axis.set_title(f"{workload.workload_id}\nTotal = 360 W")
    figure.suptitle(
        "AMD EPYC 9754-scale synthetic package benchmark - 75.4 x 72.0 mm",
        weight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Public package scale and chiplet count; synthetic region geometry/power "
        "maps, not an exact proprietary AMD thermal model.",
        ha="center",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    return figure


def _epyc_topology_figure(result_root: Path, evidence: MT3FinalEvidence) -> Figure:
    arrays_path = result_root / "epyc9754_scale_synthetic" / "designs.npz"
    verified = evidence.epyc_result["verified_scipy256"]
    families = (
        ("SENS_UNET_BEST4", "sens_best4_binary"),
        ("SENS_UNET_BEST4_R25", "sens_r25_binary"),
        ("SENS_UNET_BEST4_R50", "sens_r50_binary"),
        ("ADAM_600", "adam600_binary"),
        ("MMA_600", "mma600_binary"),
    )
    figure, axes = plt.subplots(1, 5, figsize=(15.2, 3.7))
    for axis, (family, key) in zip(axes, families, strict=True):
        design = _load_npz(arrays_path, key)
        peak = float(verified[family]["worst_peak"])
        _design_panel(axis, design, None, f"{family}\nTmax={peak:.6f}")
        axis.text(
            0.02,
            0.02,
            "sink",
            transform=axis.transAxes,
            color="#00A6D6",
            fontsize=7,
        )
    figure.suptitle("EPYC 9754-scale synthetic: frozen AI versus 600-step optimizers")
    figure.text(
        0.5,
        0.015,
        "One 64x64 cell represents about 1.18 x 1.13 mm on this scale; "
        "secondary benchmark only.",
        ha="center",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.92))
    return figure


def _epyc_temperature_figure(
    result_root: Path,
    evidence: MT3FinalEvidence,
    temperature_provider: TemperatureProvider,
) -> Figure:
    benchmark = build_epyc9754_scale_benchmark(resolution=256)
    arrays_path = result_root / "epyc9754_scale_synthetic" / "designs.npz"
    strong = str(evidence.epyc_result["strong_single_family"])
    baseline_key = epyc_strong_design_key(strong)
    methods = (
        ("SENS + R25", _load_npz(arrays_path, "sens_r25_binary")),
        (f"Strong single: {strong}-600", _load_npz(arrays_path, baseline_key)),
    )
    fields = [temperature_provider(design, benchmark.sources) for _, design in methods]
    vmin = min(
        float(np.min(field)) for method_fields in fields for field in method_fields
    )
    vmax = max(
        float(np.max(field)) for method_fields in fields for field in method_fields
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.7))
    image = None
    for row, ((label, _), method_fields) in enumerate(
        zip(methods, fields, strict=True)
    ):
        for column, (workload, field) in enumerate(
            zip(benchmark.workloads, method_fields, strict=True)
        ):
            image = axes[row, column].imshow(
                field,
                origin="lower",
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_title(
                f"{label}\n{workload.workload_id}\nTmax={np.max(field):.6f}",
                fontsize=8,
            )
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.022,
            pad=0.025,
            label="Temperature rise (model units)",
        )
    figure.suptitle("EPYC 9754-scale synthetic workload temperature fields")
    figure.text(
        0.5,
        0.015,
        "Shared normalization; synthetic 360 W workload maps; not a calibrated "
        "junction-temperature prediction.",
        ha="center",
        color=MUTED,
        fontsize=8,
    )
    figure.subplots_adjust(top=0.89, bottom=0.07, right=0.89, wspace=0.08, hspace=0.28)
    return figure


def epyc_strong_design_key(family: str) -> str:
    """Map the registered EPYC baseline family to its stored binary array."""
    mapping = {
        "ADAM_600": "adam600_binary",
        "MMA_600": "mma600_binary",
    }
    try:
        return mapping[family]
    except KeyError as error:
        raise ValueError("EPYC strong family is not registered") from error


def build_spatial_figures(
    *,
    result_root: Path,
    evidence: MT3FinalEvidence,
    temperature_provider: TemperatureProvider = scipy256_temperature_provider,
) -> tuple[dict[str, Figure], pd.DataFrame]:
    """Build registered topology, connectivity, and synthetic EPYC figures."""
    splits = build_frozen_splits()
    diagnostics: list[dict[str, object]] = []
    for split, tasks in (("test_id", splits.test_id), ("test_ood", splits.test_ood)):
        for task_index, task in enumerate(tasks):
            for method in ("SENS + R25", "FIELD + R25", "Adam-600", "MMA-600"):
                design = _task_design(
                    result_root,
                    method=method,
                    split=split,
                    task_index=task_index,
                )
                result = connectivity_diagnostics(design, task.sources)
                diagnostics.append(
                    {
                        "split": split,
                        "task_index": task_index,
                        "method": method,
                        "component_count": result.component_count,
                        "sink_connected_material_fraction": (
                            result.sink_connected_material_fraction
                        ),
                        "source_a_contact": result.source_contacts[0],
                        "source_b_contact": result.source_contacts[1],
                        "source_c_contact": result.source_contacts[2],
                        "engineering_connectivity_pass": (
                            result.engineering_connectivity_pass
                        ),
                    }
                )
    diagnostic_frame = pd.DataFrame(diagnostics)
    figures = {
        FINAL_FIGURE_SPECS[9].stem: _ranked_topology_figure(
            result_root,
            evidence,
            split="test_id",
            tasks=splits.test_id,
            temperature_provider=temperature_provider,
        ),
        FINAL_FIGURE_SPECS[10].stem: _ranked_topology_figure(
            result_root,
            evidence,
            split="test_ood",
            tasks=splits.test_ood,
            temperature_provider=temperature_provider,
        ),
        FINAL_FIGURE_SPECS[11].stem: _candidate_figure(
            result_root, evidence, splits.test_id
        ),
        FINAL_FIGURE_SPECS[12].stem: _layout_atlas_figure(
            splits.test_id, splits.test_ood
        ),
        FINAL_FIGURE_SPECS[13].stem: _method_figure(),
        FINAL_FIGURE_SPECS[14].stem: _connectivity_figure(diagnostic_frame),
        FINAL_FIGURE_SPECS[15].stem: _epyc_package_figure(),
        FINAL_FIGURE_SPECS[16].stem: _epyc_topology_figure(result_root, evidence),
        FINAL_FIGURE_SPECS[17].stem: _epyc_temperature_figure(
            result_root, evidence, temperature_provider
        ),
    }
    return figures, diagnostic_frame


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return payload


def load_mt3_final_evidence(result_root: Path) -> MT3FinalEvidence:
    """Load a complete frozen result set and reject omissions or duplicates."""
    verification_root = result_root / "verification"
    id_rows = pd.read_csv(verification_root / "test_id_primary_rows.csv")
    ood_rows = pd.read_csv(verification_root / "test_ood_primary_rows.csv")
    primary = pd.concat((id_rows, ood_rows), ignore_index=True)
    verification = pd.read_csv(verification_root / "all_scipy256_rows.csv")
    multistart = pd.read_csv(verification_root / "adam_multistart_rows.csv")
    verdicts = _read_json(verification_root / "final_verdicts.json")
    epyc_path = result_root / "epyc9754_scale_synthetic" / "result.json"
    epyc = _read_json(epyc_path)
    counts = {
        "neural_results": len(list(result_root.glob("neural/*/*/task_*/result.json"))),
        "adam_results": len(
            list(result_root.glob("baselines/adam/*/task_*/start_*/result.json"))
        ),
        "mma_results": len(
            list(result_root.glob("baselines/mma/*/task_*/start_0/result.json"))
        ),
        "verification_rows": len(verification),
        "primary_rows": len(primary),
        "multistart_rows": len(multistart),
        "epyc_results": int(epyc_path.is_file()),
    }
    validate_final_artifact_counts(counts)
    required_families = {
        "SENS_UNET_BEST4_R25",
        "FIELD_UNET_BEST4_R25",
        "SENS_UNET_BEST4",
        "SENS_UNET_BEST4_R50",
        "ADAM_600",
        "MMA_600",
    }
    if set(verification["family"]) != required_families:
        raise RuntimeError("verification family registry is incomplete")
    expected_tasks = {"test_id": 32, "test_ood": 16}
    for split, expected in expected_tasks.items():
        split_primary = primary[primary["split"] == split]
        if len(split_primary) != expected or sorted(
            split_primary["task_index"].astype(int)
        ) != list(range(expected)):
            raise RuntimeError(f"{split} primary task registry is incomplete")
        split_verification = verification[verification["split"] == split]
        if len(split_verification) != expected * len(required_families):
            raise RuntimeError(f"{split} verification registry is incomplete")
    manifest = _read_json(result_root / "opened_task_manifest.json")
    frozen = build_frozen_splits()
    for split, tasks in (("test_id", frozen.test_id), ("test_ood", frozen.test_ood)):
        rows = manifest.get(split)
        if not isinstance(rows, list) or len(rows) != len(tasks):
            raise RuntimeError(f"opened {split} manifest is incomplete")
        observed_ids = [str(row["task_id"]) for row in rows if isinstance(row, dict)]
        if observed_ids != [task.task_id for task in tasks]:
            raise RuntimeError(f"opened {split} manifest changed")
    if epyc.get("exact_proprietary_cpu_model") is not False:
        raise RuntimeError("EPYC proprietary-model disclosure is missing")
    if epyc.get("affects_primary_id_ood_verdict") is not False:
        raise RuntimeError("EPYC result may not affect the primary verdict")
    return MT3FinalEvidence(
        primary_rows=primary,
        verification_rows=verification,
        multistart_rows=multistart,
        budget_rows=collect_baseline_budget_rows(result_root),
        verdicts=verdicts,
        epyc_result=epyc,
    )


def _save_figure_triplet(
    figure: Figure,
    output_dir: Path,
    stem: str,
    *,
    dpi: int,
) -> tuple[Path, Path, Path]:
    if Path(stem).name != stem or dpi <= 0:
        raise ValueError("invalid figure output settings")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_dir / f"{stem}.{suffix}" for suffix in ("png", "svg", "pdf"))
    figure.savefig(paths[0], dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(paths[1], bbox_inches="tight", facecolor="white")
    figure.savefig(paths[2], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return paths  # type: ignore[return-value]


def _performance_table(evidence: MT3FinalEvidence) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("test_id", "test_ood"):
        frame = evidence.primary_rows[evidence.primary_rows["split"] == split]
        reference = frame["strong_single_tmax_scipy256"].to_numpy(dtype=float)
        methods = {
            "SENS_UNET_BEST4_R25": frame["candidate_tmax_scipy256"],
            "FIELD_UNET_BEST4_R25": frame["field_r25_tmax_scipy256"],
            "SENS_UNET_BEST4": frame["sens_best4_tmax_scipy256"],
            "SENS_UNET_BEST4_R50": frame["sens_r50_tmax_scipy256"],
            "STRONGER_ADAM_OR_MMA_600": frame["strong_single_tmax_scipy256"],
        }
        for method, temperatures in methods.items():
            values = temperatures.to_numpy(dtype=float)
            gaps = values / reference - 1.0
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "task_count": len(frame),
                    "median_tmax_scipy256": float(np.median(values)),
                    "mean_tmax_scipy256": float(np.mean(values)),
                    "minimum_tmax_scipy256": float(np.min(values)),
                    "maximum_tmax_scipy256": float(np.max(values)),
                    "median_relative_gap": float(np.median(gaps)),
                    "p90_relative_gap": float(np.quantile(gaps, 0.9)),
                    "worst_relative_gap": float(np.max(gaps)),
                    "win_count": int(np.count_nonzero(gaps < 0.0)),
                    "win_rate": float(np.mean(gaps < 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _readme_ru(evidence: MT3FinalEvidence) -> str:
    lines = [
        "# WaveForge MT3 - финальный пакет",
        "",
        "Это полный результат замороженного ID/OOD теста. Все значения Tmax в главном",
        "сравнении получены одним независимым SciPy solver на сетке 256x256.",
        "",
        "Главный заранее зафиксированный метод: `SENS_UNET_BEST4_R25`.",
        "FIELD-модель опубликована рядом как честный matched control.",
        "",
    ]
    for split, label in (("test_id", "ID"), ("test_ood", "OOD")):
        frame = evidence.primary_rows[evidence.primary_rows["split"] == split]
        gaps = frame["primary_relative_gap"].to_numpy(dtype=float)
        verdict_payload = evidence.verdicts[split]
        verdict = (
            verdict_payload["verdict"] if isinstance(verdict_payload, dict) else {}
        )
        status = (
            verdict.get("status", "MISSING") if isinstance(verdict, dict) else "MISSING"
        )
        lines.extend(
            (
                f"- {label}: `{status}`; median gap {100 * np.median(gaps):.3f}%; "
                f"wins {np.count_nonzero(gaps < 0)}/{len(gaps)}.",
            )
        )
    lines.extend(
        (
            "",
            "`figures/` содержит 18 paper-grade фигур в PNG 300 dpi, SVG и PDF.",
            "`models/` содержит обе выбранные полностью готовые модели и их SHA256.",
            "EPYC-фигуры являются отдельным synthetic scale benchmark и не имитируют",
            "закрытый внутренний thermal stack настоящего процессора.",
            "",
        )
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_mt3_final_package(
    paths: MT3FinalReportPaths,
    *,
    dpi: int = 300,
    temperature_provider: TemperatureProvider = scipy256_temperature_provider,
) -> Path:
    """Build the complete paper/report/model package from frozen final artifacts."""
    evidence = load_mt3_final_evidence(paths.result_root)
    output = paths.output_root
    figures_dir = output / "figures"
    data_dir = output / "data"
    models_dir = output / "models"
    for directory in (figures_dir, data_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)
    figures = build_statistical_figures(evidence)
    spatial, connectivity = build_spatial_figures(
        result_root=paths.result_root,
        evidence=evidence,
        temperature_provider=temperature_provider,
    )
    figures.update(spatial)
    expected_stems = {spec.stem for spec in FINAL_FIGURE_SPECS}
    if set(figures) != expected_stems:
        raise RuntimeError("final figure registry is incomplete")
    for stem, figure in figures.items():
        _save_figure_triplet(figure, figures_dir, stem, dpi=dpi)

    evidence.primary_rows.to_csv(data_dir / "primary_rows.csv", index=False)
    evidence.verification_rows.to_csv(data_dir / "all_scipy256_rows.csv", index=False)
    evidence.multistart_rows.to_csv(data_dir / "adam_multistart_rows.csv", index=False)
    evidence.budget_rows.to_csv(data_dir / "baseline_budget_rows.csv", index=False)
    connectivity.to_csv(data_dir / "connectivity_diagnostics.csv", index=False)
    _performance_table(evidence).to_csv(output / "performance_table.csv", index=False)
    _write_json(output / "final_verdicts.json", evidence.verdicts)
    _write_json(output / "epyc9754_scale_synthetic_result.json", evidence.epyc_result)

    model_hashes: dict[str, str] = {}
    for variant in ("field_unet", "sens_unet"):
        source = paths.training_root / variant / "checkpoint_004000.pt"
        if not source.is_file():
            raise RuntimeError(f"selected model is missing: {source}")
        destination = models_dir / f"{variant}_selected.pt"
        shutil.copy2(source, destination)
        model_hashes[destination.name] = artifact_sha256(destination)
    _write_json(models_dir / "model_hashes.json", model_hashes)
    (models_dir / "README.md").write_text(
        "# Frozen WaveForge MT3 models\n\n"
        "Both files are update-4000 checkpoints. `sens_unet_selected.pt` is the "
        "preregistered primary model; `field_unet_selected.pt` is the matched "
        "control.\n",
        encoding="utf-8",
        newline="\n",
    )

    report = build_final_report_markdown(
        primary_rows=evidence.primary_rows,
        verdicts=evidence.verdicts,
        epyc_result=evidence.epyc_result,
        figure_stems=tuple(spec.stem for spec in FINAL_FIGURE_SPECS),
    )
    (output / "MT3_FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    (output / "README_RU.md").write_text(
        _readme_ru(evidence), encoding="utf-8", newline="\n"
    )
    guide = "# Final figure guide\n\n" + "\n".join(
        f"- `{spec.stem}` - {spec.claim_limit}." for spec in FINAL_FIGURE_SPECS
    )
    (output / "FIGURE_GUIDE.md").write_text(
        guide + "\n", encoding="utf-8", newline="\n"
    )
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "scope": "frozen_final_id_ood_plus_secondary_epyc_synthetic",
        "figure_count": len(FINAL_FIGURE_SPECS),
        "png_dpi": dpi,
        "test_id_accessed": True,
        "test_ood_accessed": True,
        "files": {
            path.relative_to(output).as_posix(): artifact_sha256(path) for path in files
        },
    }
    _write_json(output / "manifest.json", manifest)
    return output
