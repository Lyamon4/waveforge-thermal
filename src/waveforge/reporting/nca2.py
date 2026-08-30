"""Scientific reporting and diagnostic figures for stabilized NCA-2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady
from waveforge.reproducibility import artifact_sha256
from waveforge.verification.high_fidelity import replicate_design

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

PRODUCTION_SEEDS = (20260911, 20260912, 20260913)
SNAPSHOT_STEPS = (0, 1, 2, 4, 8, 16, 32, 48, 64)
SOURCE_BOUNDS = {
    "A": (0.40, 0.60, 0.62, 0.82),
    "B": (0.18, 0.38, 0.62, 0.82),
    "C": (0.62, 0.82, 0.62, 0.82),
}


def render_nca2_report(
    *,
    qualification: dict[str, Any],
    verdict: dict[str, Any],
    implementation_shas: tuple[str, ...],
    report_git_sha: str,
) -> str:
    """Render the frozen outcome without selecting or hiding production seeds."""
    campaign = verdict["campaign"]
    selected = qualification["selected_protocol"]
    lines = [
        "# WaveForge Thermal — NCA-2 stabilized training",
        "",
        f"## Verdict: `{campaign['status']}`",
        "",
        "NCA-2 — новый prospective experiment. Первый fixed-sharp experiment "
        "остаётся неизменным с вердиктом `NCA_NO_GO_EFFECT` (1/3 passing seeds).",
        "",
        f"Qualification выбрала Protocol {selected}: "
        f"`{qualification['selection_reason']}`.",
        "",
        "Primary authority: independent CPU SciPy `256×256`. Connectivity "
        "публикуется отдельно и не имеет authority над thermal verdict.",
        "",
        "| Seed | Tmax 256 | Binary fraction | Tree improvement | Primary pass | "
        "Engineering connectivity |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for item in verdict["seeds"]:
        seed_verdict = item["verdict"]
        lines.append(
            f"| {item['seed']} | {seed_verdict['peak_256']:.15g} | "
            f"{seed_verdict['binary_fraction']:.15g} | "
            f"{seed_verdict['tree_improvement']:.6%} | "
            f"{seed_verdict['primary_pass']} | "
            f"{item['engineering_connectivity_pass']} |"
        )
    lines.extend(
        (
            "",
            "## Across-seed summary",
            "",
            f"- mean Tmax: `{campaign['mean_peak_256']:.15g}`",
            f"- median Tmax: `{campaign['median_peak_256']:.15g}`",
            f"- range: `[{campaign['minimum_peak_256']:.15g}, "
            f"{campaign['maximum_peak_256']:.15g}]`",
            "",
            "## Claim limits",
            "",
            "Этот experiment проверяет одну фиксированную A/B/C-задачу. Он не "
            "доказывает generalization на unseen source layouts, self-repair, "
            "реальную chip-package geometry, CFD или data-center cooling.",
            "",
            "## Provenance",
            "",
            f"- result-producing implementation SHA(s): "
            f"`{', '.join(implementation_shas)}`",
            f"- report Git SHA: `{report_git_sha}`",
            "",
        )
    )
    return "\n".join(lines)


def _plot_training(output_dir: Path) -> Path:
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 8.5), sharex=True)
    for axis, seed in zip(axes, PRODUCTION_SEEDS, strict=True):
        frame = pd.read_csv(
            output_dir / f"production_seed_{seed}" / "optimization_metrics.csv"
        )
        axis.plot(frame["iteration"], frame["total_objective"], label="total J")
        axis.plot(frame["iteration"], frame["thermal_smooth"], label="thermal smooth")
        axis.axvline(250, color="0.4", linestyle="--", linewidth=0.8)
        axis.axvline(500, color="0.4", linestyle="--", linewidth=0.8)
        axis.set_ylabel(f"seed {seed}")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Iteration (zero-based)")
    figure.tight_layout()
    path = output_dir / "training_stability.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def _plot_designs(output_dir: Path) -> Path:
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 9.0), constrained_layout=True)
    for row, seed in enumerate(PRODUCTION_SEEDS):
        continuous = np.load(
            output_dir / f"production_seed_{seed}" / "design_continuous_64.npy",
            allow_pickle=False,
        )
        binary = np.load(
            output_dir / f"production_seed_{seed}" / "design_binary_64.npy",
            allow_pickle=False,
        )
        for column, (field, title) in enumerate(
            ((continuous, "continuous D"), (binary, "strict binary D"))
        ):
            image = axes[row, column].imshow(
                field,
                origin="lower",
                extent=(0, 1, 0, 1),
                vmin=0,
                vmax=1,
                cmap="viridis",
                interpolation="nearest",
            )
            axes[row, column].set_title(f"seed {seed}: {title}")
            axes[row, column].set_xlabel("x")
            axes[row, column].set_ylabel("y")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Material fraction")
    path = output_dir / "final_design_gallery.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def _plot_rollouts(output_dir: Path) -> Path:
    figure, axes = plt.subplots(
        3, len(SNAPSHOT_STEPS), figsize=(18.0, 6.2), constrained_layout=True
    )
    snapshots: dict[tuple[int, int], np.ndarray] = {}
    limit = 1.0e-12
    for seed in PRODUCTION_SEEDS:
        archive = np.load(
            output_dir / f"production_seed_{seed}" / "rollout_snapshots.npz",
            allow_pickle=False,
        )
        for step in SNAPSHOT_STEPS:
            field = np.asarray(archive[f"step_{step}"][0], dtype=np.float64)
            snapshots[(seed, step)] = field
            limit = max(limit, float(np.max(np.abs(field))))
    for row, seed in enumerate(PRODUCTION_SEEDS):
        for column, step in enumerate(SNAPSHOT_STEPS):
            image = axes[row, column].imshow(
                snapshots[(seed, step)],
                origin="lower",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            axes[row, column].set_title(f"t={step}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if column == 0:
                axes[row, column].set_ylabel(f"seed {seed}")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="material_logit")
    path = output_dir / "rollout_snapshots.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def _plot_temperature_details(output_dir: Path) -> Path:
    metrics = pd.read_csv(output_dir / "verified_256_metrics.csv")
    grid = Grid2D(nx=256, ny=256)
    fields: dict[tuple[int, str], np.ndarray] = {}
    for seed in PRODUCTION_SEEDS:
        binary = np.load(
            output_dir / f"production_seed_{seed}" / "design_binary_64.npy",
            allow_pickle=False,
        )
        conductivity = 1.0 + 19.0 * replicate_design(binary, factor=4) ** 3
        metric = metrics.loc[metrics["seed"] == seed].iloc[0]
        for scenario, bounds in SOURCE_BOUNDS.items():
            source = area_overlap_rectangular_source(grid, bounds, 1.0)
            temperature = solve_steady(
                grid,
                conductivity,
                source,
                BoundaryConditions.production(),
            ).temperature
            if not np.isclose(
                temperature.max(),
                metric[f"peak_{scenario}"],
                rtol=1.0e-12,
                atol=1.0e-13,
            ):
                raise RuntimeError("figure solve differs from verified metric")
            fields[(seed, scenario)] = temperature
    figure, axes = plt.subplots(3, 3, figsize=(11.0, 10.0), constrained_layout=True)
    for row, seed in enumerate(PRODUCTION_SEEDS):
        local_fields = [fields[(seed, scenario)] for scenario in ("A", "B", "C")]
        local_min = min(float(field.min()) for field in local_fields)
        local_max = max(float(field.max()) for field in local_fields)
        for column, scenario in enumerate(("A", "B", "C")):
            field = fields[(seed, scenario)]
            axis = axes[row, column]
            image = axis.imshow(
                field,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="inferno",
                vmin=local_min,
                vmax=local_max,
            )
            x0, x1, y0, y1 = SOURCE_BOUNDS[scenario]
            axis.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.2,
                )
            )
            axis.plot([0, 1], [0, 0], color="deepskyblue", linewidth=3)
            axis.set_title(f"seed {seed}, {scenario}: Tmax={float(field.max()):.6f}")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
        figure.colorbar(image, ax=axes[row, :].tolist(), label="Temperature")
    path = output_dir / "temperature_scenario_details.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def generate_nca2_report(
    output_dir: Path,
    *,
    project_root: Path,
    report_git_sha: str,
) -> dict[str, Any]:
    """Generate diagnostic figures, Russian report and canonical hash manifest."""
    qualification = json.loads(
        (output_dir / "qualification_verdict.json").read_text(encoding="utf-8")
    )
    verdict = json.loads((output_dir / "nca2_verdict.json").read_text(encoding="utf-8"))
    production = [
        json.loads(
            (
                output_dir / f"production_seed_{seed}" / "production_manifest.json"
            ).read_text(encoding="utf-8")
        )
        for seed in PRODUCTION_SEEDS
    ]
    implementation_shas = tuple(
        sorted({item["implementation_git_sha"] for item in production})
    )
    report_path = output_dir / "nca2_report.md"
    report_path.write_text(
        render_nca2_report(
            qualification=qualification,
            verdict=verdict,
            implementation_shas=implementation_shas,
            report_git_sha=report_git_sha,
        ),
        encoding="utf-8",
        newline="\n",
    )
    figures = (
        _plot_training(output_dir),
        _plot_designs(output_dir),
        _plot_rollouts(output_dir),
        _plot_temperature_details(output_dir),
    )
    old_manifest = project_root / "artifacts/pure_nca_spike/artifact_hashes.json"
    old_verdict = project_root / "artifacts/pure_nca_spike/nca_spike_verdict.json"
    artifacts = {
        path.relative_to(project_root).as_posix(): artifact_sha256(path)
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    payload = {
        "schema_version": 1,
        "hash_mode": "canonical_lf_text_raw_binary",
        "old_experiment_status": "NCA_NO_GO_EFFECT",
        "old_experiment_artifact_manifest_sha256": artifact_sha256(old_manifest),
        "old_experiment_verdict_sha256": artifact_sha256(old_verdict),
        "report_git_sha": report_git_sha,
        "figures": [path.name for path in figures],
        "artifacts": dict(sorted(artifacts.items())),
    }
    hash_path = output_dir / "artifact_hashes.json"
    hash_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload
