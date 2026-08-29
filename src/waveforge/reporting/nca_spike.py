"""Scientific artifacts for the fixed-task pure-NCA feasibility spike."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady
from waveforge.reproducibility import artifact_sha256
from waveforge.verification.high_fidelity import replicate_design

matplotlib.use("Agg")

from matplotlib import pyplot as plt

PRODUCTION_SEEDS = (20260901, 20260902, 20260903)
SNAPSHOT_STEPS = (0, 1, 2, 4, 8, 16, 32, 48, 64)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def add_comparator_roles(frame: pd.DataFrame) -> pd.DataFrame:
    """Make original, post-result and pixel-optimization roles explicit."""
    result = frame.copy(deep=True)

    def role(comparator_id: str) -> str:
        if comparator_id.startswith("waveforge_"):
            return "existing_pixel_inverse_design"
        if comparator_id == "parametric_branching_tree":
            return "post_result_geometric_challenge"
        if comparator_id == "straight_path":
            return "original_simple_baseline"
        raise ValueError(f"unknown comparator role: {comparator_id}")

    if "comparator_id" not in result:
        raise ValueError("comparator_id column is required")
    result["comparator_role"] = result["comparator_id"].map(role)
    return result


def build_scientific_verdict(
    *,
    qualification: dict[str, Any],
    production: list[dict[str, Any]],
    verification: dict[str, Any],
    reproducibility: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Combine immutable phase outputs without recalculating their decisions."""
    return {
        "schema_version": 1,
        "status": verification["campaign"]["status"],
        "scientific_scope": "pure_nca_fixed_abc_neural_reparameterization",
        "qualification": qualification,
        "production": production,
        "verification": verification,
        "reproducibility": reproducibility,
        "provenance": provenance,
        "primary_authority": "independent_cpu_scipy_256",
        "secondary_128_has_verdict_authority": False,
        "unseen_layout_claim_authorized": False,
        "tree_comparator_is_gating": False,
    }


def plot_final_design_gallery(
    continuous_designs: Mapping[int, NDArray[np.floating]],
    binary_designs: Mapping[int, NDArray[np.floating]],
    output_path: Path,
) -> Path:
    """Plot copied continuous/binary arrays without changing scientific inputs."""
    seeds = tuple(continuous_designs)
    if seeds != tuple(binary_designs) or not seeds:
        raise ValueError("continuous and binary seed registries must match")
    continuous = {
        seed: np.asarray(continuous_designs[seed], dtype=np.float64).copy()
        for seed in seeds
    }
    binary = {
        seed: np.asarray(binary_designs[seed], dtype=np.float64).copy()
        for seed in seeds
    }
    figure, axes = plt.subplots(
        len(seeds),
        2,
        figsize=(7.2, 3.1 * len(seeds)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, seed in enumerate(seeds):
        for column, (field, title) in enumerate(
            ((continuous[seed], "continuous D"), (binary[seed], "strict binary D"))
        ):
            image = axes[row, column].imshow(
                field,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                interpolation="nearest",
            )
            axes[row, column].set(title=f"seed {seed}: {title}", xlabel="x", ylabel="y")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Material fraction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def _plot_training_curves(output_dir: Path, output_path: Path) -> Path:
    figure, axes = plt.subplots(
        len(PRODUCTION_SEEDS),
        1,
        figsize=(9.0, 8.5),
        constrained_layout=True,
        sharex=True,
    )
    for axis, seed in zip(axes, PRODUCTION_SEEDS, strict=True):
        frame = pd.read_csv(
            output_dir / f"production_seed_{seed}" / "optimization_metrics.csv"
        )
        iteration = frame["iteration"].to_numpy(copy=True)
        axis.plot(
            iteration,
            frame["total_objective"].to_numpy(copy=True),
            label="total objective",
            linewidth=1.0,
        )
        axis.plot(
            iteration,
            frame["thermal_smooth"].to_numpy(copy=True),
            label="thermal smooth",
            linewidth=1.0,
        )
        axis.set(title=f"seed {seed}", ylabel="Objective")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Iteration (zero-based)")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def _plot_rollout_snapshots(output_dir: Path, output_path: Path) -> Path:
    figure, axes = plt.subplots(
        len(PRODUCTION_SEEDS),
        len(SNAPSHOT_STEPS),
        figsize=(18.0, 6.4),
        constrained_layout=True,
        squeeze=False,
    )
    snapshots: dict[int, dict[int, NDArray[np.float64]]] = {}
    extrema: list[float] = []
    for seed in PRODUCTION_SEEDS:
        archive = np.load(
            output_dir / f"production_seed_{seed}" / "rollout_snapshots.npz",
            allow_pickle=False,
        )
        snapshots[seed] = {
            step: np.asarray(archive[f"step_{step}"][0], dtype=np.float64).copy()
            for step in SNAPSHOT_STEPS
        }
        extrema.extend(
            float(np.max(np.abs(field))) for field in snapshots[seed].values()
        )
    limit = max([*extrema, 1.0e-12])
    for row, seed in enumerate(PRODUCTION_SEEDS):
        for column, step in enumerate(SNAPSHOT_STEPS):
            image = axes[row, column].imshow(
                snapshots[seed][step],
                origin="lower",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_title(f"t={step}")
            if column == 0:
                axes[row, column].set_ylabel(f"seed {seed}")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="material_logit")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _temperature_fields_256(
    binary_designs: Mapping[int, NDArray[np.floating]],
    verified_metrics: pd.DataFrame,
) -> dict[tuple[int, str], NDArray[np.float64]]:
    grid = Grid2D(nx=256, ny=256)
    bounds = {
        "A": (0.40, 0.60, 0.62, 0.82),
        "B": (0.18, 0.38, 0.62, 0.82),
        "C": (0.62, 0.82, 0.62, 0.82),
    }
    fields: dict[tuple[int, str], NDArray[np.float64]] = {}
    for seed, frozen in binary_designs.items():
        transferred = replicate_design(np.asarray(frozen, dtype=np.float64), factor=4)
        conductivity = 1.0 + 19.0 * transferred**3
        row = verified_metrics.loc[verified_metrics["seed"] == seed]
        if len(row) != 1:
            raise ValueError(f"missing unique verified row for seed {seed}")
        for scenario_id, rectangle in bounds.items():
            source = area_overlap_rectangular_source(grid, rectangle, 1.0)
            temperature = solve_steady(
                grid,
                conductivity,
                source,
                BoundaryConditions.production(),
            ).temperature
            expected_peak = float(row.iloc[0][f"peak_{scenario_id}"])
            if not np.isclose(
                np.max(temperature), expected_peak, rtol=1.0e-12, atol=1.0e-13
            ):
                raise RuntimeError("plot solve differs from frozen scientific metric")
            fields[(seed, scenario_id)] = temperature.copy()
    return fields


def _plot_temperature_maps(
    fields: Mapping[tuple[int, str], NDArray[np.float64]],
    output_path: Path,
) -> Path:
    minimum = min(float(field.min()) for field in fields.values())
    maximum = max(float(field.max()) for field in fields.values())
    figure, axes = plt.subplots(
        len(PRODUCTION_SEEDS),
        3,
        figsize=(11.2, 9.5),
        constrained_layout=True,
        squeeze=False,
    )
    for row, seed in enumerate(PRODUCTION_SEEDS):
        for column, scenario_id in enumerate(("A", "B", "C")):
            image = axes[row, column].imshow(
                fields[(seed, scenario_id)],
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="inferno",
                vmin=minimum,
                vmax=maximum,
            )
            axes[row, column].set(
                title=f"seed {seed}, scenario {scenario_id}", xlabel="x", ylabel="y"
            )
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Temperature")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def render_russian_report(
    verdict: dict[str, Any],
    comparator_metrics: pd.DataFrame,
) -> str:
    """Render a bounded Russian interpretation of registered machine fields."""
    verification = verdict["verification"]
    lines = [
        "# WaveForge Thermal — pure-NCA physics-trained spike",
        "",
        f"## Verdict: `{verdict['status']}`",
        "",
        "Эксперимент проверяет neural reparameterization только на одной "
        "фиксированной A/B/C-задаче. Он не проверяет перенос на новые source layouts "
        "и не заменяет independent physics verification.",
        "",
        "Primary authority: CPU SciPy `256×256`. Grid `128×128` используется только "
        "как secondary transfer diagnostic.",
        "",
        "## Production seeds",
        "",
        "| Seed | Status | Tmax 256 | Binary fraction | Δ(128→256) |",
        "|---:|---|---:|---:|---:|",
    ]
    for item in verification["seed_verifications"]:
        seed_verdict = item["verdict"]
        lines.append(
            f"| {seed_verdict['seed']} | `{seed_verdict['status']}` | "
            f"{seed_verdict['peak_256']:.15g} | "
            f"{seed_verdict['binary_fraction']:.15g} | "
            f"{item['relative_128_to_256_change']:.8%} |"
        )
    campaign = verification["campaign"]
    registered_seed_count = len(verification["seed_verifications"])
    lines.extend(
        (
            "",
            f"Прошёл `{campaign['passing_seed_count']}` seed из "
            f"`{registered_seed_count}`; требуется минимум "
            f"`{campaign['required_passing_seed_count']}`. Численная training path "
            "была валидной, однако "
            "preregistered reproducibility-of-effect criterion не выполнен.",
            "",
            "## Comparator diagnostics",
            "",
        )
    )
    if comparator_metrics.empty:
        lines.append("Comparator table в этом fixture не предоставлена.")
    else:
        lines.extend(
            (
                "Положительная relative difference означает меньший `Tmax` у NCA.",
                "",
                "| Seed | Comparator | Role | Relative difference |",
                "|---:|---|---|---:|",
            )
        )
        for row in comparator_metrics.to_dict(orient="records"):
            lines.append(
                f"| {int(row['seed'])} | `{row['comparator_id']}` | "
                f"`{row['comparator_role']}` | "
                f"{float(row['nca_relative_difference']):.8%} |"
            )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "Один production seed получил сильный solver-verified design. Поэтому "
            "локальное neural rule способно представить полезную охлаждающую "
            "структуру. Но два других seed не подтвердили тот же эффект; fixed sharp "
            "objective без "
            "continuation недостаточно надёжен для заявленного feasibility criterion.",
            "",
            "Этот исход является `NCA_NO_GO_EFFECT`, а не training pathology: CUDA "
            "runs завершились, gradients оставались finite, а CG converged. Возможный "
            "следующий NCA-2 experiment должен быть новым prospective protocol; "
            "текущий "
            "результат не переписывается.",
            "",
        )
    )
    return "\n".join(lines)


def write_artifact_hash_manifest(
    output_path: Path,
    artifact_paths: Sequence[Path],
    *,
    root: Path,
) -> Path:
    """Hash declared text canonically and binary artifacts as raw bytes."""
    resolved_output = output_path.resolve()
    artifacts = {
        path.resolve().relative_to(root.resolve()).as_posix(): artifact_sha256(path)
        for path in sorted(artifact_paths, key=lambda item: item.as_posix())
        if path.is_file() and path.resolve() != resolved_output
    }
    return _write_json(
        output_path,
        {
            "schema_version": 1,
            "hash_mode": "canonical_lf_text_raw_binary",
            "artifacts": artifacts,
        },
    )


def generate_nca_report(
    output_dir: Path,
    *,
    project_root: Path,
    report_generation_git_sha: str,
) -> dict[str, Any]:
    """Generate final figures, verdict, report and complete artifact hashes."""
    qualification = json.loads(
        (output_dir / "lr_qualification_verdict.json").read_text(encoding="utf-8")
    )
    verification = json.loads(
        (output_dir / "nca_verification_verdict.json").read_text(encoding="utf-8")
    )
    production = [
        json.loads(
            (
                output_dir / f"production_seed_{seed}" / "production_manifest.json"
            ).read_text(encoding="utf-8")
        )
        for seed in PRODUCTION_SEEDS
    ]
    environment = json.loads(
        (output_dir / "environment.json").read_text(encoding="utf-8")
    )
    determinism = json.loads(
        (output_dir / "determinism_preflight.json").read_text(encoding="utf-8")
    )
    reproducibility = {
        "mode": environment["determinism"]["mode"],
        "status": determinism["status"],
        "exact_two_step_replay": determinism["exact_replay"],
        "full_production_replay_required": (
            environment["determinism"]["mode"] == "topology_verdict"
        ),
    }
    provenance = {
        "config_path": "configs/pure_nca_spike.yaml",
        "config_sha256": artifact_sha256(project_root / "configs/pure_nca_spike.yaml"),
        "spec_path": (
            "docs/superpowers/specs/2026-08-29-pure-nca-physics-trained-spike-design.md"
        ),
        "spec_sha256": verification["spec_sha256"],
        "result_producing_implementation_shas": sorted(
            {manifest["implementation_git_sha"] for manifest in production}
        ),
        "verification_git_sha": verification["verification_git_sha"],
        "report_generation_git_sha": report_generation_git_sha,
        "artifact_hash_manifest": "artifacts/pure_nca_spike/artifact_hashes.json",
    }
    verdict = build_scientific_verdict(
        qualification=qualification,
        production=production,
        verification=verification,
        reproducibility=reproducibility,
        provenance=provenance,
    )
    _write_json(output_dir / "nca_spike_verdict.json", verdict)

    comparators = add_comparator_roles(
        pd.read_csv(output_dir / "comparator_metrics.csv")
    )
    comparators.to_csv(
        output_dir / "comparator_metrics.csv", index=False, lineterminator="\n"
    )
    continuous = {
        seed: np.load(
            output_dir / f"production_seed_{seed}" / "design_continuous_64.npy",
            allow_pickle=False,
        )
        for seed in PRODUCTION_SEEDS
    }
    binary = {
        seed: np.load(
            output_dir / f"production_seed_{seed}" / "design_binary_64.npy",
            allow_pickle=False,
        )
        for seed in PRODUCTION_SEEDS
    }
    _plot_training_curves(output_dir, output_dir / "training_curves.png")
    _plot_rollout_snapshots(output_dir, output_dir / "rollout_snapshots.png")
    plot_final_design_gallery(
        continuous, binary, output_dir / "final_design_gallery.png"
    )
    verified = pd.read_csv(output_dir / "verified_256_metrics.csv")
    fields = _temperature_fields_256(binary, verified)
    _plot_temperature_maps(fields, output_dir / "final_temperature_maps.png")
    (output_dir / "nca_spike_report.md").write_text(
        render_russian_report(verdict, comparators),
        encoding="utf-8",
        newline="\n",
    )
    final_paths = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    ]
    write_artifact_hash_manifest(
        output_dir / "artifact_hashes.json",
        final_paths,
        root=project_root,
    )
    return verdict
