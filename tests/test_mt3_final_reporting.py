from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from waveforge.experiments.report_mt3_final import _parser
from waveforge.reporting.mt3_final import (
    FINAL_FIGURE_SPECS,
    MT3FinalEvidence,
    MT3FinalReportPaths,
    build_final_report_markdown,
    build_mt3_final_package,
    build_spatial_figures,
    build_statistical_figures,
    collect_baseline_budget_rows,
    connectivity_diagnostics,
    epyc_strong_design_key,
    load_mt3_final_evidence,
    ranked_layout_indices,
    validate_final_artifact_counts,
)


def test_final_figure_registry_is_complete_unique_and_disclosure_first() -> None:
    assert len(FINAL_FIGURE_SPECS) == 18
    assert len({spec.figure_id for spec in FINAL_FIGURE_SPECS}) == 18
    assert len({spec.stem for spec in FINAL_FIGURE_SPECS}) == 18
    assert {spec.figure_id for spec in FINAL_FIGURE_SPECS} == {
        "fig01_final_summary",
        "fig02_id_gap_distribution",
        "fig03_ood_gap_distribution",
        "fig04_solver_verified_scatter",
        "fig05_quality_compute_pareto",
        "fig06_adam_budget_trajectory",
        "fig07_adam_vs_mma",
        "fig08_field_vs_sens",
        "fig09_multistart_comparison",
        "fig10_id_topology_gallery",
        "fig11_ood_topology_gallery",
        "fig12_candidate_diversity",
        "fig13_test_layout_atlas",
        "fig14_method_diagram",
        "fig15_connectivity_diagnostics",
        "fig16_epyc_package_and_workloads",
        "fig17_epyc_topology_comparison",
        "fig18_epyc_temperature_maps",
    }
    epyc_specs = [spec for spec in FINAL_FIGURE_SPECS if "epyc" in spec.figure_id]
    assert len(epyc_specs) == 3
    assert all("synthetic" in spec.claim_limit.lower() for spec in epyc_specs)


def test_final_report_cli_registers_result_training_and_output_roots() -> None:
    arguments = _parser().parse_args(
        [
            "--result-root",
            "results",
            "--training-root",
            "training",
            "--output",
            "package",
        ]
    )

    assert arguments.result_root.name == "results"
    assert arguments.training_root.name == "training"
    assert arguments.output.name == "package"
    assert arguments.dpi == 300


def test_ranked_layout_indices_are_deterministic_best_median_worst() -> None:
    gaps = np.asarray([0.03, -0.02, 0.01, 0.08, 0.00], dtype=np.float64)

    result = ranked_layout_indices(gaps)

    assert result == {"best": 1, "median": 2, "worst": 3}


def test_connectivity_diagnostics_reports_sink_fraction_and_source_contacts() -> None:
    design = np.zeros((8, 8), dtype=np.float64)
    design[0:6, 1] = 1.0
    design[3, 1:5] = 1.0
    design[6:8, 7] = 1.0
    sources = np.zeros((3, 8, 8), dtype=np.float64)
    sources[0, 5, 1] = 1.0
    sources[1, 3, 4] = 1.0
    sources[2, 7, 7] = 1.0

    result = connectivity_diagnostics(design, sources)

    assert result.component_count == 2
    assert result.sink_connected_material_fraction == pytest.approx(9 / 11)
    assert result.source_contacts == (True, True, False)
    assert result.engineering_connectivity_pass is False


def test_final_report_discloses_primary_control_and_secondary_epyc() -> None:
    primary = pd.DataFrame(
        {
            "split": ["test_id", "test_id", "test_ood", "test_ood"],
            "primary_relative_gap": [-0.04, -0.02, 0.01, 0.03],
            "candidate_tmax_scipy256": [0.16, 0.17, 0.18, 0.19],
            "strong_single_tmax_scipy256": [0.17, 0.18, 0.178, 0.184],
            "strong_single_family": ["ADAM_600", "MMA_600", "MMA_600", "ADAM_600"],
        }
    )
    verdicts = {
        "test_id": {"verdict": {"status": "MT3_BEATS_SINGLE_START_ID"}},
        "test_ood": {"verdict": {"status": "MT3_COMPETITIVE_OOD"}},
    }
    epyc = {
        "label": "EPYC_9754_SCALE_SYNTHETIC",
        "exact_proprietary_cpu_model": False,
        "affects_primary_id_ood_verdict": False,
        "sens_r25_relative_gap": 0.04,
    }

    report = build_final_report_markdown(
        primary_rows=primary,
        verdicts=verdicts,
        epyc_result=epyc,
        figure_stems=tuple(spec.stem for spec in FINAL_FIGURE_SPECS),
    )

    assert "SENS_UNET_BEST4_R25" in report
    assert "preregistered primary" in report
    assert "FIELD_UNET" in report
    assert "independent SciPy 256x256" in report
    assert "EPYC_9754_SCALE_SYNTHETIC" in report
    assert "not an exact proprietary AMD thermal model" in report
    assert "does not affect the primary ID/OOD verdict" in report


def test_baseline_budget_rows_use_registered_start_zero_trajectories(tmp_path) -> None:
    import json

    for method, values in (
        ("adam", {"25": 0.30, "50": 0.25, "600": 0.16}),
        ("mma", {"25": 0.28, "50": 0.22, "600": 0.15}),
    ):
        destination = (
            tmp_path / "baselines" / method / "test_id" / "task_00" / "start_0"
        )
        destination.mkdir(parents=True)
        (destination / "result.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "method": method.upper(),
                    "split": "test_id",
                    "task_index": 0,
                    "start_index": 0,
                    "snapshot_tmax_scipy64": values,
                }
            ),
            encoding="utf-8",
        )
    ignored = tmp_path / "baselines" / "adam" / "test_id" / "task_00" / "start_1"
    ignored.mkdir(parents=True)
    (ignored / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "method": "ADAM",
                "split": "test_id",
                "task_index": 0,
                "start_index": 1,
                "snapshot_tmax_scipy64": {"600": 0.14},
            }
        ),
        encoding="utf-8",
    )

    frame = collect_baseline_budget_rows(tmp_path)

    assert len(frame) == 6
    assert set(frame["method"]) == {"ADAM", "MMA"}
    assert set(frame["evaluation"]) == {25, 50, 600}
    assert set(frame["start_index"]) == {0}


def test_epyc_strong_design_key_accepts_registered_family_names() -> None:
    assert epyc_strong_design_key("ADAM_600") == "adam600_binary"
    assert epyc_strong_design_key("MMA_600") == "mma600_binary"
    with pytest.raises(ValueError, match="registered"):
        epyc_strong_design_key("ADAM")


def test_final_artifact_count_gate_is_fail_closed() -> None:
    counts = {
        "neural_results": 96,
        "adam_results": 96,
        "mma_results": 48,
        "verification_rows": 288,
        "primary_rows": 48,
        "multistart_rows": 16,
        "epyc_results": 1,
    }

    validate_final_artifact_counts(counts)

    counts["mma_results"] = 47
    with pytest.raises(RuntimeError, match="mma_results"):
        validate_final_artifact_counts(counts)


def test_statistical_figure_builders_cover_frozen_quality_and_compute_views() -> None:
    primary_rows = []
    verification_rows = []
    budget_rows = []
    for split, count in (("test_id", 32), ("test_ood", 16)):
        for index in range(count):
            reference = 0.16 + 0.001 * index
            adam = reference * 1.002
            mma = reference
            sens = reference * (0.98 + 0.001 * (index % 5))
            field = reference * (0.97 + 0.001 * (index % 4))
            primary_rows.append(
                {
                    "split": split,
                    "task_index": index,
                    "candidate_tmax_scipy256": sens,
                    "adam600_tmax_scipy256": adam,
                    "mma600_tmax_scipy256": mma,
                    "strong_single_tmax_scipy256": reference,
                    "strong_single_family": "MMA_600",
                    "primary_relative_gap": sens / reference - 1.0,
                    "field_r25_tmax_scipy256": field,
                    "sens_best4_tmax_scipy256": sens * 1.02,
                    "sens_r50_tmax_scipy256": sens * 0.999,
                }
            )
            families = {
                "SENS_UNET_BEST4_R25": sens,
                "FIELD_UNET_BEST4_R25": field,
                "SENS_UNET_BEST4": sens * 1.02,
                "SENS_UNET_BEST4_R50": sens * 0.999,
                "ADAM_600": adam,
                "MMA_600": mma,
            }
            verification_rows.extend(
                {
                    "split": split,
                    "task_index": index,
                    "family": family,
                    "worst_peak": value,
                }
                for family, value in families.items()
            )
            for method, final in (("ADAM", adam), ("MMA", mma)):
                for evaluation, multiplier in (
                    (25, 1.6),
                    (50, 1.4),
                    (100, 1.25),
                    (200, 1.12),
                    (600, 1.0),
                ):
                    budget_rows.append(
                        {
                            "method": method,
                            "split": split,
                            "task_index": index,
                            "start_index": 0,
                            "evaluation": evaluation,
                            "tmax_scipy64": final * multiplier,
                        }
                    )
    multistart = pd.DataFrame(
        [
            {
                "split": split,
                "task_index": index,
                "sens_r25_tmax_scipy256": 0.16,
                "adam_multistart_tmax_scipy256": 0.165,
                "relative_gap_to_adam_multistart": -0.03,
            }
            for split in ("test_id", "test_ood")
            for index in range(8)
        ]
    )
    evidence = MT3FinalEvidence(
        primary_rows=pd.DataFrame(primary_rows),
        verification_rows=pd.DataFrame(verification_rows),
        multistart_rows=multistart,
        budget_rows=pd.DataFrame(budget_rows),
        verdicts={
            "test_id": {"verdict": {"status": "MT3_BEATS_SINGLE_START_ID"}},
            "test_ood": {"verdict": {"status": "MT3_BEATS_SINGLE_START_OOD"}},
        },
        epyc_result={
            "label": "EPYC_9754_SCALE_SYNTHETIC",
            "exact_proprietary_cpu_model": False,
            "affects_primary_id_ood_verdict": False,
        },
    )

    figures = build_statistical_figures(evidence)

    assert set(figures) == {spec.stem for spec in FINAL_FIGURE_SPECS[:9]}
    assert (
        sum(patch.get_height() for patch in figures["07_adam_vs_mma"].axes[1].patches)
        == 48
    )
    assert all(figure.axes for figure in figures.values())
    for figure in figures.values():
        plt.close(figure)


def test_spatial_figure_builders_cover_topologies_connectivity_and_epyc(
    tmp_path,
) -> None:
    from waveforge.design.epyc9754_benchmark import build_epyc9754_scale_benchmark
    from waveforge.ml.multitask_tasks import build_frozen_splits

    binary = np.zeros((64, 64), dtype=np.float64)
    binary[:, :16] = 1.0
    candidates = np.stack([np.roll(binary, shift, axis=1) for shift in range(4)])
    primary_rows = []
    verification_rows = []
    splits = build_frozen_splits()
    for split, tasks in (("test_id", splits.test_id), ("test_ood", splits.test_ood)):
        for index, task in enumerate(tasks):
            reference = 0.16 + 0.001 * index
            sens = reference * (0.98 + 0.001 * (index % 5))
            field = reference * (0.97 + 0.001 * (index % 4))
            primary_rows.append(
                {
                    "split": split,
                    "task_index": index,
                    "candidate_tmax_scipy256": sens,
                    "adam600_tmax_scipy256": reference * 1.002,
                    "mma600_tmax_scipy256": reference,
                    "strong_single_tmax_scipy256": reference,
                    "strong_single_family": "MMA_600",
                    "primary_relative_gap": sens / reference - 1.0,
                    "field_r25_tmax_scipy256": field,
                    "sens_best4_tmax_scipy256": sens * 1.02,
                    "sens_r50_tmax_scipy256": sens * 0.999,
                }
            )
            families = {
                "SENS_UNET_BEST4_R25": sens,
                "FIELD_UNET_BEST4_R25": field,
                "SENS_UNET_BEST4": sens * 1.02,
                "SENS_UNET_BEST4_R50": sens * 0.999,
                "ADAM_600": reference * 1.002,
                "MMA_600": reference,
            }
            verification_rows.extend(
                {
                    "split": split,
                    "task_index": index,
                    "family": family,
                    "worst_peak": value,
                }
                for family, value in families.items()
            )
            for variant in ("sens_unet", "field_unet"):
                destination = (
                    tmp_path / "neural" / variant / split / f"task_{index:02d}"
                )
                destination.mkdir(parents=True)
                np.savez_compressed(
                    destination / "designs.npz",
                    candidate_binary_designs=candidates,
                    r25_binary_design=binary,
                    r50_binary_design=binary,
                )
                (destination / "result.json").write_text(
                    __import__("json").dumps(
                        {"selected_head": 0, "task_id": task.task_id}
                    ),
                    encoding="utf-8",
                )
            for method in ("adam", "mma"):
                destination = (
                    tmp_path
                    / "baselines"
                    / method
                    / split
                    / f"task_{index:02d}"
                    / "start_0"
                )
                destination.mkdir(parents=True)
                np.savez_compressed(destination / "designs.npz", binary_600=binary)
                snapshots = {
                    str(evaluation): reference * multiplier
                    for evaluation, multiplier in (
                        (25, 1.6),
                        (50, 1.4),
                        (100, 1.25),
                        (200, 1.12),
                        (600, 1.0),
                    )
                }
                (destination / "result.json").write_text(
                    __import__("json").dumps(
                        {
                            "status": "PASS",
                            "method": method.upper(),
                            "split": split,
                            "task_index": index,
                            "start_index": 0,
                            "snapshot_tmax_scipy64": snapshots,
                        }
                    ),
                    encoding="utf-8",
                )
            if index < 8:
                for start_index in (1, 2, 3):
                    destination = (
                        tmp_path
                        / "baselines"
                        / "adam"
                        / split
                        / f"task_{index:02d}"
                        / f"start_{start_index}"
                    )
                    destination.mkdir(parents=True)
                    (destination / "result.json").write_text(
                        __import__("json").dumps(
                            {
                                "status": "PASS",
                                "method": "ADAM",
                                "split": split,
                                "task_index": index,
                                "start_index": start_index,
                                "snapshot_tmax_scipy64": {"600": reference},
                            }
                        ),
                        encoding="utf-8",
                    )
    epyc = build_epyc9754_scale_benchmark(resolution=64)
    epyc_destination = tmp_path / "epyc9754_scale_synthetic"
    epyc_destination.mkdir()
    np.savez_compressed(
        epyc_destination / "designs.npz",
        sources_64=epyc.sources,
        candidate_binary_designs=candidates,
        sens_best4_binary=binary,
        sens_r25_binary=binary,
        sens_r50_binary=binary,
        adam600_binary=binary,
        mma600_binary=binary,
    )
    verified_epyc = {
        family: {
            "worst_peak": 20.0 + offset,
            "scenario_peaks": [18.0, 19.0, 20.0 + offset],
        }
        for offset, family in enumerate(
            (
                "SENS_UNET_BEST4",
                "SENS_UNET_BEST4_R25",
                "SENS_UNET_BEST4_R50",
                "ADAM_600",
                "MMA_600",
            )
        )
    }
    epyc_result = {
        "label": "EPYC_9754_SCALE_SYNTHETIC",
        "exact_proprietary_cpu_model": False,
        "affects_primary_id_ood_verdict": False,
        "strong_single_family": "ADAM_600",
        "verified_scipy256": verified_epyc,
    }
    (epyc_destination / "result.json").write_text(
        __import__("json").dumps(epyc_result), encoding="utf-8"
    )
    verification = tmp_path / "verification"
    verification.mkdir()
    primary_frame = pd.DataFrame(primary_rows)
    primary_frame[primary_frame["split"] == "test_id"].to_csv(
        verification / "test_id_primary_rows.csv", index=False
    )
    primary_frame[primary_frame["split"] == "test_ood"].to_csv(
        verification / "test_ood_primary_rows.csv", index=False
    )
    pd.DataFrame(verification_rows).to_csv(
        verification / "all_scipy256_rows.csv", index=False
    )
    multistart_frame = pd.DataFrame(
        [
            {
                "split": split,
                "task_index": index,
                "sens_r25_tmax_scipy256": 0.16,
                "adam_multistart_tmax_scipy256": 0.165,
                "relative_gap_to_adam_multistart": -0.03,
            }
            for split in ("test_id", "test_ood")
            for index in range(8)
        ]
    )
    multistart_frame.to_csv(verification / "adam_multistart_rows.csv", index=False)
    verdicts = {
        "test_id": {"verdict": {"status": "MT3_BEATS_SINGLE_START_ID"}},
        "test_ood": {"verdict": {"status": "MT3_BEATS_SINGLE_START_OOD"}},
    }
    (verification / "final_verdicts.json").write_text(
        __import__("json").dumps(verdicts), encoding="utf-8"
    )
    (tmp_path / "opened_task_manifest.json").write_text(
        __import__("json").dumps(
            {
                "test_id": [
                    {"index": index, "task_id": task.task_id}
                    for index, task in enumerate(splits.test_id)
                ],
                "test_ood": [
                    {"index": index, "task_id": task.task_id}
                    for index, task in enumerate(splits.test_ood)
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = MT3FinalEvidence(
        primary_rows=primary_frame,
        verification_rows=pd.DataFrame(verification_rows),
        multistart_rows=multistart_frame,
        budget_rows=collect_baseline_budget_rows(tmp_path),
        verdicts=verdicts,
        epyc_result=epyc_result,
    )

    def fake_temperature_provider(design, source_maps):
        del design
        return tuple(np.asarray(source, dtype=np.float64) for source in source_maps)

    figures, diagnostics = build_spatial_figures(
        result_root=tmp_path,
        evidence=evidence,
        temperature_provider=fake_temperature_provider,
    )

    assert set(figures) == {spec.stem for spec in FINAL_FIGURE_SPECS[9:]}
    assert len(diagnostics) == 48 * 4
    assert set(diagnostics["method"]) == {
        "SENS + R25",
        "FIELD + R25",
        "Adam-600",
        "MMA-600",
    }
    assert all(figure.axes for figure in figures.values())
    for figure in figures.values():
        plt.close(figure)

    loaded = load_mt3_final_evidence(tmp_path)
    assert len(loaded.primary_rows) == 48
    assert len(loaded.verification_rows) == 288

    training_root = tmp_path / "training"
    for variant in ("field_unet", "sens_unet"):
        destination = training_root / variant
        destination.mkdir(parents=True)
        (destination / "checkpoint_004000.pt").write_bytes(variant.encode("ascii"))
    package = build_mt3_final_package(
        MT3FinalReportPaths(
            result_root=tmp_path,
            training_root=training_root,
            output_root=tmp_path / "package",
        ),
        dpi=72,
        temperature_provider=fake_temperature_provider,
    )

    assert len(list((package / "figures").glob("*.png"))) == 18
    assert len(list((package / "figures").glob("*.svg"))) == 18
    assert len(list((package / "figures").glob("*.pdf"))) == 18
    assert (package / "MT3_FINAL_REPORT.md").is_file()
    assert (package / "README_RU.md").is_file()
    assert (package / "models" / "sens_unet_selected.pt").is_file()
    assert (package / "data" / "connectivity_diagnostics.csv").is_file()
    assert (package / "manifest.json").is_file()
