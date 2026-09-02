from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from waveforge.experiments.run_mt2b_evaluation import validation_tasks
from waveforge.ml.mt3_evaluation import MT3CheckpointSummary, MT3DevelopmentVerdict
from waveforge.reporting.mt3 import (
    MT3ReportPaths,
    build_mt3_development_package,
    build_mt3_report_markdown,
    illustrative_layout_indices,
    save_figure_triplet,
)


def _summary(variant: str, median_gap: float) -> MT3CheckpointSummary:
    return MT3CheckpointSummary(
        completed_updates=2500,
        variant=variant,
        split_name="validation",
        task_count=32,
        invalid_count=0,
        exact_budget_count=32,
        median_r25_relative_gap=median_gap,
        p90_r25_relative_gap=0.06,
        worst_r25_relative_gap=0.14,
        r25_win_count=11,
        median_best4_relative_gap=median_gap - 0.01,
    )


def test_report_discloses_control_refinement_cost_and_development_scope() -> None:
    verdict = MT3DevelopmentVerdict(
        status="MT3_DEVELOPMENT_GO",
        test_authorized=True,
        median_gap=0.01,
        p90_gap=0.06,
        worst_gap=0.14,
        win_count=11,
        valid_count=32,
        exact_budget_count=32,
        exact_reason="all locked gates passed",
    )

    report = build_mt3_report_markdown(
        sens=_summary("SENS_UNET", 0.01),
        field=_summary("FIELD_UNET", 0.04),
        verdict=verdict,
        figure_names=("01_training_curves", "02_gap_distribution"),
    )

    assert "SENS_UNET_BEST4_R25" in report
    assert "FIELD_UNET" in report
    assert "four forward-only physics scores" in report
    assert "one candidate" in report
    assert "development validation only" in report.lower()
    assert "ID/OOD test layouts remain sealed" in report


def test_illustrative_indices_are_predeclared_best_median_and_worst_ranks() -> None:
    gaps = np.asarray([0.03, -0.02, 0.01, 0.08, 0.00], dtype=np.float64)

    indices = illustrative_layout_indices(gaps)

    assert indices == {"best": 1, "median": 2, "worst": 3}


def test_figure_triplet_writes_nonempty_png_svg_and_pdf(tmp_path: Path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [1, 0])

    paths = save_figure_triplet(figure, tmp_path, "figure")

    assert {path.suffix for path in paths} == {".png", ".svg", ".pdf"}
    assert all(path.stat().st_size > 0 for path in paths)


def test_package_builds_report_tables_and_figure_triplets_from_frozen_rows(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    references = tmp_path / "references"
    output = tmp_path / "package"
    summaries = [_summary("FIELD_UNET", 0.04), _summary("SENS_UNET", 0.01)]
    (evaluation / "checkpoint_summaries.json").parent.mkdir(parents=True)
    (evaluation / "checkpoint_summaries.json").write_text(
        __import__("json").dumps([summary.__dict__ for summary in summaries]),
        encoding="utf-8",
    )
    tasks = validation_tasks()
    binary = np.zeros((64, 64), dtype=np.float64)
    binary.reshape(-1)[:1024] = 1.0
    for variant, gap in (("field_unet", 0.04), ("sens_unet", 0.01)):
        directory = evaluation / variant / "checkpoint_002500"
        directory.mkdir(parents=True)
        rows = []
        for index, task in enumerate(tasks):
            reference = 0.2
            rows.append(
                {
                    "task_index": index,
                    "task_id": task.task_id,
                    "reference_tmax_scipy64": reference,
                    "best4_tmax_scipy64": reference * (1.0 + gap - 0.01),
                    "r25_tmax_scipy64": reference * (1.0 + gap),
                    "r25_relative_gap": gap,
                    "selected_head": index % 4,
                    "binary_cell_count": 1024,
                    "refinement_updates": 25,
                }
            )
            task_dir = directory / "tasks"
            task_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                task_dir / f"task_{index:02d}.npz",
                candidate_binary_designs=np.stack([binary] * 4),
                refined_continuous_design=binary,
                refined_binary_design=binary,
            )
            (task_dir / f"task_{index:02d}_trace.json").write_text(
                __import__("json").dumps(
                    {
                        "records": [
                            {
                                "iteration": step,
                                "total_objective": 0.3 - step * 0.002,
                                "exact_peak": 0.25 - step * 0.001,
                            }
                            for step in range(25)
                        ]
                    }
                ),
                encoding="utf-8",
            )
        pd.DataFrame(rows).to_csv(directory / "validation_metrics.csv", index=False)
        train_dir = training / variant
        train_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "update": [0, 1, 2],
                "mean_loss": [0.6, 0.4, 0.2],
                "best_candidate_thermal_smooth": [0.5, 0.3, 0.18],
                "learning_rate": [1e-4, 1e-4, 3e-5],
            }
        ).to_csv(train_dir / "training_metrics.csv", index=False)
        (train_dir / "checkpoint_002500.pt").write_bytes(variant.encode("ascii"))
    for index in range(32):
        directory = references / "references" / f"task_{index:02d}"
        directory.mkdir(parents=True)
        np.save(directory / "binary_design_64.npy", binary, allow_pickle=False)

    package = build_mt3_development_package(
        MT3ReportPaths(
            training_root=training,
            evaluation_root=evaluation,
            reference_root=references,
            output_root=output,
        ),
        include_temperature_maps=False,
    )

    assert package == output
    assert (output / "MT3_REPORT.md").is_file()
    assert (output / "README_RU.md").is_file()
    assert (output / "performance_table.csv").is_file()
    assert (output / "manifest.json").is_file()
    assert len(list((output / "figures").glob("*.png"))) >= 7
    assert len(list((output / "figures").glob("*.svg"))) >= 7
    assert len(list((output / "figures").glob("*.pdf"))) >= 7
