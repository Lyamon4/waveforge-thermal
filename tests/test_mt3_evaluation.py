from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from waveforge.experiments.run_mt2b_evaluation import validation_tasks
from waveforge.experiments.run_mt3_evaluation import (
    MT3EvaluatedTask,
    MT3ProductionEvaluationError,
    production_checkpoint_paths,
    run_checkpoint_evaluation,
)
from waveforge.ml.mt3_evaluation import (
    MT3CheckpointSummary,
    build_test_authorization_bundle,
    classify_mt3_development,
    load_sealed_registry,
    select_mt3_checkpoint,
    solver_consistent_gap,
    summarize_mt3_checkpoint_rows,
)


def _summary(
    completed: int,
    *,
    r25_median: float,
    r25_p90: float,
    oneshot_median: float,
    invalid: int = 0,
) -> MT3CheckpointSummary:
    return MT3CheckpointSummary(
        completed_updates=completed,
        variant="SENS_UNET",
        split_name="validation",
        task_count=32,
        invalid_count=invalid,
        exact_budget_count=32,
        median_r25_relative_gap=r25_median,
        p90_r25_relative_gap=r25_p90,
        worst_r25_relative_gap=0.15,
        r25_win_count=12,
        median_best4_relative_gap=oneshot_median,
    )


def test_checkpoint_selection_uses_r25_gap_before_one_shot_gap() -> None:
    summaries = [
        _summary(2000, r25_median=0.018, r25_p90=0.06, oneshot_median=0.01),
        _summary(2500, r25_median=0.015, r25_p90=0.06, oneshot_median=0.08),
        _summary(3000, r25_median=0.017, r25_p90=0.05, oneshot_median=0.00),
    ]

    selected = select_mt3_checkpoint(summaries)

    assert selected.completed_updates == 2500


def test_checkpoint_selection_rejects_ineligible_or_nonvalidation_rows() -> None:
    with pytest.raises(ValueError, match="eligible"):
        select_mt3_checkpoint(
            [
                _summary(
                    500, r25_median=0.01, r25_p90=0.02, oneshot_median=0.03, invalid=1
                )
            ]
        )
    wrong = _summary(500, r25_median=0.01, r25_p90=0.02, oneshot_median=0.03)
    wrong = MT3CheckpointSummary(**{**wrong.__dict__, "split_name": "test_id"})
    with pytest.raises(ValueError, match="validation"):
        select_mt3_checkpoint([wrong])


def test_failed_development_gate_does_not_authorize_test_access() -> None:
    verdict = classify_mt3_development(
        median_gap=0.021,
        p90_gap=0.06,
        worst_gap=0.10,
        wins=12,
        valid_count=32,
        exact_budget_count=32,
    )

    assert verdict.status == "MT3_DEVELOPMENT_NO_GO"
    assert verdict.test_authorized is False


def test_development_gate_requires_every_exact_locked_threshold() -> None:
    verdict = classify_mt3_development(
        median_gap=0.02,
        p90_gap=0.07,
        worst_gap=0.20,
        wins=8,
        valid_count=32,
        exact_budget_count=32,
    )
    assert verdict.status == "MT3_DEVELOPMENT_GO"
    assert verdict.test_authorized is True


def test_solver_consistent_gap_rejects_different_evaluator_ids() -> None:
    gap = solver_consistent_gap(
        candidate_tmax=np.float64(0.18),
        reference_tmax=np.float64(0.20),
        candidate_solver_id="independent_scipy_64",
        reference_solver_id="independent_scipy_64",
    )
    assert gap == pytest.approx(-0.10)
    with pytest.raises(ValueError, match="same solver"):
        solver_consistent_gap(
            candidate_tmax=np.float64(0.18),
            reference_tmax=np.float64(0.20),
            candidate_solver_id="torch64",
            reference_solver_id="independent_scipy_64",
        )


def test_checkpoint_rows_are_summarized_from_all_32_paired_layouts() -> None:
    gaps = [index / 1000.0 - 0.010 for index in range(32)]
    rows = [
        {
            "task_index": index,
            "candidate_solver": "independent_scipy_64",
            "reference_solver": "independent_scipy_64",
            "reference_tmax_scipy64": 0.2,
            "best4_tmax_scipy64": 0.2 * (1.0 + gap - 0.005),
            "r25_tmax_scipy64": 0.2 * (1.0 + gap),
            "binary_cell_count": 1024,
            "refinement_updates": 25,
        }
        for index, gap in enumerate(gaps)
    ]

    summary = summarize_mt3_checkpoint_rows(
        rows,
        completed_updates=2500,
        variant="SENS_UNET",
    )

    assert summary.completed_updates == 2500
    assert summary.task_count == 32
    assert summary.invalid_count == 0
    assert summary.exact_budget_count == 32
    assert summary.median_r25_relative_gap == pytest.approx(0.0055)
    assert summary.p90_r25_relative_gap == pytest.approx(0.0179)
    assert summary.worst_r25_relative_gap == pytest.approx(0.021)
    assert summary.r25_win_count == 10
    assert summary.median_best4_relative_gap == pytest.approx(0.0005)


def test_checkpoint_summary_fails_closed_on_incomplete_or_mixed_solver_rows() -> None:
    valid = {
        "task_index": 0,
        "candidate_solver": "independent_scipy_64",
        "reference_solver": "independent_scipy_64",
        "reference_tmax_scipy64": 0.2,
        "best4_tmax_scipy64": 0.19,
        "r25_tmax_scipy64": 0.18,
        "binary_cell_count": 1024,
        "refinement_updates": 25,
    }
    with pytest.raises(ValueError, match="exactly 32"):
        summarize_mt3_checkpoint_rows(
            [valid] * 31,
            completed_updates=500,
            variant="FIELD_UNET",
        )
    mixed = [dict(valid, task_index=index) for index in range(32)]
    mixed[7]["candidate_solver"] = "torch64"
    with pytest.raises(ValueError, match="same solver"):
        summarize_mt3_checkpoint_rows(
            mixed,
            completed_updates=500,
            variant="SENS_UNET",
        )


def test_sealed_registry_is_not_read_without_authorization(tmp_path: Path) -> None:
    verdict = classify_mt3_development(
        median_gap=0.03,
        p90_gap=0.08,
        worst_gap=0.21,
        wins=7,
        valid_count=32,
        exact_budget_count=32,
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    bundle = build_test_authorization_bundle(
        verdict=verdict,
        implementation_commit="a" * 40,
        artifacts={"evidence": evidence},
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(PermissionError, match="sealed"):
        load_sealed_registry(
            tmp_path / "does-not-exist.json",
            authorization_path=authorization,
            expected_bundle_sha256=bundle["bundle_sha256"],
        )


def test_production_checkpoint_schedule_requires_all_eight_frozen_files(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "sens_unet"
    directory.mkdir()
    for completed in range(500, 4001, 500):
        (directory / f"checkpoint_{completed:06d}.pt").write_bytes(b"checkpoint")
    (directory / "mt3_run_result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "completed_updates": 4000,
                "config": {"variant": "SENS_UNET", "mode": "production"},
                "test_id_accessed": False,
                "test_ood_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    paths = production_checkpoint_paths(tmp_path, "SENS_UNET")

    assert [path.name for path in paths] == [
        f"checkpoint_{completed:06d}.pt" for completed in range(500, 4001, 500)
    ]
    paths[3].unlink()
    with pytest.raises(MT3ProductionEvaluationError, match="eight"):
        production_checkpoint_paths(tmp_path, "SENS_UNET")


def test_checkpoint_evaluation_persists_and_resumes_all_32_layouts(
    tmp_path: Path,
) -> None:
    tasks = validation_tasks()
    references = {task.task_id: 0.2 for task in tasks}
    calls: list[int] = []

    def evaluator(task_index, task, reference_tmax, checkpoint, variant, device):
        calls.append(task_index)
        binary = np.zeros((64, 64), dtype=np.float64)
        binary.reshape(-1)[:1024] = 1.0
        candidates = np.stack([binary] * 4)
        return MT3EvaluatedTask(
            row={
                "task_index": task_index,
                "task_id": task.task_id,
                "candidate_solver": "independent_scipy_64",
                "reference_solver": "independent_scipy_64",
                "reference_tmax_scipy64": reference_tmax,
                "best4_tmax_scipy64": 0.19,
                "r25_tmax_scipy64": 0.18,
                "selected_head": task_index % 4,
                "binary_cell_count": 1024,
                "refinement_updates": 25,
                "test_id_accessed": False,
                "test_ood_accessed": False,
            },
            candidate_binary_designs=candidates,
            refined_continuous_design=binary,
            refined_binary_design=binary,
            refinement_trace=tuple(),
        )

    checkpoint = tmp_path / "checkpoint_001500.pt"
    checkpoint.write_bytes(b"frozen")
    output = tmp_path / "evaluation"
    summary = run_checkpoint_evaluation(
        checkpoint,
        variant="SENS_UNET",
        tasks=tasks,
        reference_peaks=references,
        output_dir=output,
        device="cpu",
        task_evaluator=evaluator,
    )
    resumed = run_checkpoint_evaluation(
        checkpoint,
        variant="SENS_UNET",
        tasks=tasks,
        reference_peaks=references,
        output_dir=output,
        device="cpu",
        task_evaluator=evaluator,
    )

    assert calls == list(range(32))
    assert summary == resumed
    assert summary.median_r25_relative_gap == pytest.approx(-0.10)
    assert summary.r25_win_count == 32
    assert (output / "validation_metrics.csv").is_file()
    assert (output / "checkpoint_summary.json").is_file()
    assert len(list((output / "tasks").glob("task_*.npz"))) == 32
