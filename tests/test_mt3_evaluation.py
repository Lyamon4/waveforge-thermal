from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from waveforge.ml.mt3_evaluation import (
    MT3CheckpointSummary,
    build_test_authorization_bundle,
    classify_mt3_development,
    load_sealed_registry,
    select_mt3_checkpoint,
    solver_consistent_gap,
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
