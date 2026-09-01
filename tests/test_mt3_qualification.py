from __future__ import annotations

import math

import pytest

from waveforge.experiments.run_mt3_qualification import (
    assert_qualification_budget,
    build_qualification_config,
    qualification_gap_summary,
)
from waveforge.experiments.run_mt3_training import MT3QualificationSpec


def test_qualification_config_uses_only_registered_sens_run() -> None:
    config = build_qualification_config(
        MT3QualificationSpec(1.0e-4, 2026092303, 2026092305)
    )

    assert config.variant == "SENS_UNET"
    assert config.model_seed == 2026092303
    assert config.task_seed == 2026092305
    assert config.base_learning_rate == 1.0e-4
    assert config.total_updates == 500
    assert config.batch_size == 4
    assert config.mode == "qualification"
    assert config.device == "cuda"


def test_qualification_gap_summary_uses_unrounded_paired_values() -> None:
    result = qualification_gap_summary(
        candidate_tmax=[0.11, 0.21],
        reference_tmax=[0.10, 0.20],
    )

    assert result.valid is True
    assert result.median_best4_r25_gap == pytest.approx(0.075)
    assert result.p90_best4_r25_gap == pytest.approx(0.095)


def test_qualification_gap_summary_fails_closed_on_invalid_value() -> None:
    result = qualification_gap_summary(
        candidate_tmax=[0.11, math.nan],
        reference_tmax=[0.10, 0.20],
    )

    assert result.valid is False
    assert math.isinf(result.median_best4_r25_gap)
    assert math.isinf(result.p90_best4_r25_gap)


def test_qualification_budget_passes_credit_as_keyword_only_argument() -> None:
    assert_qualification_budget(
        projected_hours=1.55,
        hourly_usd=0.2722222222222222,
        credit_usd=1.73,
    )
