from __future__ import annotations

import pytest

from waveforge.experiments.benchmark_mt3 import (
    MT3BenchmarkMeasurements,
    assess_runtime,
)


def _benchmark(*, update_seconds: float = 0.20) -> MT3BenchmarkMeasurements:
    return MT3BenchmarkMeasurements(
        training_update_seconds=update_seconds,
        initial_probe_seconds=0.03,
        unet_forward_seconds=0.01,
        four_scipy64_scores_seconds=0.08,
        r25_chain_seconds=2.5,
        r50_chain_seconds=5.0,
        mma_evaluation_seconds=0.10,
        peak_vram_bytes=3_000_000_000,
    )


def test_runtime_authorization_includes_full_matched_campaign() -> None:
    payload = assess_runtime(_benchmark(), hourly_usd=0.67, credit_usd=2.0)

    assert set(payload["components"]) >= {
        "qualification",
        "field",
        "sens",
        "validation",
        "mma",
    }
    expected = (
        payload["projected_cost_usd"] <= 1.70 and payload["projected_hours"] <= 2.5
    )
    assert payload["authorized"] is expected
    assert payload["safety_buffer_usd"] == pytest.approx(0.10)


def test_runtime_guard_rejects_campaign_that_does_not_fit_credit() -> None:
    payload = assess_runtime(
        _benchmark(update_seconds=2.0),
        hourly_usd=0.67,
        credit_usd=2.0,
    )

    assert payload["authorized"] is False
    assert "PROJECTED_RUNTIME_EXCEEDS_LOCK" in payload["reason_codes"]


@pytest.mark.parametrize("hourly_usd", [0.0, -1.0, float("nan")])
def test_runtime_assessment_rejects_invalid_price(hourly_usd: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        assess_runtime(_benchmark(), hourly_usd=hourly_usd, credit_usd=2.0)
