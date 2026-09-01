from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from waveforge.ml.mt3_protocol import (
    MT3Protocol,
    assert_paid_runtime_authorized,
    load_mt3_protocol,
    training_settings_at,
)

CONFIG_PATH = Path("configs/mt3_sensitivity_warmstart.yaml")


def test_mt3_protocol_locks_single_candidate_refinement() -> None:
    protocol = load_mt3_protocol(CONFIG_PATH)

    assert protocol.scope == "mt3_sensitivity_conditioned_learned_warmstart"
    assert protocol.architecture.condition_channels == 5
    assert protocol.architecture.candidate_count == 4
    assert protocol.refinement.candidate_scores == 4
    assert protocol.refinement.selected_candidates == 1
    assert protocol.refinement.primary_steps == 25
    assert protocol.refinement.secondary_steps == 50
    assert protocol.training.updates == 4000
    assert protocol.training.batch_size == 4
    assert protocol.split_access.test_id == "sealed"
    assert protocol.split_access.test_ood == "sealed"


@pytest.mark.parametrize(
    ("update", "beta", "alpha", "binary_weight", "lr_multiplier"),
    [
        (0, 2.0, 100.0, 0.0, 1.0),
        (799, 2.0, 100.0, 0.0, 1.0),
        (800, 4.0, 250.0, 0.01, 0.3),
        (1599, 4.0, 250.0, 0.01, 0.3),
        (1600, 8.0, 500.0, 0.02, 0.1),
        (3999, 8.0, 500.0, 0.02, 0.1),
    ],
)
def test_mt3_training_schedule_boundaries(
    update: int,
    beta: float,
    alpha: float,
    binary_weight: float,
    lr_multiplier: float,
) -> None:
    stage = training_settings_at(update)

    assert stage.projection_beta == beta
    assert stage.smooth_max_alpha == alpha
    assert stage.binarization_weight == binary_weight
    assert stage.tv_weight == 0.001
    assert stage.learning_rate_multiplier == lr_multiplier


@pytest.mark.parametrize("update", [-1, 4000])
def test_mt3_training_schedule_rejects_out_of_range_update(update: int) -> None:
    with pytest.raises(ValueError, match="outside MT3 training range"):
        training_settings_at(update)


def test_paid_runtime_guard_preserves_credit_buffer() -> None:
    assert_paid_runtime_authorized(2.5, 0.67, credit_usd=2.0)

    with pytest.raises(RuntimeError, match="paid runtime is not authorized"):
        assert_paid_runtime_authorized(2.6, 0.67, credit_usd=2.0)
    with pytest.raises(RuntimeError, match="paid runtime is not authorized"):
        assert_paid_runtime_authorized(2.5, 0.70, credit_usd=1.8)


@pytest.mark.parametrize(
    ("projected_hours", "hourly_usd", "credit_usd"),
    [(-1.0, 0.67, 2.0), (1.0, 0.0, 2.0), (1.0, 0.67, 0.0)],
)
def test_paid_runtime_guard_rejects_nonpositive_inputs(
    projected_hours: float,
    hourly_usd: float,
    credit_usd: float,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        assert_paid_runtime_authorized(
            projected_hours,
            hourly_usd,
            credit_usd=credit_usd,
        )


def test_mt3_protocol_forbids_more_than_one_refined_candidate() -> None:
    protocol = load_mt3_protocol(CONFIG_PATH)
    payload = protocol.model_dump(mode="python")
    payload["refinement"]["selected_candidates"] = 4

    with pytest.raises(ValidationError):
        MT3Protocol.model_validate(payload)
