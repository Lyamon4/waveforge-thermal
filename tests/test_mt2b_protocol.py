from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from waveforge.ml.mt2b_protocol import (
    MT2BProtocol,
    load_mt2b_protocol,
    protocol_bundle_hash,
    training_settings_at,
)

CONFIG_PATH = Path("configs/nca_mt2b.yaml")
SPEC_PATH = Path(
    "docs/superpowers/specs/2026-08-31-nca-mt2b-physics-conditioning-design.md"
)


def _payload() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    assert isinstance(payload, dict)
    return payload


def test_mt2b_protocol_loads_exact_scientific_lock() -> None:
    protocol = load_mt2b_protocol(CONFIG_PATH)

    assert protocol.scope == "nca_mt2b_physics_informed_multitask_ablation"
    assert protocol.architecture.condition_channels == 4
    assert protocol.architecture.parameter_count == 12624
    assert protocol.conditioning.fixed_temperature_scale == 0.900613256638055
    assert protocol.task_distribution.batch_size == 4
    assert protocol.task_distribution.strata.horizontal_span_threshold == 0.46
    assert protocol.task_distribution.strata.vertical_span_threshold == 0.21
    assert protocol.validation.candidate_binary_evaluator == "independent_scipy_64"
    assert protocol.validation.reference_binary_evaluator == "independent_scipy_64"
    assert protocol.validation.selected_design_secondary_evaluator == (
        "independent_scipy_256"
    )
    assert protocol.bootstrap.resamples == 10000
    assert protocol.bootstrap.seed == 2026092203
    assert protocol.bootstrap.percentile_bounds == (0.025, 0.975)
    assert protocol.split_access.test_id == "sealed"
    assert protocol.split_access.test_ood == "sealed"
    assert protocol.runtime.long_training_authorized is False


@pytest.mark.parametrize(
    ("update", "beta", "alpha", "binary_weight", "tv_weight", "lr"),
    [
        (0, 2.0, 100.0, 0.0, 0.001, 0.001),
        (399, 2.0, 100.0, 0.0, 0.001, 0.001),
        (400, 4.0, 250.0, 0.01, 0.001, 0.0003),
        (799, 4.0, 250.0, 0.01, 0.001, 0.0003),
        (800, 8.0, 500.0, 0.02, 0.001, 0.0001),
        (1999, 8.0, 500.0, 0.02, 0.001, 0.0001),
    ],
)
def test_mt2b_training_schedule_boundaries(
    update: int,
    beta: float,
    alpha: float,
    binary_weight: float,
    tv_weight: float,
    lr: float,
) -> None:
    stage = training_settings_at(update)
    assert stage.projection_beta == beta
    assert stage.smooth_max_alpha == alpha
    assert stage.binarization_weight == binary_weight
    assert stage.tv_weight == tv_weight
    assert stage.learning_rate == lr


@pytest.mark.parametrize("update", [-1, 2000])
def test_mt2b_training_schedule_rejects_out_of_range_updates(update: int) -> None:
    with pytest.raises(ValueError, match="outside MT2B training range"):
        training_settings_at(update)


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("architecture", "rollout_steps", 80),
        ("conditioning", "fixed_temperature_scale", 1.0),
        ("task_distribution", "batch_size", 8),
        ("bootstrap", "resamples", 9999),
        ("bootstrap", "seed", 1),
        ("runtime", "long_training_authorized", True),
        ("split_access", "test_id", "development_only"),
    ],
)
def test_mt2b_protocol_rejects_changes_to_locked_values(
    section: str, field: str, bad_value: object
) -> None:
    payload = deepcopy(_payload())
    payload[section][field] = bad_value

    with pytest.raises(ValidationError):
        MT2BProtocol.model_validate(payload)


def test_mt2b_protocol_rejects_unknown_fields() -> None:
    payload = _payload()
    payload["result_dependent_override"] = True

    with pytest.raises(ValidationError):
        MT2BProtocol.model_validate(payload)


def test_protocol_bundle_hash_uses_canonical_lf_component_hashes(
    tmp_path: Path,
) -> None:
    config_lf = tmp_path / "config_lf.yaml"
    config_crlf = tmp_path / "config_crlf.yaml"
    spec_lf = tmp_path / "spec_lf.md"
    spec_crlf = tmp_path / "spec_crlf.md"
    config_lf.write_bytes(b"a: 1\nb: 2\n")
    config_crlf.write_bytes(b"a: 1\r\nb: 2\r\n")
    spec_lf.write_bytes(b"# Spec\nlocked\n")
    spec_crlf.write_bytes(b"# Spec\r\nlocked\r\n")

    assert protocol_bundle_hash(config_lf, spec_lf) == protocol_bundle_hash(
        config_crlf, spec_crlf
    )
    assert len(protocol_bundle_hash(CONFIG_PATH, SPEC_PATH)) == 64


def test_protocol_lock_artifact_matches_current_canonical_bundle() -> None:
    lock = json.loads(
        Path("artifacts/nca_mt2b_protocol/protocol_lock.json").read_text(
            encoding="utf-8"
        )
    )

    assert lock["protocol_bundle_sha256"] == protocol_bundle_hash(
        CONFIG_PATH, SPEC_PATH
    )
    assert lock["long_training_authorized"] is False
    assert lock["test_id_accessed"] is False
    assert lock["test_ood_accessed"] is False
