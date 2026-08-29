"""Проверки prospective pure-NCA protocol lock."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from waveforge.ml.nca_protocol import NCAProtocol, load_nca_protocol

CONFIG_PATH = Path("configs/pure_nca_spike.yaml")


def _raw_protocol() -> dict[str, object]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    assert isinstance(payload, dict)
    return payload


def test_locked_protocol_loads_exact_scientific_values() -> None:
    protocol = load_nca_protocol(CONFIG_PATH)

    assert protocol.architecture.mutable_channels == 16
    assert protocol.architecture.material_channel == 0
    assert protocol.architecture.rollout_steps == 64
    assert protocol.architecture.parameter_count == 11472
    assert protocol.conditioning.fixed_source_scale == 25.0
    assert protocol.objective.projection_beta == 8.0
    assert protocol.objective.smooth_max_alpha == 500.0
    assert protocol.objective.tv_weight == 1.0e-3
    assert protocol.objective.binarization_weight == 0.02
    assert protocol.objective.material_penalty == 0.0
    assert protocol.qualification.candidate_learning_rates == (
        3.0e-4,
        1.0e-3,
        3.0e-3,
    )
    assert protocol.qualification.early_window == (20, 40)
    assert protocol.qualification.late_window == (180, 200)
    assert protocol.production.seeds == (20260901, 20260902, 20260903)
    assert protocol.production.iterations == 2000
    assert protocol.verification.peak_threshold_256 == 0.1721575074379424
    assert protocol.verification.primary_resolution == 256
    assert protocol.verification.secondary_resolution == 128
    assert protocol.hashing.mode == "canonical_lf_text_raw_binary"


@pytest.mark.parametrize(
    ("section", "field", "invalid_value"),
    [
        ("architecture", "rollout_steps", 63),
        ("architecture", "parameter_count", 11473),
        ("objective", "tv_weight", -1.0e-3),
        ("objective", "binarization_weight", -0.02),
        ("qualification", "iterations", 199),
        ("qualification", "early_window", [20, 41]),
        ("qualification", "late_window", [179, 200]),
        ("production", "iterations", 1999),
        ("production", "seeds", [20260901, 20260902, 20260904]),
        ("verification", "peak_threshold_256", 0.18),
    ],
)
def test_protocol_rejects_changes_to_locked_values(
    section: str,
    field: str,
    invalid_value: object,
) -> None:
    payload = deepcopy(_raw_protocol())
    section_payload = payload[section]
    assert isinstance(section_payload, dict)
    section_payload[field] = invalid_value

    with pytest.raises(ValidationError):
        NCAProtocol.model_validate(payload)


def test_protocol_rejects_unknown_fields() -> None:
    payload = _raw_protocol()
    payload["unregistered_setting"] = True

    with pytest.raises(ValidationError):
        NCAProtocol.model_validate(payload)
