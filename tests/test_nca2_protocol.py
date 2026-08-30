from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from waveforge.ml.nca2_protocol import NCA2Protocol, load_nca2_protocol

CONFIG_PATH = Path("configs/nca2_stabilization.yaml")


def _payload() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    assert isinstance(payload, dict)
    return payload


def test_nca2_protocol_loads_locked_campaign() -> None:
    protocol = load_nca2_protocol(CONFIG_PATH)

    assert protocol.scope == "pure_nca_fixed_abc_stabilization"
    assert protocol.architecture.parameter_count == 11472
    assert protocol.architecture.update_scale == 0.1
    assert protocol.development.seeds == (20260901, 20260902, 20260903)
    assert protocol.development.iterations == 700
    assert protocol.production.seeds == (20260911, 20260912, 20260913)
    assert protocol.production.iterations == 1500
    assert protocol.verification.tree_peak == 0.1650978093408512
    assert protocol.verification.pass_peak == 0.1617958531540342
    assert protocol.verification.noncollapse_peak == 0.1683997655276682
    assert protocol.verification.connectivity_has_primary_authority is False
    assert protocol.runtime.maximum_projected_gpu_hours == 6.6


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("architecture", "update_scale"), 0.05),
        (("development", "seeds"), [20260901, 20260902, 20260904]),
        (("production", "iterations"), 1499),
        (("verification", "pass_peak"), 0.162),
    ],
)
def test_nca2_protocol_rejects_changes_to_locked_values(
    path: tuple[str, str], bad_value: object
) -> None:
    payload = deepcopy(_payload())
    payload[path[0]][path[1]] = bad_value

    with pytest.raises(ValidationError):
        NCA2Protocol.model_validate(payload)


def test_nca2_protocol_rejects_unknown_fields() -> None:
    payload = _payload()
    payload["unregistered_knob"] = True

    with pytest.raises(ValidationError):
        NCA2Protocol.model_validate(payload)
