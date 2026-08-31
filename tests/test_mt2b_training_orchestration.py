from __future__ import annotations

import json
from pathlib import Path

import pytest

from waveforge.experiments.run_mt2b_training import (
    MT2BExecutionError,
    build_variant_identity,
    validate_variant_identity,
)


def test_variant_identity_locks_protocol_variant_and_task_stream() -> None:
    identity = build_variant_identity(
        "RAW",
        source_sha="f" * 40,
    )

    assert identity["variant"] == "RAW"
    assert identity["protocol_bundle_sha256"] == (
        "567606c870720ca48001868efa9db1c6918e42345a1892932826c1ab0691d103"
    )
    assert identity["model_seed"] == 2026092202
    assert identity["task_stream_seed"] == 2026092201
    assert identity["updates"] == 2000
    assert identity["batch_size"] == 4
    assert identity["validation_accessed"] is False
    assert identity["test_id_accessed"] is False
    assert identity["test_ood_accessed"] is False


def test_variant_identity_rejects_cross_variant_resume(tmp_path: Path) -> None:
    identity_path = tmp_path / "run_identity.json"
    identity_path.write_text(
        json.dumps(build_variant_identity("RAW", source_sha="a" * 40)),
        encoding="utf-8",
    )

    with pytest.raises(MT2BExecutionError, match="identity"):
        validate_variant_identity(
            identity_path,
            expected=build_variant_identity("PHYSICS", source_sha="a" * 40),
        )


def test_variant_identity_rejects_different_execution_source(tmp_path: Path) -> None:
    identity_path = tmp_path / "run_identity.json"
    identity_path.write_text(
        json.dumps(build_variant_identity("RAW", source_sha="a" * 40)),
        encoding="utf-8",
    )

    with pytest.raises(MT2BExecutionError, match="identity"):
        validate_variant_identity(
            identity_path,
            expected=build_variant_identity("RAW", source_sha="b" * 40),
        )
