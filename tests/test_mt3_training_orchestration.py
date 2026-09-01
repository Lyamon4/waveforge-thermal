from __future__ import annotations

import json
from pathlib import Path

import pytest

from waveforge.experiments.run_mt3_training import (
    MT3ExecutionError,
    build_variant_identity,
    validate_variant_identity,
)


def test_mt3_identity_keeps_test_splits_sealed() -> None:
    identity = build_variant_identity(
        "FIELD_UNET",
        source_sha="a" * 40,
        selected_learning_rate=1.0e-4,
    )

    assert identity["variant"] == "FIELD_UNET"
    assert identity["model_seed"] == 2026092311
    assert identity["task_stream_seed"] == 2026092312
    assert identity["updates"] == 4000
    assert identity["batch_size"] == 4
    assert identity["test_id_accessed"] is False
    assert identity["test_ood_accessed"] is False


def test_mt3_identity_rejects_cross_variant_resume(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    path.write_text(
        json.dumps(
            build_variant_identity(
                "FIELD_UNET",
                source_sha="b" * 40,
                selected_learning_rate=3.0e-4,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(MT3ExecutionError, match="identity"):
        validate_variant_identity(
            path,
            expected=build_variant_identity(
                "SENS_UNET",
                source_sha="b" * 40,
                selected_learning_rate=3.0e-4,
            ),
        )
