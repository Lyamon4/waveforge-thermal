"""Tests for immutable production registry and artifact backup evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from waveforge.ml.multitask_protocol import PRODUCTION_SEEDS
from waveforge.ml.multitask_provenance import (
    ProvenanceError,
    build_hash_manifest,
    create_production_registry,
    validate_backup_readiness,
    validate_production_registry,
)


def test_registry_rejects_replacement_or_reordered_production_seeds() -> None:
    registry = create_production_registry(
        updates_per_seed=5000,
        microbatch_size=2,
        training_hours_cap=8.0,
        source_sha256="a" * 64,
        spec_sha256="b" * 64,
        config_sha256="c" * 64,
    )
    validate_production_registry(registry)

    registry["production_seeds"] = [2026083102, 2026083104, 2026083103]
    with pytest.raises(ProvenanceError, match="seeds"):
        validate_production_registry(registry)


def test_registry_rejects_changed_or_malformed_hashes() -> None:
    registry = create_production_registry(
        updates_per_seed=5000,
        microbatch_size=2,
        training_hours_cap=8.0,
        source_sha256="a" * 64,
        spec_sha256="b" * 64,
        config_sha256="c" * 64,
    )
    registry["spec_sha256"] = "changed"
    with pytest.raises(ProvenanceError, match="hash"):
        validate_production_registry(registry)


def test_registry_requires_locked_positive_training_hours_cap() -> None:
    registry = create_production_registry(
        updates_per_seed=5000,
        microbatch_size=1,
        training_hours_cap=8.0,
        source_sha256="a" * 64,
        spec_sha256="b" * 64,
        config_sha256="c" * 64,
    )
    registry["training_hours_cap"] = 0.0
    with pytest.raises(ProvenanceError, match="hours"):
        validate_production_registry(registry)


def test_hash_manifest_uses_canonical_text_and_raw_binary(tmp_path: Path) -> None:
    text_lf = tmp_path / "first.json"
    text_crlf = tmp_path / "second.json"
    binary = tmp_path / "weights.pt"
    text_lf.write_bytes(b'{\n  "x": 1\n}\n')
    text_crlf.write_bytes(b'{\r\n  "x": 1\r\n}\r\n')
    binary.write_bytes(b"\x00\r\n\xff")

    manifest = build_hash_manifest([text_lf, text_crlf, binary], root=tmp_path)

    assert manifest["first.json"] == manifest["second.json"]
    assert len(manifest["weights.pt"]) == 64


def test_backup_readiness_requires_every_frozen_seed_and_manifest(
    tmp_path: Path,
) -> None:
    required = [tmp_path / "hash_manifest.json"] + [
        tmp_path / f"frozen_seed_{seed}.pt" for seed in PRODUCTION_SEEDS
    ]
    for path in required[:-1]:
        path.write_bytes(b"data")

    with pytest.raises(ProvenanceError, match=str(PRODUCTION_SEEDS[-1])):
        validate_backup_readiness(required)

    required[-1].write_bytes(b"data")
    result = validate_backup_readiness(required)
    assert result["backup_ready"] is True
    assert result["required_file_count"] == 4
