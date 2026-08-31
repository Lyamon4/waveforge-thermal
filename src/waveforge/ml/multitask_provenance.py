"""Immutable production registry and fail-closed backup evidence."""

from __future__ import annotations

import re
from pathlib import Path

from waveforge.ml.multitask_protocol import PRODUCTION_SEEDS
from waveforge.reproducibility import artifact_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceError(RuntimeError):
    """Raised when production identity or backup evidence is incomplete."""


def create_production_registry(
    *,
    updates_per_seed: int,
    microbatch_size: int,
    source_sha256: str,
    spec_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    """Create the exact prospective three-seed production identity."""
    registry: dict[str, object] = {
        "schema_version": 1,
        "production_seeds": list(PRODUCTION_SEEDS),
        "updates_per_seed": updates_per_seed,
        "microbatch_size": microbatch_size,
        "source_sha256": source_sha256,
        "spec_sha256": spec_sha256,
        "config_sha256": config_sha256,
    }
    validate_production_registry(registry)
    return registry


def validate_production_registry(registry: dict[str, object]) -> None:
    """Reject seed substitution, insufficient training, or changed hashes."""
    if registry.get("schema_version") != 1:
        raise ProvenanceError("unsupported production registry schema")
    if registry.get("production_seeds") != list(PRODUCTION_SEEDS):
        raise ProvenanceError("production seeds were replaced or reordered")
    updates = registry.get("updates_per_seed")
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or not 5_000 <= updates <= 15_000
    ):
        raise ProvenanceError("production update count is outside [5000,15000]")
    if registry.get("microbatch_size") not in (1, 2, 4):
        raise ProvenanceError("production microbatch is not registered")
    hashes = (
        registry.get("source_sha256"),
        registry.get("spec_sha256"),
        registry.get("config_sha256"),
    )
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in hashes
    ):
        raise ProvenanceError("production hash identity is malformed")


def build_hash_manifest(
    paths: list[Path],
    *,
    root: Path,
) -> dict[str, str]:
    """Hash canonical-LF text and raw binary artifacts by relative path."""
    if not paths:
        raise ProvenanceError("hash manifest cannot be empty")
    resolved_root = root.resolve()
    manifest: dict[str, str] = {}
    for path in sorted(paths, key=lambda item: item.as_posix()):
        if not path.is_file():
            raise ProvenanceError(f"artifact is missing: {path}")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise ProvenanceError(
                f"artifact lies outside manifest root: {path}"
            ) from error
        manifest[relative] = artifact_sha256(resolved)
    return manifest


def validate_backup_readiness(required_paths: list[Path]) -> dict[str, object]:
    """Return readiness only when every named required artifact exists."""
    if not required_paths:
        raise ProvenanceError("backup requirement list cannot be empty")
    for path in required_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ProvenanceError(f"required backup artifact is missing: {path}")
    return {
        "schema_version": 1,
        "backup_ready": True,
        "required_file_count": len(required_paths),
        "required_files": [str(path) for path in required_paths],
    }
