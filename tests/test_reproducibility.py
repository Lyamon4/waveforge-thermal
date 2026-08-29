"""Проверки воспроизводимости scientific inputs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from waveforge.environment import collect_environment
from waveforge.physics.validation import compute_gate1_validation
from waveforge.reproducibility import (
    DeterminismSnapshot,
    artifact_sha256,
    configure_cuda_reproducibility,
    content_hash,
    set_deterministic_seed,
)


def test_seed_recreates_numpy_and_torch_sequences() -> None:
    """Изменение seed propagation должно ломать этот test."""
    set_deterministic_seed(20260828)
    numpy_first = np.random.random(8)
    torch_first = torch.rand(8)

    set_deterministic_seed(20260828)
    numpy_second = np.random.random(8)
    torch_second = torch.rand(8)

    np.testing.assert_array_equal(numpy_first, numpy_second)
    torch.testing.assert_close(torch_first, torch_second, rtol=0.0, atol=0.0)


def test_content_hash_changes_with_array_content_shape_and_dtype() -> None:
    """Hash не должен скрывать изменение values, shape или dtype."""
    base = np.array([[1.0, 2.0]], dtype=np.float64)
    changed = np.array([[1.0, 3.0]], dtype=np.float64)
    reshaped = np.array([[1.0], [2.0]], dtype=np.float64)
    lower_precision = base.astype(np.float32)

    base_hash = content_hash(base)
    assert base_hash != content_hash(changed)
    assert base_hash != content_hash(reshaped)
    assert base_hash != content_hash(lower_precision)


def test_text_hash_is_identical_for_lf_crlf_and_cr(tmp_path: Path) -> None:
    paths = []
    for index, newline in enumerate(("\n", "\r\n", "\r")):
        path = tmp_path / f"artifact_{index}.json"
        path.write_bytes(f'{{"a":1}}{newline}{{"b":2}}{newline}'.encode())
        paths.append(path)

    assert len({artifact_sha256(path) for path in paths}) == 1


def test_binary_hash_uses_raw_bytes(tmp_path: Path) -> None:
    first = tmp_path / "a.npy"
    second = tmp_path / "b.npy"
    first.write_bytes(b"a\r\nb")
    second.write_bytes(b"a\nb")

    assert artifact_sha256(first) != artifact_sha256(second)


def test_configure_cuda_reproducibility_sets_locked_flags() -> None:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        snapshot = configure_cuda_reproducibility(20260831, warn_only=False)
        assert snapshot.seed == 20260831
        assert snapshot.deterministic_algorithms is True
        assert snapshot.warn_only is False
        assert snapshot.cudnn_benchmark is False
        assert snapshot.cudnn_deterministic is True
        assert snapshot.mode == "strict"
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled,
            warn_only=previous_warn_only,
        )
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic


def test_environment_report_records_optional_determinism_snapshot(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "waveforge.environment.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="nvidia-smi fixture"),
    )
    snapshot = DeterminismSnapshot(
        seed=20260831,
        deterministic_algorithms=True,
        warn_only=False,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        mode="strict",
    )

    without = collect_environment("pip install fixture")
    with_snapshot = collect_environment(
        "pip install fixture",
        determinism=snapshot,
    )

    assert "determinism" not in without
    assert with_snapshot["determinism"] == {
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "mode": "strict",
        "seed": 20260831,
        "warn_only": False,
    }


def test_gate1_validation_repeats_metrics_and_input_hashes() -> None:
    """Benchmark timings исключены, physics metrics сравниваются с tolerance."""
    with open("configs/steady_validation.yaml", encoding="utf-8") as stream:
        steady_config = yaml.safe_load(stream)
    with open("configs/transient_validation.yaml", encoding="utf-8") as stream:
        transient_config = yaml.safe_load(stream)

    first = compute_gate1_validation(steady_config, transient_config)
    second = compute_gate1_validation(steady_config, transient_config)

    assert first.config_hash == second.config_hash
    assert first.input_hashes == second.input_hashes
    assert [metric.name for metric in first.metrics] == [
        metric.name for metric in second.metrics
    ]
    np.testing.assert_allclose(
        [metric.value for metric in first.metrics],
        [metric.value for metric in second.metrics],
        rtol=1e-12,
        atol=1e-14,
    )
    assert first.passed == second.passed
