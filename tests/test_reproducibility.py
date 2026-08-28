"""Проверки воспроизводимости scientific inputs."""

import numpy as np
import torch

from waveforge.reproducibility import content_hash, set_deterministic_seed


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
