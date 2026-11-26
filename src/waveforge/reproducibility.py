"""Воспроизводимость scientific inputs и pseudo-random generators."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from numpy.typing import NDArray


def set_deterministic_seed(seed: int) -> None:
    """Установить один seed для Python, NumPy и PyTorch generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def content_hash(array: NDArray[np.generic]) -> str:
    """Вычислить SHA-256 с учётом dtype, shape и binary content array."""
    contiguous = np.ascontiguousarray(array)
    metadata = f"{contiguous.dtype.str}|{contiguous.shape}".encode()
    return hashlib.sha256(metadata + contiguous.tobytes()).hexdigest()
