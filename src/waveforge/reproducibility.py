"""Воспроизводимость scientific inputs и pseudo-random generators."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

_CANONICAL_TEXT_EXTENSIONS = frozenset({".md", ".json", ".csv", ".yaml", ".yml"})


@dataclass(frozen=True)
class DeterminismSnapshot:
    """Exact PyTorch determinism state retained in experiment provenance."""

    seed: int
    deterministic_algorithms: bool
    warn_only: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    mode: Literal["strict", "topology_verdict"]


def set_deterministic_seed(seed: int) -> None:
    """Установить один seed для Python, NumPy и PyTorch generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_reproducibility(
    seed: int,
    *,
    warn_only: bool = False,
) -> DeterminismSnapshot:
    """Apply and read back the locked CUDA reproducibility policy."""
    set_deterministic_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return DeterminismSnapshot(
        seed=seed,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        mode="topology_verdict" if warn_only else "strict",
    )


def content_hash(array: NDArray[np.generic]) -> str:
    """Вычислить SHA-256 с учётом dtype, shape и binary content array."""
    contiguous = np.ascontiguousarray(array)
    metadata = f"{contiguous.dtype.str}|{contiguous.shape}".encode()
    return hashlib.sha256(metadata + contiguous.tobytes()).hexdigest()


def artifact_sha256(path: Path) -> str:
    """Hash text through canonical LF and binary files through raw bytes."""
    payload = path.read_bytes()
    if path.suffix.lower() in _CANONICAL_TEXT_EXTENSIONS:
        text = payload.decode("utf-8")
        payload = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
