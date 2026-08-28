"""Сбор воспроизводимого manifest для native Windows environment."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil
import torch


def collect_environment(torch_install_command: str) -> dict[str, Any]:
    """Собрать platform, PyTorch и CUDA facts без смешения их терминов."""
    nvidia_smi = subprocess.run(
        ["nvidia-smi"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    cuda_available = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "windows_version": platform.version(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "bf16_supported": (
            torch.cuda.is_bf16_supported() if cuda_available else None
        ),
        "compute_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_available else None
        ),
        "nvidia_smi": nvidia_smi,
        "torch_install_command": torch_install_command,
        "pytorch_selector": {
            "build": "Stable 2.13.0",
            "os": "Windows",
            "package": "Pip",
            "language": "Python",
            "compute_platform": "CUDA 13.0",
            "url": "https://pytorch.org/get-started/locally/",
            "checked_at": "2026-08-28",
        },
    }


def write_environment_report(
    output_path: Path,
    torch_install_command: str,
) -> dict[str, Any]:
    """Записать environment manifest в стабильном JSON формате."""
    manifest = collect_environment(torch_install_command)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest

