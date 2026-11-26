"""Explicit open-dev artifact allowlist and immutable input seals."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .stage3_training import task_from_dict

METHODS = (
    "Straight",
    "Uniform-Parallel",
    "Vascular-Adam",
    "Vascular-MMA",
    "Vascular-Adam600-Halo",
    "Vascular-Adam600-Decoupled",
    "Vascular-UNet-S31-R0",
    "Vascular-UNet-S31-R25",
)


def seal_files(paths):
    return {
        str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in paths
    }


def verify_seal(seal):
    for path, expected in seal.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"immutable artifact changed: {path}")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_inventory(root, selection):
    root = Path(root)
    selection = Path(selection)
    old = root / "artifacts/mcl_stage2_20261017/run_001"
    stage3 = root / "artifacts/mcl_stage3_20261018/run_001"
    run = root / "artifacts/mcl_stage3_20261018/run_002"
    manifest = _read(run / "manifest.json")
    verify_seal(
        {str(root / name): sha for name, sha in manifest["source_sha256"].items()}
    )
    verify_seal({str(selection): manifest["selection_sha256"]})
    tasks = tuple(task_from_dict(t) for t in _read(selection)["tasks"][:8])
    if [t.fingerprint() for t in tasks] != manifest["development_task_sha256"]:
        raise ValueError("open development tasks changed")
    for folder in (stage3, run):
        receipt = _read(folder / "completion.json")
        if not receipt.get("completed") or receipt.get("smoke_only"):
            raise ValueError("incomplete source run")
        verify_seal({str(folder / name): sha for name, sha in receipt["files"].items()})
    paths = [
        selection,
        run / "manifest.json",
        run / "completion.json",
        run / "freeze.json",
        run / "model_frozen.pt",
    ]
    paths.extend(root / name for name in manifest["source_sha256"])
    entries = []
    for task in range(8):
        for method in METHODS:
            if method in METHODS[:4]:
                folder = old / f"task_{task:02d}_{method}"
            elif method == "Vascular-Adam600-Halo":
                folder = stage3 / f"task_{task:02d}" / method
            else:
                folder = run / f"task_{task:02d}" / method
            report_path = folder / "result.json"
            report = _read(report_path)
            paths.append(report_path)
            if method in METHODS[:4]:
                receipt = _read(folder / "completion.json")
                verify_seal({str(folder / n): h for n, h in receipt["files"].items()})
            selected = report if method in METHODS[:2] else report.get("selected")
            if not selected or not selected["accepted"]:
                entries.append(
                    dict(task=task, method=method, source_accepted=False, path=None)
                )
                continue
            candidates = (
                [folder / "design.npz"]
                if method in METHODS[:2]
                else (
                    [folder / "selected.npz"]
                    if (folder / "selected.npz").exists()
                    else sorted(folder.glob("checkpoint_*.npz"))
                )
            )
            selected_path = None
            for candidate in candidates:
                with np.load(candidate, allow_pickle=False) as saved:
                    d = saved["binary"]
                    sha = hashlib.sha256(d.astype(np.uint8).tobytes()).hexdigest()
                if sha == selected["binary_sha256"]:
                    selected_path = candidate
                    break
            if selected_path is None:
                raise ValueError(f"selected binary missing: {folder}")
            paths.append(selected_path)
            entries.append(
                dict(
                    task=task,
                    method=method,
                    source_accepted=True,
                    path=str(selected_path),
                    binary_sha256=selected["binary_sha256"],
                    original_peak_rise_k=selected["peak_rise_k"],
                    original_flow_ml_s=selected["flow_ml_s"],
                )
            )
    return tasks, entries, seal_files(paths)
