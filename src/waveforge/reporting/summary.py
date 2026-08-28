"""Russian-language scientific summary writer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from waveforge.physics.validation import ValidationMetric


def write_gate1_report(
    metrics: Sequence[ValidationMetric],
    passed: bool,
    output_path: Path,
    *,
    config_hash: str,
) -> Path:
    """Записать Gate 1 numerical report без изменения metrics."""
    status = "PASS" if passed else "FAIL"
    failed = [metric for metric in metrics if not metric.passed]
    lines = [
        "# WaveForge Thermal — Gate 1 physics report",
        "",
        f"## Gate 1: {status}",
        "",
        f"Config hash: `{config_hash}`.",
        "",
        "Metrics вычислены до plotting и file I/O. SciPy reference solver "
        "использует cell-centered finite-volume flux form и harmonic face "
        "conductivity.",
        "",
        "## Validation metrics",
        "",
        "| Category | Metric | Grid | Value | Criterion | Status |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        criterion = (
            "informational"
            if metric.threshold is None
            else f"{metric.comparison} {metric.threshold:.8e}"
        )
        metric_status = "PASS" if metric.passed else "FAIL"
        lines.append(
            f"| {metric.category} | `{metric.name}` | {metric.grid} | "
            f"{metric.value:.8e} | {criterion} | {metric_status} |"
        )

    lines.extend(("", "## Blocking failures", ""))
    if failed:
        lines.extend(
            f"- `{metric.name}`: {metric.value:.8e} {metric.comparison} "
            f"{metric.threshold:.8e} — FAIL."
            for metric in failed
        )
    else:
        lines.append("Численные PASS/FAIL criteria выполнены.")

    lines.extend(
        (
            "",
            "## Scientific scope",
            "",
            "Этот отчёт валидирует только Gate 1 reference physics. Он не "
            "доказывает качество inverse design, не присваивает `128×128` "
            "статус high fidelity без Gate 2 comparison с `256×256` и не "
            "обосновывает применение ML surrogate.",
            "",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
