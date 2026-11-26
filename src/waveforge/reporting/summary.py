"""Russian-language scientific summary writer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from waveforge.physics.validation import ValidationMetric
from waveforge.verification.compare import (
    CampaignVerdict,
    Gate2Status,
    SeedVerdict,
    classify_campaign,
)


def final_gate2a_verdict(
    nominal: dict[int, SeedVerdict],
    robustness: dict[int, SeedVerdict],
    *,
    mandatory_valid: bool,
    config_hash: str = "",
) -> CampaignVerdict:
    """Combine locked seed criteria with technical-invalidity precedence."""
    same_registry = set(nominal) == set(robustness) and bool(nominal)
    all_verdicts = (*nominal.values(), *robustness.values())
    valid = (
        mandatory_valid
        and same_registry
        and all(
            verdict.status is not Gate2Status.INVALID_RUN for verdict in all_verdicts
        )
    )
    passing_seeds = tuple(
        seed
        for seed in nominal
        if nominal[seed].status is Gate2Status.PASS
        and robustness[seed].status is Gate2Status.PASS
    )
    return classify_campaign(
        valid=valid,
        passing_seed_count=len(passing_seeds),
        required=2,
        metrics={
            "registered_seed_count": len(nominal),
            "passing_seeds": list(passing_seeds),
            "mandatory_valid": mandatory_valid,
        },
        config_hash=config_hash,
    )


def write_gate1_report(
    metrics: Sequence[ValidationMetric],
    passed: bool,
    output_path: Path,
    *,
    config_hash: str,
    benchmark_frame: pd.DataFrame | None = None,
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

    if benchmark_frame is not None:
        required_columns = {
            "solver",
            "resolution",
            "time_steps",
            "scenarios",
            "mode",
            "phase",
            "runs",
            "mean",
            "median",
            "p90",
            "std",
        }
        missing = required_columns.difference(benchmark_frame.columns)
        if missing:
            raise ValueError(f"benchmark frame missing columns: {sorted(missing)}")
        lines.extend(
            (
                "",
                "## Solver benchmark",
                "",
                "Times указаны в seconds; plotting и file I/O исключены из "
                "timed regions.",
                "",
                "| Solver | Grid | Steps | Scenarios | Mode | Phase | Runs | "
                "Median | P90 | Mean | Std |",
                "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
            )
        )
        for row in benchmark_frame.to_dict(orient="records"):
            lines.append(
                f"| {row['solver']} | {int(row['resolution'])}×"
                f"{int(row['resolution'])} | {int(row['time_steps'])} | "
                f"{int(row['scenarios'])} | {row['mode']} | {row['phase']} | "
                f"{int(row['runs'])} | {float(row['median']):.6e} | "
                f"{float(row['p90']):.6e} | {float(row['mean']):.6e} | "
                f"{float(row['std']):.6e} |"
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
