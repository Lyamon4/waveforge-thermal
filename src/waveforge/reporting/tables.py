"""Tabular serialization for scientific metrics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from waveforge.physics.validation import ValidationMetric

METRIC_COLUMNS = [
    "category",
    "name",
    "grid",
    "value",
    "threshold",
    "comparison",
    "passed",
]


def metrics_frame(metrics: Sequence[ValidationMetric]) -> pd.DataFrame:
    """Преобразовать immutable metric records в stable table schema."""
    rows = [
        {
            "category": metric.category,
            "name": metric.name,
            "grid": metric.grid,
            "value": metric.value,
            "threshold": metric.threshold,
            "comparison": metric.comparison,
            "passed": metric.passed,
        }
        for metric in metrics
    ]
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def write_validation_metrics(
    metrics: Sequence[ValidationMetric],
    output_path: Path,
) -> Path:
    """Записать Gate 1 metrics CSV после завершения calculations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame(metrics).to_csv(output_path, index=False, float_format="%.17g")
    return output_path
