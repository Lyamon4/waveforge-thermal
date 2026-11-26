"""Проверки изоляции plotting от scientific metrics."""

from pathlib import Path

import numpy as np

from waveforge.reporting.figures import save_field_figure


def test_plotting_does_not_mutate_fields_or_metrics(tmp_path: Path) -> None:
    """In-place normalization ради картинки не должна менять result arrays."""
    field = np.arange(16, dtype=np.float64).reshape(4, 4)
    field_before = field.copy()
    metrics = {"relative_l2": 0.125}
    metrics_before = metrics.copy()
    output = tmp_path / "field.png"

    save_field_figure(field, output, title="fixture")

    np.testing.assert_array_equal(field, field_before)
    assert metrics == metrics_before
    assert output.is_file()
    assert output.stat().st_size > 0
