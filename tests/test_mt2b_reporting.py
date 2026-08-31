from __future__ import annotations

import numpy as np

from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.reporting.mt2b import (
    _architecture_figure,
    _conditioning_figure,
    _save_figure,
    geometry_stratum,
    illustrative_task_indices,
)


def _task(centers):
    return SourceLayoutTask(
        task_id="synthetic",
        centers=centers,
        bounds=((0.0, 0.1, 0.0, 0.1),) * 3,
        sources=np.zeros((3, 64, 64), dtype=np.float64),
    )


def test_geometry_stratum_uses_locked_prospective_thresholds() -> None:
    assert geometry_stratum(_task(((0.2, 0.6), (0.3, 0.7), (0.4, 0.8)))) == "compact"
    assert (
        geometry_stratum(_task(((0.2, 0.6), (0.4, 0.7), (0.66, 0.8))))
        == "wide_horizontal"
    )
    assert (
        geometry_stratum(_task(((0.2, 0.55), (0.3, 0.7), (0.4, 0.78))))
        == "vertically_spread"
    )
    assert geometry_stratum(_task(((0.2, 0.55), (0.4, 0.7), (0.66, 0.78)))) == "mixed"


def test_illustrative_indices_are_transparent_best_median_worst_ranks() -> None:
    gaps = np.arange(32, dtype=np.float64)[::-1]

    selected = illustrative_task_indices(gaps)

    assert selected == {"best": 31, "median": 15, "worst": 0}


def test_static_paper_figures_render_in_all_locked_formats(tmp_path) -> None:
    for stem, figure in (
        ("architecture", _architecture_figure()),
        ("conditioning", _conditioning_figure()),
    ):
        paths = _save_figure(figure, tmp_path, stem, title=stem)
        assert {path.suffix for path in paths} == {".png", ".svg", ".pdf"}
        assert all(path.stat().st_size > 1000 for path in paths)
