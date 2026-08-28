"""Проверки scientific table и summary serialization."""

from pathlib import Path

from waveforge.physics.validation import ValidationMetric, run_gate1_validation
from waveforge.reporting.summary import write_gate1_report
from waveforge.reporting.tables import metrics_frame


def test_metrics_frame_preserves_registered_criterion() -> None:
    metric = ValidationMetric(
        category="manufactured",
        name="relative_l2",
        grid="32x32",
        value=0.01,
        threshold=0.02,
        comparison="<=",
        passed=True,
    )
    frame = metrics_frame((metric,))

    assert list(frame.columns) == [
        "category",
        "name",
        "grid",
        "value",
        "threshold",
        "comparison",
        "passed",
    ]
    assert frame.to_dict(orient="records") == [
        {
            "category": "manufactured",
            "name": "relative_l2",
            "grid": "32x32",
            "value": 0.01,
            "threshold": 0.02,
            "comparison": "<=",
            "passed": True,
        }
    ]


def test_gate1_report_writes_russian_status_and_metric(tmp_path: Path) -> None:
    metric = ValidationMetric(
        category="operator",
        name="matrix_symmetry_max_abs",
        grid="5x4",
        value=0.0,
        threshold=1e-13,
        comparison="<=",
        passed=True,
    )
    output = tmp_path / "gate1_report.md"

    write_gate1_report((metric,), True, output, config_hash="abc123")

    text = output.read_text(encoding="utf-8")
    assert "Gate 1: PASS" in text
    assert "matrix_symmetry_max_abs" in text
    assert "abc123" in text


def test_gate1_runner_writes_required_numerical_artifacts(tmp_path: Path) -> None:
    """Artifact set должен строиться из одного immutable validation bundle."""
    artifact_dir = tmp_path / "gate1_physics"
    passed = run_gate1_validation(Path("configs"), artifact_dir)

    assert passed
    required = {
        "solver_config.json",
        "validation_metrics.csv",
        "convergence_plot.png",
        "linear_solution.png",
        "manufactured_solution_exact.png",
        "manufactured_solution_predicted.png",
        "manufactured_solution_error.png",
        "transient_convergence.gif",
        "gate1_report.md",
    }
    for name in required:
        output = artifact_dir / name
        assert output.is_file(), name
        assert output.stat().st_size > 0, name
