"""Static checks for resumable Vast.ai campaign supervision."""

from pathlib import Path


def test_vast_wrapper_runs_locked_cli_in_foreground() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "ops" / "vast" / "run_campaign_phase.sh").read_text(
        encoding="utf-8"
    )
    assert "set -euo pipefail" in script
    assert "exec .venv/bin/python -m waveforge.experiments.run_multitask_nca" in script
    assert '"$1"' in script


def test_supervisor_jobs_do_not_autorestart_scientific_failures() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("pilot", "production"):
        config = (root / "ops" / "vast" / f"waveforge-{name}.conf").read_text(
            encoding="utf-8"
        )
        assert "autostart=false" in config
        assert "autorestart=false" in config
        assert f" {name}" in config
