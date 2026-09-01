"""Create a frozen MT3 development verdict and sealed-test authorization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from waveforge.ml.mt3_evaluation import (
    MT3CheckpointSummary,
    build_test_authorization_bundle,
    classify_mt3_development,
    select_mt3_checkpoint,
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def freeze_development_verdict(
    *,
    summaries: list[MT3CheckpointSummary],
    implementation_commit: str,
    frozen_artifacts: dict[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    """Select the primary checkpoint, apply the gate, and hash authorization."""
    selected = select_mt3_checkpoint(summaries)
    verdict = classify_mt3_development(
        median_gap=selected.median_r25_relative_gap,
        p90_gap=selected.p90_r25_relative_gap,
        worst_gap=selected.worst_r25_relative_gap,
        wins=selected.r25_win_count,
        valid_count=selected.task_count - selected.invalid_count,
        exact_budget_count=selected.exact_budget_count,
    )
    _atomic_json(output_dir / "selected_checkpoint.json", asdict(selected))
    _atomic_json(output_dir / "development_verdict.json", asdict(verdict))
    bundle = build_test_authorization_bundle(
        verdict=verdict,
        implementation_commit=implementation_commit,
        artifacts=frozen_artifacts,
    )
    _atomic_json(output_dir / "test_authorization.json", bundle)
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    rows = json.loads(arguments.summaries.read_text(encoding="utf-8"))
    summaries = [MT3CheckpointSummary(**row) for row in rows]
    artifacts = {path.name: path.resolve() for path in arguments.artifact}
    freeze_development_verdict(
        summaries=summaries,
        implementation_commit=arguments.implementation_commit,
        frozen_artifacts=artifacts,
        output_dir=arguments.output.resolve(),
    )


if __name__ == "__main__":
    main()
