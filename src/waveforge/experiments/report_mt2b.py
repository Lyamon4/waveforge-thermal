"""Generate the frozen NCA-MT2B paper and presentation package."""

from __future__ import annotations

import argparse
from pathlib import Path

from waveforge.reporting.mt2b import MT2BReportPaths, generate_mt2b_paper_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    generate_mt2b_paper_package(
        MT2BReportPaths(
            training_root=arguments.training_root.resolve(),
            reference_root=arguments.reference_root.resolve(),
            evaluation_root=arguments.evaluation_root.resolve(),
            output_root=arguments.output.resolve(),
        )
    )


if __name__ == "__main__":
    main()
