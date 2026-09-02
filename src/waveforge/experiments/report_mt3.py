"""Build the WaveForge MT3 development figure and report package."""

from __future__ import annotations

import argparse
from pathlib import Path

from waveforge.reporting.mt3 import MT3ReportPaths, build_mt3_development_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-temperature-maps", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    package = build_mt3_development_package(
        MT3ReportPaths(
            training_root=arguments.training_root.resolve(),
            evaluation_root=arguments.evaluation_root.resolve(),
            reference_root=arguments.reference_root.resolve(),
            output_root=arguments.output.resolve(),
        ),
        include_temperature_maps=not arguments.without_temperature_maps,
    )
    print(f"MT3_DEVELOPMENT_PACKAGE={package}", flush=True)


if __name__ == "__main__":
    main()
